#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
分布式架构
"""

import copy
from datetime import datetime
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp
import queue
from time import sleep

import os

from Networks import PolicyNetwork
from Networks import C51ValueNetwork
from utils import Logger
from utils import empty_torch_queue
from utils import create_replay_buffer

from AgentWorker import Agent
from D4PG_LearnerWorker import D4PGLearnerWorker as LearnerD4PG

class DistributedEngine(object):
    def __init__(self,config):
        self.config = config

    def train(self):
        config = self.config

        batch_queue_size = config['batch_queue_size']
        n_agents = config['num_agents']  # Nums of Actors

        # 创建实验的目录
        experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H%M%S}"
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir,exist_ok=True)

        # 数据结构
        Processes = []
        replay_queue = mp.Queue(maxsize=config['replay_queue_size'])  # FIFO QUEUE
        learner_w_queue = torch_mp.Queue(maxsize=n_agents)
        replay_priorities_queue = mp.Queue(maxsize=config['replay_queue_size'])

        # 用于多进程共享的状态变量
        training_on = mp.Value('i',1)
        update_step = mp.Value('i',0)
        global_episode = mp.Value('i',0) # 共享变量

        # Data Sampler -- 采样进程
        # (1)
        batch_queue = mp.Queue(maxsize=batch_queue_size)
        p = torch_mp.Process(target=sampler_worker,
                             args=(config, replay_queue, batch_queue, replay_priorities_queue, training_on,
                                   global_episode, update_step, experiment_dir))
        Processes.append(p)

        # Learner -- 学习进程
        # (2)
        policy_net = PolicyNetwork(config['state_dim'], config['action_dim'],
                                          config['dense_size'], device=config['device'])
        target_policy_net = copy.deepcopy(policy_net)
        policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'],
                                          config['dense_size'], device=config['agent_device'])

        # 记住每次将Tensor放入multiprocessing.Queue时，必须将其移动到共享内存中
        target_policy_net.share_memory()
        p = torch_mp.Process(target=learner_worker, args=(config, training_on, policy_net, target_policy_net, learner_w_queue,
                                                          replay_priorities_queue, batch_queue, update_step, experiment_dir))
        Processes.append(p)

        # Single Agent For "利用"
        # (3)
        p = torch_mp.Process(target=agent_worker,
                             args=(config, target_policy_net, None, global_episode, 0, "exploitation", experiment_dir,
                                   training_on, replay_queue, update_step))
        Processes.append(p)

        # (4,5,6,7)
        # Agents(探索进程) "探索"
        for i in range(n_agents):
            p = torch_mp.Process(target=agent_worker,
                                 args=(config, copy.deepcopy(policy_net_cpu), learner_w_queue, global_episode,
                                       i+1, "exploration", experiment_dir, training_on, replay_queue, update_step))
            Processes.append(p)

        for p in Processes:
            p.start()

        for p in Processes:
            p.join()

        print("-----[End...]----- ")

"""[Sampling worker]"""
def sampler_worker(config,replay_queue, batch_queue, replay_priorities_queue, training_on, global_episode, update_step,log_dir=''):

    batch_size = config['batch_size']
    logger = Logger(f"{log_dir}/data_struct")

    # Create replay buffer
    replay_buffer = create_replay_buffer(config)

    while training_on.value: # 共享变量
        # Step1.将replays迁移到Global buffer
        """图中: [replay_queue] --(s,a,r,s',d)--> [replay_buffer]"""
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # Step2.将replay buffer中的数据迁移到batch_queue，并过呢更新replay_priority_queue中的权重
        if len(replay_buffer) < batch_size:
            # 超过batch_size才开始后面的sample
            continue

        """图中:[batch_queue]--(transition priority)-->[replay_priority_queue]"""
        try:
            # 更新各条Transition权重
            # get_nowait(): Remove and return an item from the queue without blocking.
            inds, weights = replay_priorities_queue.get_nowait()
            replay_buffer.update_priorities(inds,weights)
        except queue.Empty:
            pass

        """图中:[replay_buffer]--(batch sample)-->[batch_queue]"""
        try:
            batch = replay_buffer.sample(batch_size)
            batch_queue.put_nowait(batch)
        except:
            sleep(0.1)
            continue

        # 记录Data Structures Sizes:
        step = update_step.value
        logger.scalar_summary("data_struct/global_episode", global_episode.value, step)
        logger.scalar_summary("data_struct/replay_queue", replay_queue.qsize(), step)
        logger.scalar_summary("data_struct/batch_queue", batch_queue.qsize(), step)
        logger.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    if config['save_buffer_on_disk']:
        replay_buffer.dump(config['results_path'])

    empty_torch_queue(batch_queue)
    print("-----[Stop Sampler worker...]-----")

"""[Learner Worker]"""
def learner_worker(config,training_on,policy,target_policy_net,learner_w_queue, replay_priority_queue,batch_queue,update_step,experiment_dir):
    learner = LearnerD4PG(config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    learner.run(training_on, batch_queue, replay_priority_queue, update_step)

"""[Evaluation Worker(no noise)] & [Exploration Worker]"""
def agent_worker(config, policy, learner_w_queue, global_episode, i, agent_type,
                 experiment_dir, training_on, replay_queue, update_step):
    agent = Agent(config,
                  policy=policy,
                  global_episode=global_episode,
                  n_agent=i,
                  agent_type=agent_type,
                  log_dir=experiment_dir)
    agent.run(training_on, replay_queue, learner_w_queue, update_step)


def load_engine(config):
    print(f"Loading {config['model']} for {config['env']}.")
    if config["model"] == "d4pg":
        return DistributedEngine(config)
    if config["model"] in ["ddpg", "d3pg"]:
        pass


# 测试进程共享变量
# def func1(a,arr):
#     a.value = 3.14
#     for i in range(len(arr)):
#         arr[i] = -arr[i]

# if __name__ == "__main__":
#     # Value Array 是通过共享内存的方式共享数据
#     # Manager是通过共享进程的方式共享数据
#     # import multiprocessing
#     # num = multiprocessing.Value('d',1.0)
#     # arr = multiprocessing.Array('i',range(10))
#     # p = multiprocessing.Process(target=func1,args=(num,arr))
#     # p.start()
#     # p.join()
#     # print(num.value) # 3.14
#     # print(arr[:]) # [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
#     pass