#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import d4rl
import gym
import time
import datetime
import os
import warnings
from termcolor import colored

from Utils import ReplayBuffer
from Utils import count_vars
from Utils import Logger

from Networks import AWActorCritic
from Networks import DEVICE
from tqdm import tqdm

class Config:
    alpha = 0.00
    seed = 0
    steps_per_epoch = 100
    epochs = 1200
    replay_size = int(2000000)
    gamma = 0.99

    # Interpolation Factor in Polyak Averaging for Target Network
    # Soft Update Coeff.
    polyak = 0.995

    lr = 3e-4
    p_lr = 3e-4
    batch_size = 1024
    start_steps = 10000
    update_after = 0
    update_every = 50
    num_test_episodes = 10
    max_ep_len = 1000
    logger_kwargs = dict()
    save_freq = 1
    algo = 'AWAC'
    awac_hidden_sizes = (256, 256, 256, 256)
    activation = nn.ReLU

    logdir = "./logdir/MAAC_HalfCheetah_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


class AWAC:
    def __init__(self, env_fn, actor_critic=AWActorCritic, args=Config()):

        # Logger
        self.logger = Logger(args.logdir)
        self.global_test_counter = 0

        # Seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Env
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space,special_policy='awac')
        self.ac_targ = actor_critic(self.env.observation_space, self.env.action_space,special_policy='awac')
        self.ac_targ.load_state_dict(self.ac.state_dict())
        # Target Freeze
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        # chain('ABC', 'DEF') --> A B C D E F
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        self.gamma = args.gamma

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,size=args.replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        self.algo = args.algo
        #Lr
        self.p_lr = args.p_lr
        self.lr = args.lr
        #Entropy Weights
        self.alpha = args.alpha # Temperature Coeffs.

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

        self.num_test_episodes = args.num_test_episodes
        self.max_ep_len = args.max_ep_len
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.update_after = args.update_after
        self.update_every = args.update_every
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq
        self.polyak = args.polyak

        # Model Saving

        print("Running Offline RL algorithm: {}".format(self.algo))

    def populate_replay_buffer(self,env_name):
        """填充ReplayBuffer With Mixture Offline Data"""
        data_envs = {
            'HalfCheetah-v2': (
                "awac_data/hc_action_noise_15.npy",
                "awac_data/hc_off_policy_15_demos_100.npy"),
            'Ant-v2': (
                "awac_data/ant_action_noise_15.npy",
                "awac_data/ant_off_policy_15_demos_100.npy"),
            'Walker2d-v2': (
                "awac_data/walker_action_noise_15.npy",
                "awac_data/walker_off_policy_15_demos_100.npy"),
        }
        if env_name in data_envs:
            print('Loading saved data')
            for file in data_envs[env_name]:
                if not os.path.exists(file):
                    warnings.warn(colored('Offline data not found. Follow awac_data/instructions.txt to download. Running without offline data.', 'red'))
                    break
                data = np.load(file, allow_pickle=True)
                for demo in data:
                    # 两个，专家数据和Suboptimal的数据
                    for transition in list(zip(demo['observations'], demo['actions'], demo['rewards'],
                                               demo['next_observations'], demo['terminals'])):
                        self.replay_buffer.store(*transition)
        else:
            dataset = d4rl.qlearning_dataset(self.env)
            N = dataset['rewards'].shape[0]
            for i in range(N):
                self.replay_buffer.store(dataset['observations'][i], dataset['actions'][i],
                                         dataset['rewards'][i], dataset['next_observations'][i],
                                         float(dataset['terminals'][i]))
            print("Loaded dataset")

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            # 注意是用Current Policy的预测...
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        #               Q2Vals=q2.detach().numpy())
        q_info = {}

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        beta = 2
        q1_old_actions = self.ac.q1(o, data['act'])
        q2_old_actions = self.ac.q2(o, data['act'])
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)

        adv_pi = q_old_actions - v_pi
        weights = F.softmax(adv_pi / beta, dim=0) # 带Z
        # weights = torch.exp(adv_pi) # 不带Z
        policy_logpp = self.ac.pi.get_logprob(o, data['act'])
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        # Useful info for logging
        # pi_info = dict(LogPi=policy_logpp.detach().numpy())
        pi_info = {}

        return loss_pi, pi_info

    def update(self,data, update_timestep):
        # First run one gd step for Q1, Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        """Record: Loss Q"""
        # self.logger.scalar_summary()

        """Policy Learning阶段冻结QNets梯度"""
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        """Record: Loss Pi"""
        # self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        obs = torch.as_tensor(o, dtype=torch.float32).to(DEVICE)
        # print(obs.device)
        return self.ac.act(obs, deterministic)

    def test_agent(self):
        self.global_test_counter += 1
        bar = tqdm(np.arange(0,self.num_test_episodes))
        # for j in range(tqdm(self.num_test_episodes)):
        sum_eps = 0
        sum_tp = 0
        for j in bar:
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            bar.set_description("[Eval] Current Eps %d, Ret %.2f, TpLen %d"%(j, ep_ret, ep_len))
            """Record Ep_ret & Ep_Len"""

            sum_eps += ep_ret
            sum_tp += ep_len
        print("[Eval] Test Mean Return: %.2f, Test Mean Tp Consumption: %d \n"%(sum_eps/self.num_test_episodes,sum_tp/self.num_test_episodes))
        self.logger.scalar_summary("Test/AvgRet",sum_eps/self.num_test_episodes,self.global_test_counter)
        self.logger.scalar_summary("Test/AvgTp", sum_tp / self.num_test_episodes, self.global_test_counter)

    def run(self):
        # Prepare for interaction with env
        total_steps = self.epochs * self.steps_per_epoch
        start_time = time.time()
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        done = True
        num_train_episodes = 0

        # Main loop: collect experience in env and update/log each epoch
        bar = tqdm(np.arange(0,total_steps))
        # for t in range(tqdm(total_steps)):
        for t in bar:
            # Reset stuff if necessary
            if done and t > 0:
                bar.set_description("Current Eps %d, Ret %.2f, TpLen %d"%(num_train_episodes, ep_ret, ep_len))
                self.logger.scalar_summary("Trainer/Ret", ep_ret, num_train_episodes)
                self.logger.scalar_summary("Trainer/Tp", ep_len, num_train_episodes)
                """记录Training过程中的 Ep_ret & Ep_Len """
                # self.logger.store(ExplEpRet=ep_ret, ExplEpLen=ep_len)
                obs, ep_ret, ep_len = self.env.reset(), 0, 0
                num_train_episodes += 1

                # Test一下
                self.test_agent()

            # Collect experience
            act = self.get_action(obs, deterministic=False)

            next_obs, rew, done, info = self.env.step(act)

            ep_ret += rew
            ep_len += 1

            self.replay_buffer.store(obs, act, rew, next_obs, done) # 将当前策略采样数据添加到ReplayBuffer
            obs = next_obs

            self.logger.scalar_summary("Trainer/TpRet", ep_ret, t)
            # Update Handling
            if t > self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch, update_timestep=t)

            # End of Epoch Handling ...
            # if (t + 1) % *self.steps_per_epoch == 0:
            #     epoch = (t + 1) // self.steps_per_epoch

                # Save model
                # if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                #     self.logger.save_state({'env': self.env}, None)
                # Test the performance of the deterministic version of the agent.



if __name__ == "__main__":
    env_name = "HalfCheetah-v2"
    # env = gym.make(env_name)
    # print(env.action_space.high)
    env_fn = lambda : gym.make(env_name)
    agent = AWAC(env_fn)

    # 填充ReplayBuffer
    agent.populate_replay_buffer(env_name)
    agent.run()
