#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
C51算法Step By Step
https://github.com/Kchu/DeepRL_PyTorch/blob/master/Distributional_RL/1_C51.py
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import ReplayBuffer
from utils import PrioritizedReplayBuffer

from tensorboardX import SummaryWriter

import random
import os
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt
from wrapper import wrap, wrap_cover, SubprocVecEnv

from Config import Config_C51
from Network import ConvNet


class C51:
    def __init__(self,args,Priority=None):

        self.pred_net = ConvNet(args.N_STATES, args.N_ACTIONS, args.N_ATOM)
        self.target_net = ConvNet(args.N_STATES, args.N_ACTIONS, args.N_ATOM)
        self.args = args

        # GPU
        if args.USE_GPU:
            self.pred_net.to(args.device)
            self.target_net.cuda(args.device)

        # Sim Step Counter
        self.memory_counter = 0

        # target network step counter
        self.learn_step_counter = 0

        # ceate the replay buffer
        self.replay_buffer = ReplayBuffer(args.MEMORY_CAPACITY)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=args.LR)

        # discrete values
        self.value_range = torch.FloatTensor(args.V_RANGE) # (N_ATOM)
        if args.USE_GPU:
            self.value_range = self.value_range.to(args.device)

    def update_target(self,target,pred,update_rate):
        # update target network parameters using predcition network
        # Soft Update
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) * target_param.data + update_rate*pred_param.data)

    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(self.args.PRED_PATH)
        self.target_net.save(self.args.TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(self.args.PRED_PATH)
        self.target_net.load(self.args.TARGET_PATH)

    def choose_action(self, x, EPSILON):
        x = torch.FloatTensor(x)
        if self.args.USE_GPU:
            x = x.cuda()

        # Epsilon Greedy
        if np.random.uniform() >= EPSILON:
            # greedy case
            action_value_dist = self.pred_net(x) # (N_ENVS, N_ACTIONS, N_ATOM)
            action_value = torch.sum(action_value_dist * self.value_range.view(1, 1, -1), dim=2) # (N_ENVS, N_ACTIONS)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, self.args.N_ACTIONS, (x.size(0)))
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # Target Net Params Update
        if self.learn_step_counter % self.args.TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)

        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(self.args.BATCH_SIZE)
        b_w, b_idxes = np.ones_like(b_r), None

        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_s_ = torch.FloatTensor(b_s_)

        if self.args.USE_GPU:
            b_s, b_a, b_s_ = b_s.cuda(), b_a.cuda(), b_s_.cuda()

        # action value distribution prediction
        q_eval = self.pred_net(b_s) # (m, N_ACTIONS, N_ATOM)
        mb_size = q_eval.size(0)

        # index_select: 选择在维度0上，按照b_a[i]的index来选择相应的q_evals
        q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1) # (m, N_ATOM)

        # Target Distribution
        q_target = np.zeros((mb_size, self.args.N_ATOM)) # (m, N_Atom)

        # get next state value
        q_next = self.target_net(b_s_).detach() # (m, N_ACTIONS, N_ATOM)

        # next value mean (TD Target)
        """
        C51中，计算a*:
        - Q(s',a) = Sum(zi*pi(s',a))
        - a* = argmaxQ(s',a) 
        """
        q_next_mean = torch.sum(q_next * self.value_range.view(1, 1, -1), dim=2) # (m, N_ACTIONS)
        best_actions = q_next_mean.argmax(dim=1) # (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        q_next = q_next.data.cpu().numpy()  # (m, N_ATOM)

        # Categorical Projection
        '''
        next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
        next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
        '''
        # we vectorized the computation of support and position
        next_v_range = np.expand_dims(b_r, 1) + self.args.GAMMA * np.expand_dims((1. - b_d),1) \
        * np.expand_dims(self.value_range.data.cpu().numpy(),0)
        next_v_pos = np.zeros_like(next_v_range)
            # clip for categorical distribution
        next_v_range = np.clip(next_v_range, self.args.V_MIN, self.args.V_MAX)
        # calc relative position of possible value
        next_v_pos = (next_v_range - self.args.V_MIN)/ self.args.V_STEP
        # get lower/upper bound of relative position
        lb = np.floor(next_v_pos).astype(int)
        ub = np.ceil(next_v_pos).astype(int)
        # we didn't vectorize the computation of target assignment.
        for i in range(mb_size):
            for j in range(self.args.N_ATOM):
                # calc prob mass of relative position weighted with distance
                q_target[i, lb[i,j]] += (q_next * (ub - next_v_pos))[i,j]
                q_target[i, ub[i,j]] += (q_next * (next_v_pos - lb))[i,j]

        q_target = torch.FloatTensor(q_target)
        if self.args.USE_GPU:
            q_target = q_target.cuda()

        # calc huber loss, dont reduce for importance weight
        # Cross Entropy Loss :
        # -sum_x{p(x)log(q(x))}
        loss = q_target * (- torch.log(q_eval + 1e-8))  # (m , N_ATOM)
        loss = torch.mean(loss)

        # calc importance weighted loss
        b_w = torch.Tensor(b_w)
        if self.args.USE_GPU:
            b_w = b_w.cuda()
        loss = torch.mean(b_w * loss)

        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def C51main(args,Gameenv):

    env = Gameenv
    C51_Model = C51(args)

    # model load with check
    if args.LOAD and os.path.isfile(args.PRED_PATH) and os.path.isfile(args.TARGET_PATH):
        C51_Model.load_model()
        pkl_file = open(args.RESULT_PATH, 'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()
        print('Load complete!')
    else:
        result = []
        print('Initialize results!')

    print('Collecting experience...')

    # episode step for accumulate reward
    epinfobuf = deque(maxlen=100)
    # check learning time
    start_time = time.time()

    # env reset
    s = np.array(env.reset())

    for step in range(1,args.STEP_NUM//args.N_ENVS+1):
        a = C51_Model.choose_action(s,args.EPSILON)

        # take action and get next state
        s_, r, done, infos = env.step(a)

        # log arrange
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo: epinfobuf.append(maybeepinfo)

        s_ = np.array(s_)

        # clip rewards for numerical stability
        clip_r = np.sign(r)

        # store the transition
        for i in range(args.N_ENVS):
            C51_Model.store_transition(s[i], a[i], clip_r[i], s_[i], done[i])

        # annealing the epsilon(exploration strategy)
        if step <= int(1e+3):
            # linear annealing to 0.9 until million step
            args.EPSILON -= 0.9 / 1e+3
        elif step <= int(1e+4):
            # linear annealing to 0.99 until the end
            args.EPSILON -= 0.09 / (1e+4 - 1e+3)

        # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
        if (args.LEARN_START <= C51_Model.memory_counter) and (C51_Model.memory_counter % args.LEARN_FREQ == 0):
            C51_Model.learn()

        # print log and save
        if step % args.SAVE_FREQ == 0:
            # check time interval
            time_interval = round(time.time() - start_time, 2)
            # calc mean return
            mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
            result.append(mean_100_ep_return)
            # print log
            print('Used Step:', C51_Model.memory_counter,
                  'EPS: ', round(args.EPSILON, 3),
                  '| Mean ep 100 return: ', mean_100_ep_return,
                  '| Used Time:', time_interval)
            # save model
            C51_Model.save_model()
            pkl_file = open(args.RESULT_PATH, 'wb')
            pickle.dump(np.array(result), pkl_file)
            pkl_file.close()

        s = s_

        if args.RENDERING:
            env.render()

    print("Training Done ... ")

def main(args=Config_C51()):
    # openai gym env name
    ENV_NAME = args.GAME + 'NoFrameskip-v4'
    # ENV_NAME = GAME + '-v0'
    # 进程池相关代码必须在Main中才可以...
    # env = SubprocVecEnv([wrap_cover("BreakoutNoFrameskip-v4") for i in range(args.N_ENVS)])
    # N_ACTIONS = env.action_space.n
    # N_STATES = env.observation_space.shape
    #
    # args.N_ACTIONS = N_ACTIONS # 4
    # args.N_STATES = N_STATES  # 4 x 84 x 84
    # print(args.N_STATES)
    # print(args.N_ACTIONS)

    """
    多进程Error: in _recv_bytes raise EOFError EOFError
    https://its301.com/article/qxqxqzzz/104384920
    """
    # 取消注释直接执行
    # ----------------------------------
    # C51main(args,env)
    # ----------------------------------

    N_STATES = (4,84,84)
    N_ACTIONS = 4
    pred_net = ConvNet(N_STATES,N_ACTIONS,args.N_ATOM)
    target_net = ConvNet(N_STATES,N_ACTIONS,args.N_ATOM)

    # Sim b_s
    b_s = np.random.rand(32,4,84,84)
    b_s = torch.FloatTensor(b_s)
    print(b_s.shape)

    # Sim b_s_
    b_s_ = np.random.rand(32,4,84,84)
    b_s_ = torch.FloatTensor(b_s_)

    b_a = np.ones((32,)) # 0,1,2,3 动作 我们只取1
    b_a = torch.LongTensor(b_a)
    print(b_a.shape)

    # Sim q_eval
    q_eval = pred_net(b_s) # [32,4,51]
    print(q_eval.shape)
    mb_size = q_eval.size(0) # bs (32)
    print("mb_size:",mb_size)

    """
    q_eval_list = []
    for i in range(mb_size):
        # print("current i:", i)
        # print("current b_a[i]",b_a[i])
        q_eval[i].index_select(0,b_a[i]) # 获得对应action组合，e.g. (0,1,2,3)、(0,1,0,1)的q_eval(在第0维) # 4,51
        q_eval_list.append(q_eval[i])

    print(len(q_eval_list)) #32
    q_eval = torch.stack(q_eval_list).squeeze(1) # 
    print(q_eval.shape)  
    """
    """
    q:output: 
    -------------------------
        z0(Vmin), z1, z2, z3, ... zN(Vmax)(m_atoms)
    a0: q1       q2  q3  q4
    a1: q1'      q2' q3' q4' 
    a2: q1`      q2` qe` q4` 
    (n_actions) 
    ------------------------- 
    
    b_a[i] 第i个batch对应的a 
    index_select: 查 q_eval[i]中，这个action对应的q值 
    """
    q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1)
    print(q_eval.shape) #32,51

    q_target = np.zeros((mb_size, args.N_ATOM)) # (bs, N_ATOM)
    # get next state value
    q_next = target_net(b_s_).detach()  # (m, N_ACTIONS, N_ATOM)
    # next value mean
    print(args.V_RANGE)
    value_range = torch.FloatTensor(args.V_RANGE)
    print(value_range.shape) #(51,)
    # value_range = value_range.view(1,1,-1)
    # print(value_range.shape) #(1,1,51)
    # 求a*
    q_next_mean = torch.sum(q_next * value_range.view(1,1,-1), dim=2)  # (m, N_ACTIONS)  -> # sum_{i=0~N}{zi*pi}
    best_actions = q_next_mean.argmax(dim=1)
    print(best_actions.shape)
    print("best_actions:",best_actions) #(bs)

    # td_target
    # Seek
    q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
    print("q_next.shape:",q_next.shape) # (bs,atoms)
    q_next = q_next.data.cpu().numpy()  # (m, N_ATOM)

    """
    Categorical projection 
        next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
        next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
    z_{j+1} = r + gamma * z_j 
    按距离做  bellman update projection
    """
    b_r = np.random.rand(32,)
    b_w, b_idxes = np.ones_like(b_r), None

    print(b_w)
    next_v_range = np.expand_dims(b_r,1) + args.GAMMA * np.expand_dims(value_range.data.cpu().numpy(),0)
    print(next_v_range.shape) # (32,51)
    next_v_pos = np.zeros_like(next_v_range)  # (32,51)
    # 1. Clip
    next_v_range = np.clip(next_v_range, args.V_MIN, args.V_MAX)
    # 2. 计算相对位置
    next_v_pos = (next_v_range - args.V_MIN) / args.V_STEP
    # 3. 获得相对位置的上、下界
    lb = np.floor(next_v_pos).astype(int)
    print(lb.shape) #(32,51)
    print(lb)
    ub = np.ceil(next_v_pos).astype(int)
    print(ub)
    print(ub.shape) #(32,51)
    # 4. Target Assignment
    for i in range(mb_size):
        for j in range(args.N_ATOM):
            # print(lb[i,j])
            # calc prob mass of relative position weighted with distance
            q_target[i, lb[i, j]] += (q_next * (ub - next_v_pos))[i, j]
            q_target[i, ub[i, j]] += (q_next * (next_v_pos - lb))[i, j]











if __name__ == "__main__":
    main()
    # UnitsTest()