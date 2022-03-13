#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""Main入口For Decision Transformer"""
import gym
import numpy as np
import torch

# Tensorboard
from tensorboardX import SummaryWriter

# 读offline .pkl dataset
import pickle
import random
import sys

# DT
from model import DecisionTransformer
from train import SequenceTrainer
from eval import evaluate_episode_rtg

# %行为克隆
from model import MLPBCModel
from train import ActTrainer
from eval import evaluate_episode

from utils import discount_cumsum,model_save,load_model

import datetime

# 参数1 -  Decision-Transformer for Mujoco Gym
class Config:
    env = "hopper"
    dataset = "medium"
    mode = "normal" # "delayed" : all rewards moved to end of trajectory
    device = 'cuda'
    log_dir = 'TB_log/'
    record_algo = 'DT_Hopper_v1'
    test_cycles = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 模型
    model_type = "DT"
    activation_function = 'relu'

    # Scalar
    max_length = 20 # max_len # K
    pct_traj = 1.
    batch_size = 64
    embed_dim = 128
    n_layer = 3
    n_head = 1
    dropout = 0.1
    lr = 1e-4
    wd = 1e-4
    # warmup_steps = 1000
    warmup_steps=10
    # num_eval_episodes = 100
    num_eval_episodes = 10
    max_iters = 50
    # num_steps_per_iter = 1000
    num_steps_per_iter = 10

    # Bool
    log_to_tb = True

def main_dt(args=Config()):

    device = args.device
    if args.log_to_tb:
        writer = SummaryWriter(logdir=args.log_dir + args.record_algo + '_'+args.test_cycles)
        print("建立TB文件夹结束")

    env_name = args.env
    dataset = args.dataset

    dataset_path  = f'{env_name}-{dataset}-v2.pkl'
    print("===== Dataset Path: {} =====".format(dataset_path))

    # 以Hopper作为Test Benchmark
    if env_name == "hopper":
        env = gym.make("Hopper-v3")
        print("成功装载环境!")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print("Observation Space:", env.observation_space, "|| Dims: ", state_dim)
        print("Action Space:",env.action_space, "|| Dims: ", action_dim)
        max_ep_len = 1000

        env_targets = [3600, 1800]  # 预设的期望奖励值

        print(env_targets[:1])
        scale = 1000.  # 回报Scale Coeff.
    else:
        raise NotImplementedError

    # if args.model_type == 'BC': # 行为克隆算法模型
    env_targets = env_targets[:1] # since BC ignores target, no need for different evaluations

    # 数据集读取
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
        print("数据读取完毕... ...")

    # 将path info分别保存在对应的List内
    mode = args.mode
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed': # 此时回报移动到轨迹的最后
            """Delayed版本用于评估稀疏奖励下的算法表现
            前期没有任何Reward Signal，只有最后才会拿到一个总体的Reward. 
            """
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.

        states.append(path['observations']) # path['obs'] Dim: (XXX, state_dim) XXX: traj_len(每一个Traj的timesteps)
        traj_lens.append(len(path['observations'])) # path['rw'] Dim: (XXX, )
        returns.append(path['rewards'].sum()) # Returns 累计奖励----回报

    # Traj Lens Dims: (2186, )
    # Returns Lens Dims: (2186,)
    # 最大回报: 3222.36
    # 最小回报: 315.87
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # 输入正则化(State)
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    # num_timesteps:总共所有path的tp数之和
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = args.max_length
    batch_size = args.batch_size
    num_eval_episodes = args.num_eval_episodes
    pct_traj = args.pct_traj

    # 对于行为克隆，只在Top bct_traj 轨迹进行Train (For %BC Experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # low-to-high 排序
    num_trajectories = 1
    # 拿到分最高的那个traj的tp数
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1 # 按照Return从高到低

    # 输出是除了最小的Return之外的索引列表
    sorted_inds = sorted_inds[-num_trajectories:]

    # Reweight Sampling
    # 根据timesteps的长度的相对比重来进行sample
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size = batch_size,
            replace = True,
            p = p_sample, #依据timesteps采样
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]

            si = random.randint(0,traj['rewards'].shape[0] - 1)  # 采样位置

            # debug
            # print("Start sampling position: ", si)

            # get sequences from dataset
            # s
            # print(traj['observations'][si:si + max_len].reshape(1, -1, state_dim).shape) # shape: 1, 20, 11
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim)) # 1, XXX ,state_dim

            # a
            # print(traj['actions'][si:si + max_len].reshape(1, -1, action_dim).shape) # 1, 20, 3
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, action_dim))

            # r
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1)) # append(Dims: 1,20, 1)
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))

            # timesteps
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1)) # Append Dim: (1,20,1)
            # 检查是否有tps数量大于max_ep_len的情况  （max_ep_len=1000) 若有，按照max_ep_len-1来padding
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            # Return-to-go
            ret = discount_cumsum(traj['rewards'][si:],gamma=1.)
            # ret = ret[:s[-1].shape[1]+1].reshape(1,-1,1)
            ret = ret[:s[-1].shape[1]].reshape(1, -1, 1)
            rtg.append(ret) # Append(1,21,1) 是21>max_len

            if rtg[-1].shape[1] <= s[-1].shape[1]: # 21 & 20
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # print("timestep len: ", tlen) # 20
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            # print("Return-to-go:",rtg[-1].shape)

            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)

            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        # debug
        # print("="*40)
        # print("Dim s:", s.shape) # BS,20(max_len),1
        # print("Dim a:", a.shape)
        # print("Dim r:", r.shape)
        # print("Dim d:", d.shape)
        # print("Dim rtg:", rtg.shape) # BS,21(max_len+1),1
        # print("Dim timesteps:", timesteps.shape)
        # print("Dim mask:", mask.shape)
        # print("=" * 40)

        return s,a,r,d,rtg,timesteps,mask

    def eval_episodes(target_rew,log_tb=args.log_to_tb):
        def fn(model,log_tb=log_tb):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if args.model_type == 'DT':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            action_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            action_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)

            mean_returns = np.mean(returns)
            mean_tplen = np.mean(lengths)

            if log_tb:
                return   {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }, mean_tplen, mean_returns
            else:
                return {
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                }

        return fn

    if args.model_type == 'DT':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=action_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=args.embed_dim,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_inner=4*args.embed_dim,
            activation_function=args.activation_function,
            n_positions=1024,
            resid_pdrop=args.dropout,
            attn_pdrop=args.dropout,
        )
    elif args.model_type == 'BC':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=action_dim,
            max_length=K,
            hidden_size=args.embed_dim,
            n_layer=args.n_layer,
        )
    else:
        raise NotImplementedError

    # To Cuda
    model = model.to(device)

    # Warmup stage
    warmup_steps = args.warmup_steps

    # Optim
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    # Scheduler学习率优化
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    # Trainer 选择
    if args.model_type == 'DT':
        trainer = SequenceTrainer(
            model = model,
            optimizer = optimizer,
            batch_size = batch_size,
            get_batch = get_batch,
            scheduler = scheduler,
            loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            # eval_target第一个期望return-to-go
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    elif args.model_type == 'BC':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    for iter in range(args.max_iters):
        if args.log_to_tb:
            output,mean_ret,mean_len = trainer.train_iteration(
                num_steps = args.num_steps_per_iter,
                iter_num = iter+1,
                print_logs = True,
                TB_log = args.log_to_tb
            )
            # tb writer
            writer.add_scalar(tag='DT/mean_return',global_step=iter,scalar_value=mean_ret)
            writer.add_scalar(tag='DT/mean_len', global_step=iter, scalar_value=mean_len)
            writer.add_scalar(tag='DT/mean_mse_a',global_step=iter, scalar_value=output['training/train_loss_mean'])
            writer.add_scalar(tag='DT/std_mse_a', global_step=iter, scalar_value=output['training/train_loss_std'])
            # print("成功写入!")

            model_save(args.record_algo+args.test_cycles,model)


        else:
            output = trainer.train_iteration(
                num_steps = args.num_steps_per_iter,
                iter_num = iter+1,
                print_logs = True,
                TB_log = args.log_to_tb
            )



if __name__ == "__main__":
    DTargs = Config()
    main_dt(args=DTargs)

    # rewards = np.array([1,2,3,4,5])
    # print(rewards.sum())
    #
    # # 读取数据可视化
    # dataset_path = 'hopper-medium-v2.pkl'
    # with open(dataset_path, 'rb') as f:
    #     trajectories = pickle.load(f)
    #     print("数据读取完毕... ...")
    #
    # returns = []
    # for path in trajectories:
    #     print(path['observations'].shape)
    #     print(path['rewards'].shape)
    #     returns.append(path['rewards'].sum())
    #
    # print(np.array(returns).shape)

    # Reward排序
    # returns = np.array([12,333,445,6,789,991,23,76])
    # traj_lens = np.array([7,8,9,20,15,12,16,17])
    # sorted_inds = np.argsort(returns)
    # print(sorted_inds)  # [3 0 6 7 1 2 4 5] #对应回报最大的索引排序...
    # num_traj = 1
    # # 最大Return对应的tp数
    # timesteps = traj_lens[sorted_inds[-1]] # 12 991
    # print(timesteps) # 12个tps获得991的累计return
    #
    # num_trajectories = 1
    #
    # len_trajs = len(returns)
    #
    # ind = len_trajs - 2
    #
    # num_timesteps = traj_lens.sum()
    #
    # while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
    #     timesteps += traj_lens[sorted_inds[ind]]
    #     num_trajectories += 1
    #     ind -= 1
    #
    # # 输出是除了最小的Return之外的索引列表
    # sorted_inds = sorted_inds[-num_trajectories:]
    # print(sorted_inds)



