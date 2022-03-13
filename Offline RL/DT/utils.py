#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import gym
import os
import numpy as np
import collections
import pickle
import torch

import d4rl
import warnings
warnings.filterwarnings("ignore")

"""
工具说明: 打包benchmark数据
用于Offline RL的训练任务
"""

# 所使用的环境
# hopper / halfcheetch
env_name = 'hopper'
dataset_type = ['medium','expert','medium-replay']
h5path = "d4rl_dataset/"+f"{env_name}_{dataset_type[0]}-v2.hdf5"
# 可以手动下载
dataset_url = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"

def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)

def dataset_download(env_name,dataset_type,h5path):
    # 默认下载的路径：C:\Users\Irving123\.d4rl\datasets
    name = f'{env_name}-{dataset_type}-v2'
    env = gym.make(name)
    dataset = env.get_dataset(h5path)

    data_ = collections.defaultdict(list)

    N = dataset['rewards'].shape[0]
    print("数据集大小:",N)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []

    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i] # 超过给定的timesteps数也是traj
        else:
            final_timestep = (episode_step == 1000 - 1) # 手动指定timeout
        # 添加数据集
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
        # 一个episode done了，一条完整的Traj
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            # 再次重新初始化
            data_ = collections.defaultdict(list)
        episode_step += 1

    # 每个Traj的累计Rewards即为回报
    returns = np.array([np.sum(p['rewards']) for p in paths])
    # 每个Traj的sample的数量
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(paths, f)


# 计算回报（累计奖励）
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

# 模型保存
def model_save(record_name,model):
    if not os.path.exists("./Model/"+str(record_name)+"/"):
        os.makedirs("./Model/" + str(record_name) +"/", exist_ok=True)
    torch.save(model,"./Model/" + str(record_name) +"/"+str(record_name) + ".pth")
    print("\n +++Model Saved!++++ \n")

def load_model(model,path):
    model.load_state_dict(torch.load(path).state_dict())
    # print("=========Pre-train Model Loaded!===========")

if __name__ == "__main__":
    print(os.getcwd())

    # medium
    dataset_download(env_name,dataset_type[0],h5path)
    # medium_replay
    # dataset_download()
    # expert






