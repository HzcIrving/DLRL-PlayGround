#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn
import torch
import numpy as np


# 推理用函数
# 1- episode --- MLPBC
# 2- return-to-go --- DT

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # 环境初始化
    state = env.reset()

    # 我们将所有的histories都加载到device之上；
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32) # [] 初始化
    rewards = torch.zeros(0, device=device, dtype=torch.float32) # []  初始化

    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # 添加padding后的Mask
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            # 状态标准化
            # 将actions与rewards进行pad，方便预测当前的action
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        # 交互
        state, reward, done, _ = env.step(action)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # 环境初始化
    state = env.reset()

    # 带噪声？
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # 我们将所有的histories都加载到device之上；
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32) # [] 初始化
    rewards = torch.zeros(0, device=device, dtype=torch.float32) # []  初始化

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # 添加padding
        # 注意到 latest action以及reward都会被 " padding " 上
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            # 状态标准化
            # 将actions与rewards进行pad，方便预测当前的action
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long)
        )
        # print(action)

        actions[-1] = action # Dim [3]

        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        # 向episode中添加最新的states、rewards
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # 这里，每次推理一步，就用上一步的return-to-go减去这一步的reward；
        # 相当于t+1步的target_return是: pred_return
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


# if __name__ == "__main__":
#     a = torch.zeros((0,3))
#     a = torch.zeros((1, 3))
#     print(a)