#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import numpy as np
import torch
import time

class Trainer:
    """Basic训练器"""
    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns

        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self,num_steps,iter_num=0,print_logs=True,TB_log=False):
        train_losses = []
        logs = dict()

        train_start = time.time()

        # 开始训练
        self.model.train()
        for tp in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            print("第 {} 个 step, train loss MSE(a_pred,a_tar): {}".format(tp,train_loss))
            # 学习率退火？
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        eval_start = time.time()

        # eval
        self.model.eval()
        for eval_fn in self.eval_fns:
            if not TB_log:
                outputs = eval_fn(self.model,log_tb=TB_log)
            else:
                outputs, mean_tplen, mean_returns = eval_fn(self.model,log_tb=TB_log)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]


        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        if not TB_log:
            return logs
        else:
            return logs,mean_returns,mean_tplen

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks = None,
            attention_mask = attention_mask,
            target_return = returns # 第一个return-to-go是期望达到重点的return，比如没有到达终点reward=0，到达终点reward=1;
        )

        # 注意target检索需要进行更改
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

class ActTrainer(Trainer):
    """
    For MLP行为克隆
    """
    def train_step(self):
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:, 0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:, -1].reshape(-1, act_dim)

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

class SequenceTrainer(Trainer):
    """序列式训练方式
    for Decision Transformer
    """
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        ) # MSE(action_preds - action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()