#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""DecisionTransformer"""
import numpy as np
import torch
import torch.nn as nn

import transformers

from transformerGPT2 import GPT2Model

class TrajectoryModel(nn.Module):
    def __init__(self,state_dim,act_dim,max_length=None):
        super(TrajectoryModel, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length=max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self,states,actions,rewards,**kwargs):
        # come as tensors on the correct device
        return torch.zeros_like(actions[-1])

class DecisionTransformer(TrajectoryModel):

    """
    这个是基于Transformer
    Tokens: (Return1, state1, action1, Return2, state2, action2, ... )
    这里是Return不是单步的reward，而是从当前t时刻开始到最终T的累计奖励（回报）。
    """
    def __init__(self,
                 state_dim,
                 act_dim,
                 hidden_size,
                 max_length=None,
                 max_ep_len=4096, # 每个episode的最大timestep数
                 action_tanh=True,
                 **kwargs
                 ):
        super(DecisionTransformer, self).__init__(state_dim,act_dim,max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1, # 无关紧要，因为我们不用vocab
            n_embd=hidden_size,
            **kwargs
        )

        # 重写的GPT2Model与默认的Huggingface版本之间的区别就是移除了之前的
        # positional embedding，然后加入了Decision Transformer版本的形式
        self.transformer = GPT2Model(config)

        """For Offline Data"""
        # 对timestep进行embed
        # max_ep_len是有4096个timestep，转换成hidden_size维
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # 对累计回报Rt进行embed
        # 标量，len(Rt) = 1
        self.embed_return = torch.nn.Linear(1,hidden_size)
        # 对状态St进行embed
        self.embed_state = torch.nn.Linear(self.state_dim,hidden_size)
        # 对动作At进行embed
        self.embed_action = torch.nn.Linear(self.act_dim,hidden_size)

        # LayerNorm
        self.embed_ln = nn.LayerNorm(hidden_size)

        """动作决策"""
        # decorder的输出只输出action
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )


        self.predict_return = torch.nn.Linear(hidden_size,1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        """
        Input Dims:
        - States: (Bs x  max_seq_length x state_dim )
        """

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # 注意力的Mask
            # (bs,seq_len)
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # 多头分别给输入做embed
        state_embeddings = self.embed_state(states) # (Bs, seq_len, hidden_size)
        action_embeddings = self.embed_action(actions) #(Bs, seq_len, hidden_size)
        returns_embeddings = self.embed_return(returns_to_go) #(Bs, seq_len, hidden_size)
        time_embeddings = self.embed_timestep(timesteps) #(Bs, seq_len, time_step)

        # 时间信息嵌入(类比位置信息嵌入)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # 输入序列化(R1,S1,A1,R2,S2,A2,...,)
        stacked_inputs = torch.stack((returns_embeddings,state_embeddings,action_embeddings),dim=1)  #(Bs,3,seqlen,hid_size)
        stacked_inputs = stacked_inputs.permute(0,2,1,3) # (Bs,seq_len,3,hid_size)
        stacked_inputs = stacked_inputs.reshape(batch_size,3*seq_length, self.hidden_size)
        # LayerNorm
        stacked_inputs = self.embed_ln(stacked_inputs)

        # 使Attention mask和stacked inputs匹配
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0,2,1).reshape(batch_size,3*seq_length) # (Bs,3*seq_len)

        transformer_outputs = self.transformer(
            inputs_embeds = stacked_inputs, # shape(Bs, 3*seq_len, hidden_size)
            attention_mask = stacked_attention_mask,  #(Bs, 3*seq_len)
        )

        # x是Transformer的最后一个hidden_state输出
        x = transformer_outputs['last_hidden_state']
        # 重塑x，使得维度恢复原始维度 (Bs,3,seq_len,hidden_size)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # 获得预测
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self,states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim) # Dims(1, bs*seq_len, state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # 右最大length限制的条件下：
        if self.max_length is not None:
            # 取最后max_length
            # St-1, At-1, R_t-1, times
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # 将所有tokens进行pad到seq的长度
            # max_length > seq_length时，就需要进行Padding
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]),torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            # Pad states
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            # Pad actions
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            # Pad Rt
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            # Pad timesteps
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds,return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )

        # print(action_preds.shape)
        # Dim: (1,20,3)

        return action_preds[0,-1] # 3

"""MLP行为克隆方法"""
class MLPBCModel(TrajectoryModel):
    """
    Simple MLP
    输入past states: s
    预测next action: a
    """
    def __init__(self,state_dim,act_dim,hidden_size,n_layer,dropout=0.1,max_length=1,**kwargs):
        super(MLPBCModel, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [nn.Linear(max_length*self.state_dim,hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    def get_action(self,states,actions,rewards,**kwargs):
        states = states.reshape(1, -1, self.state_dim)
        # PADDING
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1] # 3


# debug
# if __name__ == "__main__":
#     a = torch.randn((1,20,3))
#     print(a[0,-1].shape)
#     state_embed = torch.randn((5,10,32))
#     time_embed = torch.randn((5,10,32))
#     action_embed = torch.randn((5, 10, 32))
#
#
#     c = torch.stack( (state_embed,time_embed,action_embed),dim=1)
#     print(c.shape)
#
#
#     # (1,2560,32)
#     state_embed = state_embed.reshape(1,-1,32)
#     print(state_embed.shape)
#     print(state_embed)
#
#     # 从第max_length到最后一个元素
#     state_embed = state_embed[:,-100:]
#     print(state_embed.shape)
#     print(state_embed)
#
#
#     print(c.shape)
