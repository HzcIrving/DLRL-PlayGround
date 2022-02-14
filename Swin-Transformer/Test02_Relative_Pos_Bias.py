#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import torch
import torch.nn as nn

# window size 以windows size为2进行说明
WS = [2,2]
coords_h = torch.arange(WS[0]) # 0,1
print(coords_h)
coords_w = torch.arange(WS[1])
print(coords_w)

relative_matrix = torch.meshgrid([coords_h, coords_w], indexing="ij")
print(relative_matrix)
rel_matrix = torch.stack(relative_matrix)
print(rel_matrix)
rel_matrix_flat = torch.flatten(rel_matrix,1) # [2,4]
print(rel_matrix_flat.shape)
"""
tensor([[0, 0, 1, 1],
        [0, 1, 0, 1]])
"""
print(rel_matrix_flat)

# 相对位置信息
relative_coords = rel_matrix_flat[:, :, None] - rel_matrix_flat[:, None, :]  # [2, Mh*Mw, Mh*Mw]
print(relative_coords.shape) # 2,4,4
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]

relative_coords[:, :, 0] += WS[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += WS[1] - 1
relative_coords[:, :, 0] *= 2 * WS[1] - 1
print(relative_coords)

# 最后做Sum
"""
tensor([[4, 3, 1, 0],
        [5, 4, 2, 1],
        [7, 6, 4, 3],
        [8, 7, 5, 4]])
"""
relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
print(relative_position_index)

# reshape然后加上attn
# 64: Bs
# 4: num_windows
# 3: num_heads / nH
# 2: winH / winW
nH = 3
attn = torch.rand((64*4,3,2*2,2*2))

# [3*3,3]
relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * WS[0] - 1) * (2 * WS[1] - 1), nH))  # [2*Mh-1 * 2*Mw-1, nH]
print(relative_position_bias_table.shape)

# 展开成1维向量
relative_position_index=relative_position_index.view(-1)
print(relative_position_index.shape)

# [16,3]
# 4,4,3
relative_pos_bias = relative_position_bias_table[relative_position_index].view(WS[0]*WS[1],WS[0]*WS[1],-1)
relative_position_bias = relative_pos_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]

attn = attn + relative_position_bias.unsqueeze(0)
print(attn.shape)