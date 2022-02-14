#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

from BasicModule import window_partition
import numpy as np
import torch

"""测试脚本"""
# Mask Create
H = 6  # 14
W = 6  # 14
patch_size = 1
window_size = 3  # 7
shift_size = 2  # 3
# 确保H,W是Window_size的整数倍
Hp = int(np.ceil(H / window_size)) * window_size
Wp = int(np.ceil(W / window_size)) * window_size
print("Hp:", Hp)
print("Wp:", Wp)

img_mask = torch.zeros((1, H, W, 1))  # [1, Hp, Wp, 1]
print(img_mask.shape)

# Slice 以4x4, w_s=2, s_s=1为例
# (0,-2) 即 [0、1], (-2,-1) 即 [2], (-1)即[3]
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
print(h_slices)
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
print(w_slices)

# 区域Flag
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

print(img_mask[0, :, :, 0])

# [nW, Mh, Mw, 1] Mh代表windows height Mw同理
mask_windows = window_partition(img_mask, window_size)
print(mask_windows[0, :, :, 0])
print(mask_windows[1, :, :, 0])
print(mask_windows[2, :, :, 0])
print(mask_windows[3, :, :, 0])

# [nW,Mh*Mw]
mask_windows = mask_windows.view(-1, window_size * window_size)
print(mask_windows)

# [nW, Mh*Mw, Mh*Mw]
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  #
# [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
# [nW, 4, 4]
print(attn_mask.shape)
# print(attn_mask) # 会进行broadcasting


attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
print(attn_mask[0, :, :])  # 全0
print(attn_mask[1, :, :])
print(attn_mask[2, :, :])
print(attn_mask[3, :, :])
# print(attn_mask[0])
# print(attn_mask)
