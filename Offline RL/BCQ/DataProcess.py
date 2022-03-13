#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn


import numpy as np
import matplotlib.pyplot as plt

# plt.style.use(['science'])

DataDir = "results/"

BehaviorData = "behavioral_Hopper-v3_0"
Buffer_PerformanceData = "buffer_performance_Hopper-v3_0"

data1 = np.load(DataDir+BehaviorData+".npy")
data2 = np.load(DataDir+Buffer_PerformanceData+".npy")
print(data1.shape)
print(data2.shape)

plt.plot(data1)
plt.plot(data2)
plt.show()