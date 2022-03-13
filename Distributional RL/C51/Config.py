#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

# Parameters
from wrapper import SubprocVecEnv, wrap_cover
import numpy as np
import argparse
import torch
import gym

# parser = argparse.ArgumentParser(description='Some settings of the experiment.')
# parser.add_argument('games', type=str, nargs=1, help='name of the games. for example: Breakout')
# args = parser.parse_args()
# args.games = "".join(args.games)

class Config_C51:
    # sequential images to define state
    STATE_LEN = 4
    # target policy sync interval
    TARGET_REPLACE_ITER = 1
    # simulator steps for start learning
    LEARN_START = int(1e+3)
    # (prioritized) experience replay memory size
    MEMORY_CAPACITY = int(1e+5)
    # simulator steps for learning interval
    LEARN_FREQ = 1
    # atom number. default is C51 algorithm
    N_ATOM = 51

    '''Environment Settings'''
    # number of environments for C51
    N_ENVS = 4
    GAME = "Breakout"
    N_ACTIONS = 0
    N_STATES = 0

    # prior knowledge of return distribution,
    V_MIN = -5.
    V_MAX = 10.
    V_RANGE = np.linspace(V_MIN, V_MAX, N_ATOM)
    V_STEP = ((V_MAX - V_MIN) / (N_ATOM - 1))

    # Total simulation step
    STEP_NUM = int(1e+8)
    # gamma for MDP
    GAMMA = 0.99
    # visualize for agent playing
    RENDERING = False

    '''Training settings'''
    # check GPU usage
    USE_GPU = torch.cuda.is_available()
    print('USE GPU: ' + str(USE_GPU))
    device = "cuda"

    # mini-batch size
    BATCH_SIZE = 32
    # learning rage
    LR = 1e-4
    # epsilon-greedy
    EPSILON = 1.0

    '''Save&Load Settings'''
    # check save/load
    SAVE = True
    LOAD = False
    # save frequency
    SAVE_FREQ = int(1e+3)
    # paths for predction net, target net, result log
    PRED_PATH = './data/model/C51_pred_net_' + GAME + '.pkl'
    TARGET_PATH = './data/model/C51_target_net_' + GAME + '.pkl'
    RESULT_PATH = './data/plots/C51_result_' + GAME + '.pkl'

if __name__ == "__main__":
    Config_C51()

