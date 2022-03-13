#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self,state_len, n_actions, n_atoms):
        super(ConvNet, self).__init__()
        self.n_actions = n_actions
        self.state_len = state_len # (4,84,84), channels=4, h=84, w=84
        self.n_atoms = n_atoms

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(state_len[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(7 * 7 * 64, 512)

        # action value distribution
        self.fc_q = nn.Linear(512, n_actions * n_atoms)  # 输出是n_action x n_atoms的近似分布矩阵

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        # x: (m, 84, 84, 4) tensor
        x = self.feature_extraction(x / 255.0)
        # x.size(0) : mini-batch size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        # note that output of C-51 is prob mass of value distribution
        action_value = F.softmax(self.fc_q(x).view(mb_size, self.n_actions, self.n_atoms), dim=2)

        return action_value

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

