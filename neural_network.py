# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:44:28 2019

@author: August
"""

#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision.transforms as T


"""
DNN
"""
class DNN(nn.Module):
    
    def __init__(self, obs_shape, n_outputs, learning_rate):
        super(DNN, self).__init__()
        
        w, h, ch = obs_shape
        self.conv_out = w * h
        self.conv_out = (self.conv_out - (8-1) - 1) // 4 + 1
        self.conv_out = (self.conv_out - (4-1) - 1) // 2 + 1
        self.conv_out = (self.conv_out - (3-1) - 1) // 1 + 1
        self.conv_out = 4608
        
        # Network
        self.conv1 = nn.Conv2d(
                in_channels=ch, 
                out_channels=16,
                kernel_size=8,
                stride=4,
                padding=4
        )
        
        self.conv2 = nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=2
        )
        
        self.conv3 = nn.Conv2d(
                in_channels=32, 
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
        )
        
        self.hidden = nn.Linear(self.conv_out, 128)
        
        self.out = nn.Linear(128, n_outputs)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        
    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
#        return F.softmax(x)
        return x
    
    def loss(self, action_values, target_values):
        # Mean Squared Error
        return F.mse_loss(action_values.squeeze(), torch.FloatTensor(target_values))
        