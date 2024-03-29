# -*- coding: utf-8 -*-

#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random, randrange
#import torchvision.transforms as T


"""
DEEP Q-NETWORK
"""
class DQN(nn.Module):
    
    def __init__(self, obs_shape, n_outputs, device):
        super(DQN, self).__init__()

        self.ch, self.w, self.h = obs_shape
        self.n_outputs = n_outputs
        self.device = device
        
        # Convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ch, 
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_outputs)
        )
        
    @property
    def conv_out(self):
        x = self.conv(torch.zeros(1, self.ch, self.w, self.h))
        return x.view(1, -1).size(1)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def get_action(self, obs, eps=0):
        # Epsilon-greedy policy
        if random() < eps:
            action = randrange(self.n_outputs)
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_val = self.forward(obs)
                action = q_val.max(1)[1].item()
        return action
    
    def loss(self, action_values, target_values):
        # Mean Squared Error
        # return F.mse_loss(action_values, target_values)
        # Huber loss
        return F.smooth_l1_loss(action_values, target_values)


"""
DUELING NETWORK
"""
class DuelingDQN(nn.Module):
    
    def __init__(self, obs_shape, n_outputs, device):
        super(DuelingDQN, self).__init__()

        self.ch, self.w, self.h = obs_shape
        self.n_outputs = n_outputs
        self.device = device
        
        # Convolution layers:
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ch, 
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU()
        )
        
        # State value function
        self.fc_V = nn.Sequential(
            nn.Linear(self.conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Action advantage function
        self.fc_A = nn.Sequential(
            nn.Linear(self.conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_outputs)
        )
        
    @property
    def conv_out(self):
        x = self.conv(torch.zeros(1, self.ch, self.w, self.h))
        return x.view(1, -1).size(1)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        adv = self.fc_A(x)
        val = self.fc_V(x).expand(x.size(0), self.n_outputs)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_outputs)
        return x
    
    def get_action(self, obs, eps=0):
        # Epsilon-greedy policy
        if random() < eps:
            action = randrange(self.n_outputs)
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_val = self.forward(obs)
            action = q_val.max(1)[1].item()
        return action
    
    def loss(self, action_values, target_values):
        # Mean Squared Error
        # return F.mse_loss(action_values, target_values)
        # Huber loss
        return F.smooth_l1_loss(action_values, target_values)
