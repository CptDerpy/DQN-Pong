# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:45:43 2019

@author: August
"""

from collections import deque
import numpy as np
import torch


"""
EXPERIENCE BUFFER
"""
class ExperienceBuffer():
    
    def __init__(self, buffer_size):
        self.obs_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=buffer_size)
        self.action_buffer = deque(maxlen=buffer_size)
        self.obs2_buffer = deque(maxlen=buffer_size)
        self.done_buffer = deque(maxlen=buffer_size)
    
    def add(self, obs, reward, action, obs2, done):
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.action_buffer.append(action)
        self.obs2_buffer.append(obs2)
        self.done_buffer.append(done)
        
    def sample_minibatch(self, batch_size):
        mb_idxs = np.random.randint(len(self.obs_buffer), size=batch_size)
        
        mb_obs = torch.stack([self.obs_buffer[i] for i in mb_idxs])
        mb_reward = torch.stack([self.reward_buffer[i] for i in mb_idxs])
        mb_action = torch.stack([self.action_buffer[i] for i in mb_idxs])
        mb_obs2 = torch.stack([self.obs2_buffer[i] for i in mb_idxs])
        mb_done = torch.stack([self.done_buffer[i] for i in mb_idxs])
        
        return mb_obs, mb_reward, mb_action, mb_obs2, mb_done
        
    def __len__(self):
        return len(self.obs_buffer)
