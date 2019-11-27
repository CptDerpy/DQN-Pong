# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:45:43 2019

@author: August
"""

from collections import deque
import numpy as np


"""
EXPERIENCE BUFFER
"""
class ExperienceBuffer():
    
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, obs, reward, action, obs2, done):
        self.buffer.append((obs, reward, action, obs2, done))
        
    def sample_minibatch(self, batch_size):
        mb_idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        mb = zip(*[self.buffer[i] for i in mb_idxs])
        obs, reward, action, obs2, done = mb
        return (np.array(obs), np.array(reward, dtype=np.float32), np.array(action), np.array(obs2), np.array(done, dtype=np.uint8))
        
    def __len__(self):
        return len(self.buffer)
