# -*- coding: utf-8 -*-

from collections import deque
import numpy as np


"""
EXPERIENCE REPLAY
"""
class ExperienceReplay:
    
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, obs, reward, action, obs2, done):
        self.buffer.append((obs, reward, action, obs2, done))
        
    def sample_minibatch(self, batch_size):
        mb_idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        mb = zip(*[self.buffer[i] for i in mb_idxs])
        obs, reward, action, obs2, done = mb
        return (
            np.array(obs), 
            np.array(reward, dtype=np.float32), 
            np.array(action), 
            np.array(obs2), 
            np.array(done, dtype=np.uint8)
        )
        
    def __len__(self):
        return len(self.buffer)


"""
PRIORITIZED EXPERIENCE REPLAY
"""
class PER:
    
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
    
    def add(self, obs, reward, action, obs2, done):
        self.buffer.append((obs, reward, action, obs2, done))
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        batch_probabilities = scaled_priorities / sum(scaled_priorities)
        return batch_probabilities

    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        normalized_importance = importance / max(importance)
        return normalized_importance
        
    def sample_minibatch(self, batch_size, priority_scale=1):
        probabilities = self.get_probabilities(priority_scale)

        mb_idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probabilities)
        
        mb = zip(*[self.buffer[i] for i in mb_idxs])

        importance = self.get_importance(probabilities[mb_idxs])

        obs, reward, action, obs2, done = mb
        return (
            np.array(obs), 
            np.array(reward, dtype=np.float32), 
            np.array(action), 
            np.array(obs2), 
            np.array(done, dtype=np.uint8),
            importance,
            mb_idxs
        )
    
    def set_priorities(self, idxs, errors, offset=0.1):
        for idx, err in zip(idxs, errors):
            self.priorities[idx] = abs(err) + offset
        
    def __len__(self):
        return len(self.buffer)
