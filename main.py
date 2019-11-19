# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 00:06:32 2019

@author: August
"""

from dqn import DQN

if __name__ == '__main__':
    ENV = 'PongNoFrameskip-v4'
    LEARNING_RATE       =       1e-2
    EPISODES            =    2000
    REPLAY_MEMORY_SIZE  =   int(1e6)
    DISCOUNT            =       0.99
    RENDER_CYCLE        =     100
    UPDATE_TARGET_NET   =    1000
    BATCH_SIZE          =      32
    UPDATE_FREQUENCY    =       2
    FRAMES_NUMBER       =       2
    MINIMUM_BUFFER_SIZE =   10000
    TEST_FREQUENCY      =      20
    EPSILON_START       =       1
    EPSILON_END         =       0.1
    EXPLORE_STEPS       =       1e5
    
    DQN(ENV, LEARNING_RATE, EPISODES, REPLAY_MEMORY_SIZE, DISCOUNT, RENDER_CYCLE, UPDATE_TARGET_NET, BATCH_SIZE, UPDATE_FREQUENCY, FRAMES_NUMBER, MINIMUM_BUFFER_SIZE, TEST_FREQUENCY, EPSILON_START, EPSILON_END, EXPLORE_STEPS)