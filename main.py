# -*- coding: utf-8 -*-

from dqn import DQN

if __name__ == '__main__':
    ENV = 'PongNoFrameskip-v4'
    LEARNING_RATE       =       2.5e-4
    EPISODES            =    1500
    REPLAY_MEMORY_SIZE  =   int(1e6)
    DISCOUNT            =       0.99
    UPDATE_TARGET_NET   =   10000
    BATCH_SIZE          =      32
    UPDATE_FREQUENCY    =       1
    FRAMES_NUMBER       =       4
    MINIMUM_BUFFER_SIZE =   10000
    TEST_FREQUENCY      =      10
    EPSILON_START       =       1
    EPSILON_END         =       0.1
    EXPLORE_STEPS       =   int(1e6)
    
    DQN(ENV, LEARNING_RATE, EPISODES, REPLAY_MEMORY_SIZE, DISCOUNT, UPDATE_TARGET_NET, BATCH_SIZE, UPDATE_FREQUENCY, FRAMES_NUMBER, MINIMUM_BUFFER_SIZE, TEST_FREQUENCY, EPSILON_START, EPSILON_END, EXPLORE_STEPS)
