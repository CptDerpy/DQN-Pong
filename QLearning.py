# -*- coding: utf-8 -*-

#%matplotlib inline
#import matplotlib.pyplot as plt
import numpy as np
import pickle
import gym
#from gym import wrappers
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import torchvision.transforms as T
# from IPython.display import Video


def get_discrete_state(state):
    discrete_state = (state - os_low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

def greedy(Q, s):
    return np.argmax(Q[s])

def eps_greedy(Q, s, eps=0.1):
    if np.random.uniform(0,1) < eps:
        # Choose a random action
        return env.action_space.sample()
    else:
        # Choose the greedy action
        return greedy(Q, s)

def run_episodes(env, Q, num_episodes=100, to_print=False):
    tot_rew = []
    state = env.reset()
    state = get_discrete_state(state)
    for i in range(num_episodes):
        done = False
        game_rew = 0
        while not done:
            state, rew, done, _ = env.step(greedy(Q, state))
            # if i == num_episodes-1:
            #     env.render()
            state = get_discrete_state(state)
            game_rew += rew
            if done:
                state = env.reset()
                state = get_discrete_state(state)
                tot_rew.append(game_rew)
    if to_print:
        print(f'Mean score: {np.mean(tot_rew):.3f} of {num_episodes} games!')
    else:
        return np.mean(tot_rew)

def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    """
    nA = env.action_space.n
    nS = env.observation_space.n

    # Rows = states, Columns = actions
    Q = np.zeros((nS, nA))
    """

    DISCRETE_OS_SIZE = [20] * env.observation_space.shape[0]

    os_high = env.observation_space.high
    os_high[1] = 5.0
    os_high[3] = 15.0

    global os_low
    os_low = env.observation_space.low
    os_low[1] = -5.0
    os_low[3] = -15.0

    global discrete_os_win_size

    discrete_os_win_size = (os_high - os_low) / DISCRETE_OS_SIZE

    Q = np.zeros((DISCRETE_OS_SIZE + [env.action_space.n]))

    games_reward = []
    test_rewards = []
    episodes = []

    # Run episode cycle
    for ep in range(num_episodes+1):
        # Initialize episode variables
        state = env.reset()
        state = get_discrete_state(state)
        done = False
        tot_rew = 0
        if eps > 0.1:
            eps -= eps_decay
        
        # Q-learning update
        while not done:
            action = eps_greedy(Q, state, eps)
            next_state, rew, done, _ = env.step(action)
            next_state = get_discrete_state(next_state)
            Q[state][action] += lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
            tot_rew += rew
        games_reward.append(tot_rew)

        # Every 300th iteration, run 1000 games to test the agent
        if ep % 100 == 0:
            test_rew = run_episodes(env, Q, 1000)
            print(f'Episode: {ep:5d}\t Eps: {eps:1.4f}\t Rew: {test_rew:3.4f}')
            test_rewards.append(test_rew)
            episodes.append(ep)
    pickle.dump((episodes, test_rewards), open('q_learning_data2.p', 'wb'))
    return Q

if __name__ == '__main__':
    # Initialize environment
    env = gym.make('CartPole-v0')
#    gym.envs.register(
#            id='CartPole-500eps-v0',
#            entry_point='gym.envs.classic_control:CartPoleEnv',
#            max_episode_steps=500
#    )
    # env = gym.make('CartPole-500eps-v0')

    # Hyperparameter initialization
    LEARNING_RATE = 0.01
    EPISODES = 25000
    EPSILON = 1.0
    DISCOUNT = 0.99
    DECAY = 0.00005

    Q = Q_learning(env, lr=LEARNING_RATE, num_episodes=EPISODES, eps=EPSILON, gamma=DISCOUNT, eps_decay=DECAY)

    # Close environment
    env.close()