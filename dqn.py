# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:39:51 2019

@author: August
"""

from gym.wrappers import Monitor
from atari_wrappers import make_env
from neural_network import DNN
from replay_memory import ExperienceBuffer
from functions import scale_frames, eps_greedy, q_target_values, test_agent
from time import time
import numpy as np
import torch


"""
DQN
"""
def DQN(env_name, lr=1e-2, num_episodes=2000, buffer_size=1e5, discount=0.99, render_cycle=100, update_target_net=1000, batch_size=64, update_freq=4, frames_num=2, min_buffer_size=5000, test_freq=20, start_explore=1, end_explore=0.1, explore_steps=1e5):
    
    env = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    test_env = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    test_env = Monitor(test_env, f'videos/{env_name}_{int(time()*1000)}', force=True, video_callable=lambda x: x % 20 == 0)
    
    obs_dim = env.observation_space.shape
    n_outputs = env.action_space.n
    
    
    # Initialize Q function with random weight
    Q = DNN(obs_dim, n_outputs, lr).cuda()
    
    # Initialize Q_target function with random weight equal to Q's weight
    Q_target = DNN(obs_dim, n_outputs, lr).cuda()
    Q_target.load_state_dict(Q.state_dict())
    
    # Initialize empty replay memory
    exp_buffer = ExperienceBuffer(buffer_size)
    
    # Initialize environment
    obs = env.reset()
    step_count = 0
    last_update_loss = []
    batch_reward = []
    
    eps = start_explore
    eps_decay = (start_explore - end_explore) / explore_steps
    
    def agent_op(obs):
        obs = scale_frames(obs)
        obs = torch.from_numpy(obs).cuda()
        return Q.forward(obs)
    
    
    for episode in range(num_episodes):
        print('Episode:', episode)
        
        game_reward = 0
        done = False
        
        while not done:
            # Collect observation from environment
            action = eps_greedy(agent_op(obs).squeeze(), eps)
            new_obs, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
#            if len(exp_buffer) > buffer_size:
#                exp_buffer.pop()
            exp_buffer.add(obs, reward, action, new_obs, done)
            
            obs = new_obs
            game_reward += reward
            step_count += 1
            
            # Epsilon decay
            if eps > end_explore:
                eps -= eps_decay
            
            # Train online network
            if (len(exp_buffer) > min_buffer_size) and (step_count % update_freq == 0):
                mb_obs, mb_reward, mb_action, mb_obs2, mb_done = exp_buffer.sample_minibatch(batch_size)
                
                # Calculate action values
                av = Q.forward(torch.from_numpy(mb_obs).cuda()).gather(1, torch.LongTensor(mb_action).unsqueeze(1).cuda())
                
                # Calculate target values
                mb_target_qv = Q_target.forward(torch.from_numpy(mb_obs2).cuda())
                y_r = q_target_values(mb_reward, mb_done, mb_target_qv, discount)
                
                # Update gradient
                Q.optimizer.zero_grad()
                loss = Q.loss(av, y_r)
                loss.backward()
                for param in Q.parameters():
                    param.grad.data.clamp_(-1, 1)
                Q.optimizer.step()
                last_update_loss.append(loss)
            
            
            # Update target network:
            # Every C steps set Q_target's weight equal to Q's weight
            if (len(exp_buffer) > min_buffer_size) and (step_count % update_target_net == 0):
                Q_target.load_state_dict(Q.state_dict())
                last_update_loss = []
            
            if done:
                obs = env.reset()
                batch_reward.append(game_reward)
                
        if (episode+1) % test_freq == 0:
            test_reward = test_agent(test_env, agent_op, num_games=10)
            print('Ep: {:3d} Rew: {:3.2f}, Eps: {:1.2f} -- Step: {:4d} -- Test: {:3.2f} {:3.2f}'.format(episode, np.mean(batch_reward), eps, step_count, np.mean(test_reward), np.std(test_reward)))
    
    env.close()
    test_env.close()
    