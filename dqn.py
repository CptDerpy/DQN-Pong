# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:39:51 2019

@author: August
"""

#from gym.wrappers import Monitor
from atari_wrappers import make_env
from neural_network import DNN
from replay_memory import ExperienceBuffer
from datetime import datetime
import pickle
import numpy as np
import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


"""
TEST AGENT
"""
def test_agent(env, Q_target, num_games=20):
    games_reward = []
    
    for _ in range(num_games):
        done = False
        game_reward = 0
        obs = env.reset()
        
        while not done:
#            env.render()
            action = Q_target.get_action(obs, 0.05)
            obs, reward, done, _ = env.step(action)
            game_reward += reward
        
        games_reward.append(game_reward)
    
    return games_reward


"""
DQN
"""
def DQN(env_name, lr=1e-2, num_episodes=2000, buffer_size=1e5, discount=0.99, render_cycle=100, update_target_net=1000, batch_size=64, update_freq=4, frames_num=2, min_buffer_size=5000, test_freq=20, start_explore=1, end_explore=0.1, explore_steps=1e5):
    
    env = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    test_env = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    
    init_time = datetime.isoformat(datetime.now()).replace(':', '-')[:-7]
 #   test_env = Monitor(test_env, f'videos/{env_name}_{init_time}', force=True, video_callable=lambda x: x % 20 == 0)
    
    obs_dim = env.observation_space.shape
    n_outputs = env.action_space.n
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'runs/{env_name}_{init_time}')
    with open(f'runs/{env_name}_{init_time}/hyperparams.txt', 'w') as file:
        file.write(f"Learning rate:\t\t {lr}\n")
        file.write(f"Episodes:\t\t {num_episodes}\n")
        file.write(f"Buffer size:\t\t {buffer_size}\n")
        file.write(f"Discount:\t\t {discount}\n")
        file.write(f"Update target:\t\t {update_target_net}\n")
        file.write(f"Batch size:\t\t {batch_size}\n")
        file.write(f"Update frequenzy:\t {update_freq}\n")
        file.write(f"Frames num:\t\t {frames_num}\n")
        file.write(f"Min. buffer size:\t {min_buffer_size}\n")
        file.write(f"Test Frequenzy:\t\t {test_freq}\n")
        file.write(f"Start exploration:\t {start_explore}\n")
        file.write(f"End exploration:\t {end_explore}\n")
        file.write(f"Exploration steps:\t {explore_steps}")
    # Open summary by running tensorboard --logdir=runs from the command line and opening https://localhost:6006
    
    
    # Initialize Q function with random weight
    Q = DNN(obs_dim, n_outputs, lr, device).to(device)
    
    # Initialize Q_target function with random weight equal to Q's weight
    Q_target = DNN(obs_dim, n_outputs, lr, device).to(device)
    Q_target.load_state_dict(Q.state_dict())
    
    # Initialize empty replay memory
    exp_buffer = ExperienceBuffer(buffer_size)
    
    # Initialize environment
    obs = env.reset()
    step_count = 0
    batch_reward = []
    running_loss = 0
    
    eps = start_explore
    eps_decay = (start_explore - end_explore) / explore_steps
    
    
    for episode in range(num_episodes):
        print('\nEpisode:', episode)
        
        game_reward = 0
        done = False
        
        while not done:
#            print(f'\rStep {step_count}', end='', flush=True)
            # Collect observation from environment
            action = Q.get_action(obs, eps)
            new_obs, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
#            if len(exp_buffer) > buffer_size:
#                exp_buffer.pop()
            exp_buffer.add(
                obs,
                reward,
                action,
                new_obs,
                done
            )
            
            obs = new_obs
            game_reward += reward
            step_count += 1
            
            # Epsilon decay
            if eps > end_explore:
                eps -= eps_decay
            
            # Train online network
            if (len(exp_buffer) > min_buffer_size) and (step_count % update_freq == 0):
                mb_obs, mb_reward, mb_action, mb_obs2, mb_done = exp_buffer.sample_minibatch(batch_size)
                mb_obs = Variable(FloatTensor(mb_obs)).to(device)
                mb_reward = Variable(FloatTensor(mb_reward)).to(device)
                mb_action = Variable(LongTensor(mb_action)).to(device)
                mb_obs2 = Variable(FloatTensor(mb_obs2)).to(device)
                mb_done = Variable(FloatTensor(mb_done)).to(device)
                
                # Calculate action values
                qv = Q.forward(mb_obs).gather(1, mb_action.unsqueeze(1)).squeeze()
                
                # Calculate target values
                mb_target_qv = Q_target.forward(mb_obs2).max(1)[0]
#                y_r = q_target_values(mb_reward, mb_done, mb_target_qv, discount)
                
                # Calculate expected values
                e_qv = mb_reward + discount * mb_target_qv * (1 - mb_done)
                
                
                # Update gradient
                Q.optimizer.zero_grad()
                loss = Q.loss(qv, e_qv)
                loss.backward()
#                for param in Q.parameters():
#                    param.grad.data.clamp_(-1, 1)
                Q.optimizer.step()
                
                running_loss += loss.item()
                
                if step_count % 1000 == 0:
                    writer.add_scalar('training loss', running_loss/1000, step_count)
                    running_loss = 0
            
            
            # Update target network:
            # Every C steps set Q_target's weight equal to Q's weight
            if (len(exp_buffer) > min_buffer_size) and (step_count % update_target_net == 0):
 #               print('\nUpdating target network')
                Q_target.load_state_dict(Q.state_dict())
            
            if done:
                obs = env.reset()
                writer.add_scalar('training reward', game_reward, step_count)
                batch_reward.append(game_reward)
                print()
                
        if (episode+1) % test_freq == 0:
            print('Testing target network')
            test_reward = test_agent(test_env, Q_target, num_games=20)
            writer.add_scalar('test reward', np.mean(test_reward), step_count)
            print('Ep: {:3d} Rew: {:3.2f}, Eps: {:1.2f} -- Step: {:4d} -- Test: {:3.2f} {:3.2f}'.format(episode, np.mean(batch_reward), eps, step_count, np.mean(test_reward), np.std(test_reward)))
    
            torch.save(Q.state_dict(), f'runs/{env_name}_{init_time}/Q_params.pt')
            torch.save(Q_target.state_dict(), f'runs/{env_name}_{init_time}/Q_target_params.pt')
            
    with open(f'runs/{env_name}_{init_time}/replay_buffer.dat', 'wb') as out_file:
        pickle.dump(exp_buffer, out_file)
    env.close()
    test_env.close()
    
