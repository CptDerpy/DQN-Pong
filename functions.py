# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:48:25 2019

@author: August
"""

import numpy as np

"""
NORMALIZE FRAMES
"""
def scale_frames(frames):
    '''
    Scale the frame with number between 0 and 1
    '''
    return np.array(frames, dtype=np.float32) / 255.0
    

"""
GREEDY POLICY
"""
def greedy(action_values):
    return np.argmax(action_values.detach().cpu().numpy())


"""
EPSILON-GREEDY POLICY
"""
def eps_greedy(action_values, eps=0.1):
    if np.random.uniform(0, 1) < eps:
        return np.random.randint(len(action_values))
    else:
        return greedy(action_values)
    

"""
CALCULATE TARGET VALUE FOR EACH TRANSITION
"""
def q_target_values(mb_reward, mb_done, action_values, discounted):
    max_av = np.max(action_values.detach().cpu().numpy(), axis=1)
    
    ys = []
    for r, d, av in zip(mb_reward, mb_done, max_av):
        if d:
            ys.append(r)
        else:
            q_step = r + discounted * av
            ys.append(q_step)
    
    assert len(ys) == len(mb_reward)
    return ys



"""
TEST AGENT
"""
def test_agent(env, agent_op, num_games=20):
    games_reward = []
    
    for _ in range(num_games):
        done = False
        game_reward = 0
        obs = env.reset()
        
        while not done:
            action = eps_greedy(agent_op(obs).cuda(), eps=0.05)
            obs, reward, done, _ = env.step(action)
            game_reward += reward
        
        games_reward.append(game_reward)
    
    return games_reward



















