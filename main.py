# -*- coding: utf-8 -*-

import gym
import torch
import torch.optim as optim

from neural_network import DQN, DuelingDQN
from replay_memory import ExperienceReplay, PER
from gym.wrappers import AtariPreprocessing, FrameStack
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def optimize(eps):
    # Sample minibatch
    if PRIORITIZED_REPLAY:
        mb_obs, mb_reward, mb_action, mb_obs2, mb_done, importance, idxs = memory.sample_minibatch(BATCH_SIZE, 0.7)
    else:
        mb_obs, mb_reward, mb_action, mb_obs2, mb_done = memory.sample_minibatch(BATCH_SIZE)
    mb_obs = Variable(FloatTensor(mb_obs)).to(device)
    mb_reward = Variable(FloatTensor(mb_reward)).to(device)
    mb_action = Variable(LongTensor(mb_action)).to(device)
    mb_obs2 = Variable(FloatTensor(mb_obs2)).to(device)
    mb_done = Variable(FloatTensor(mb_done)).to(device)

    # Calculate action values
    qv = policy_net.forward(mb_obs).gather(1, mb_action.unsqueeze(1)).squeeze()

    # Calculate target values
    if DDQN:
        mb_target_qv = target_net.forward(mb_obs2).gather(1, policy_net.forward(mb_obs2).max(1)[1].unsqueeze(1)).squeeze()
    else:
        mb_target_qv = target_net.forward(mb_obs2).max(1)[0]
    
    # Calculate expected values
    e_qv = mb_reward + DISCOUNT * mb_target_qv * (1 - mb_done)

    if PRIORITIZED_REPLAY:
        # Calculate TD-errors
        errors = (e_qv - qv).cpu().detach().numpy()

        # Update transition priorities
        memory.set_priorities(idxs, errors)

    # Calculate loss
    loss = (FloatTensor(importance**(1 - eps)).to(device) * policy_net.loss(qv, e_qv)).mean() if PRIORITIZED_REPLAY else policy_net.loss(qv, e_qv)

    # Update gradient
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()


def train():
    # Initialize environment
    step_count = 0
    running_loss = 0
    batch_reward = []
    eps = EPSILON_START
    decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY

    # Run episodes
    for episode in range(EPISODES):
        print('Episode', episode)
        
        obs = env.reset()
        game_reward = 0
        done = False

        while not done:
            # Retrieve action from epsilon-greedy policy
            action = policy_net.get_action(obs, eps)
            new_obs, reward, done, _ = env.step(action)

            # Store transition in replay memory
            memory.add(obs, reward, action, new_obs, done)

            obs = new_obs
            game_reward += reward
            step_count += 1

            # Epsilon decay
            if eps > EPSILON_END:
                eps -= decay

            # Optimize policy network
            if len(memory) > MIN_MEMORY_SIZE:
                running_loss += optimize(eps)
                
                # Update TensorBoard loss graph
                if step_count % 1000 == 0:
                    writer.add_scalar('training loss', running_loss/1000, step_count)
                    running_loss = 0

                # Update target network
                if step_count % UPDATE_TARGET == 0:
                    target_net.load_state_dict(policy_net.state_dict())
        
        # Update TensorBoard train reward graph
        writer.add_scalar('training reward', game_reward, step_count)

        # Test target network
        if episode % TEST_FREQUENCY == 0:
            mean_test_reward = test()
            
            # Update Tensorboard test reward graph
            writer.add_scalar('test reward', mean_test_reward, step_count)

            # Save target network parameters to file
            torch.save(target_net.state_dict(), f'{log_path}/target_net_params.pt')


def test():
    print('Testing target network')

    games_reward = 0

    for episode in range(10):
        print('Test episode', episode)

        obs = env.reset()
        done = False

        while not done:
            action = target_net.get_action(obs)
            obs, reward, done, _ = env.step(action)
            games_reward += reward
    
    return games_reward / 10


if __name__ == '__main__':

    # Hyperparameters
    ENV_NAME            =   'PongNoFrameskip-v4'
    RUN_TITLE           =   'PER_Dueling_DDQN'
    LEARNING_RATE       =   1e-4
    EPISODES            =   1000
    DISCOUNT            =   0.99
    BATCH_SIZE          =   32
    MEMORY_SIZE         =   int(1e5)
    MIN_MEMORY_SIZE     =   int(1e4)
    UPDATE_TARGET       =   int(1e3)
    FRAMES_NUMBER       =   4
    TEST_FREQUENCY      =   10
    EPSILON_START       =   1
    EPSILON_END         =   0.1
    EPSILON_DECAY       =   int(5e5)
    DDQN                =   True
    DUELING_DQN         =   True
    PRIORITIZED_REPLAY  =   True

    # GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to log files
    init_time = datetime.isoformat(datetime.now()).replace(':', '-')[:-7]
    log_path = f'runs/{ENV_NAME}_{RUN_TITLE}_{init_time}'

    # TensorBoard summary writer
    writer = SummaryWriter(log_path)

    # Hyperparameter log file
    with open(f'{log_path}/hyperparams.txt', 'w') as file:
        file.write(f"{RUN_TITLE} Hyperparameters\n\n")
        file.write(f"Learning rate:\t\t {LEARNING_RATE}\n")
        file.write(f"Episodes:\t\t\t {EPISODES}\n")
        file.write(f"Discount:\t\t\t {DISCOUNT}\n")
        file.write(f"Batch size:\t\t\t {BATCH_SIZE}\n")
        file.write(f"Memory size:\t\t {MEMORY_SIZE}\n")
        file.write(f"Min. memory size:\t {MIN_MEMORY_SIZE}\n")
        file.write(f"Update target:\t\t {UPDATE_TARGET}\n")
        file.write(f"Frames num:\t\t\t {FRAMES_NUMBER}\n")
        file.write(f"Test Frequenzy:\t\t {TEST_FREQUENCY}\n")
        file.write(f"Start exploration:\t {EPSILON_START}\n")
        file.write(f"End exploration:\t {EPSILON_END}\n")
        file.write(f"Exploration steps:\t {EPSILON_DECAY}\n")
        file.write(f"Double DQN:\t\t\t {DDQN}\n")
        file.write(f"Dueling DQN:\t\t {DUELING_DQN}\n")
        file.write(f"Prioritized Replay:\t {PRIORITIZED_REPLAY}")

    # Environment
    env = FrameStack(AtariPreprocessing(gym.make(ENV_NAME)), FRAMES_NUMBER)

    # Dimensions of observations
    obs_dim = env.observation_space.shape

    # Amount of actions
    n_outputs = env.action_space.n
    
    # Neural networks
    policy_net = DuelingDQN(obs_dim, n_outputs, device).to(device) if DUELING_DQN \
            else DQN(obs_dim, n_outputs, device).to(device)
    target_net = DuelingDQN(obs_dim, n_outputs, device).to(device) if DUELING_DQN \
            else DQN(obs_dim, n_outputs, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    # Replay memory
    memory = PER(MEMORY_SIZE) if PRIORITIZED_REPLAY \
        else ExperienceReplay(MEMORY_SIZE)

    # Train policy network
    train()

    # Close environment
    env.close()
