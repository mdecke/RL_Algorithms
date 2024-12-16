import random
import torch

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.DDPG import *


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    np.random.seed(42)

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
    policy_tilde = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
    policy_tilde.load_state_dict(policy.state_dict())
    
    value = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
    value_tilde = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
    value_tilde.load_state_dict(value.state_dict())

    training_steps = 10000
    warm_up = 1000
    discount_gamma = 0.99
    buffer_length = 6000
    batch_size = 128
    
    agent = DDPG(policy_network=policy, target_policy=policy_tilde, env=env,
                 value_network=value, target_value_function=value_tilde,
                 discount_factor=discount_gamma, total_training_time=training_steps)
    
    memory = DDPGMemory(buffer_length=buffer_length)


    noise = OUNoise(action_dim=action_dim)

    obs, _ = env.reset()
    noise.reset()
    episode_rewards = []
    total_reward = 0
    exploration_noise = []

    for t in range(training_steps):
        obs = torch.tensor(obs).to(device=device)

        if t <= warm_up:
            noisy_action = env.action_space.sample()
        else:
            action = policy.forward(obs)
            expl_noise = noise.sample()
            exploration_noise.append(expl_noise)
            noisy_action = action.detach().cpu().numpy() + expl_noise
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
        
        obs_, reward, termination, truncation, _ = env.step(noisy_action)
        done = termination + truncation
        total_reward += reward
        
        state_tensor = torch.tensor(obs,dtype=torch.float32,device=device)
        action_tensor = torch.tensor(noisy_action,dtype=torch.float32, device=device)
        reward_tensor = torch.tensor(reward,dtype=torch.float32, device=device)
        next_state_tensor = torch.tensor(obs_, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.int, device=device)
        
        memory.add_sample(state=state_tensor, action=action_tensor, reward=reward_tensor,
                          next_state=next_state_tensor, done=done)
        
        if t>=warm_up and len(memory.states) >= batch_size:
            # print('training')
            agent.train(memory_buffer=memory, train_iteration=t, batch_size=batch_size,epochs=1)
        # else:
            # print('not training')
        
        if done.item():
            episode_rewards.append(total_reward)
            print(f'Return for episode {len(episode_rewards)} is : {total_reward}')
            total_reward = 0
            obs, _ = env.reset()
            noise.reset()
        else:
            obs = obs_
    
    pi_loss_series = pd.Series(agent.pi_loss)
    q_loss_series = pd.Series(agent.q_loss)
    # time = np.linspace(0,training_steps-warm_up, 1)

    window_size = 50  # window size for smoothing

    # Compute rolling mean and std
    pi_loss_mean = pi_loss_series.rolling(window=window_size).mean()
    pi_loss_std = pi_loss_series.rolling(window=window_size).std()
    q_loss_mean = q_loss_series.rolling(window=window_size).mean()
    q_loss_std = q_loss_series.rolling(window=window_size).std()

    plt.figure(figsize=(10, 8))

    # Q-function loss subplot
    plt.subplot(3, 1, 1)
    plt.plot(q_loss_mean, label='Q-value Loss (smoothed)', color='blue')
    plt.fill_between(x=range(len(q_loss_mean)),
                    y1=q_loss_mean - q_loss_std,
                    y2=q_loss_mean + q_loss_std,
                    color='blue', alpha=0.2)
    plt.ylabel('Value Loss')
    plt.xlabel('Training Steps')
    plt.title('Q-function Loss (Smoothed)')
    plt.grid(True)
    plt.legend()

    # Policy loss subplot
    plt.subplot(3, 1, 2)
    plt.plot(pi_loss_mean, label='Policy Loss (smoothed)', color='orange')
    plt.fill_between(x=range(len(pi_loss_mean)),
                    y1=pi_loss_mean - pi_loss_std,
                    y2=pi_loss_mean + pi_loss_std,
                    color='orange', alpha=0.2)
    plt.ylabel('Policy Loss')
    plt.xlabel('Training Steps')
    plt.title('Policy Loss (Smoothed)')
    plt.grid(True)
    plt.legend()

    # Episodic return subplot (no smoothing here, just raw)
    plt.subplot(3, 1, 3)
    plt.plot(episode_rewards, label='Episodic Return', color='green')
    plt.ylabel('Return')
    plt.xlabel('Episodes')
    plt.title('Episodic Return')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


