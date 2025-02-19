import torch

import numpy as np
import pandas as pd
import gymnasium as gym

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.DDPG import *
from MiscFunctions.Plotting import *
from MiscFunctions.DataProcessing import get_state


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

NB_TRAINING_CYCLES = 10
NOISE = 'Gaussian' # 'Gaussian' or 'OrnsteinUhlenbeck'
PLOTTING = True


if __name__ == '__main__':
    np.random.seed(42)
    
    if NOISE == 'Gaussian':
        noise = Normal(loc=0, scale=0.1)
    elif NOISE == 'OrnsteinUhlenbeck':
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=0.1)
    else:
        raise ValueError('Noise must be either Gaussian or OrnsteinUhlenbeck')
    
    env = gym.make('Pendulum-v1')
    state_dim = 3
    action_dim = env.action_space.shape[0]

    policy_losses = pd.DataFrame()
    value_losses = pd.DataFrame()
    training_returns = pd.DataFrame()

    training_steps = 15000
    warm_up = 1
    discount_gamma = 0.99
    buffer_length = 15000
    batch_size = 100

    for cycles in range(10):
        print(f'\n... training cycle {cycles+1}/{NB_TRAINING_CYCLES} ...')

        policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
        policy_tilde = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
        policy_tilde.load_state_dict(policy.state_dict())
        
        value = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
        value_tilde = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
        value_tilde.load_state_dict(value.state_dict())

        
        agent = DDPG(policy_network=policy, target_policy=policy_tilde, env=env,
                    value_network=value, target_value_function=value_tilde,
                    discount_factor=discount_gamma, total_training_time=training_steps)
        
        memory = DDPGMemory(buffer_length=buffer_length)


        obs, _ = env.reset()
        episode_rewards = []
        total_reward = 0

        for t in range(training_steps):
        
            if t <= warm_up:
                noisy_action = env.action_space.sample()
            else:
                action = policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                expl_noise = noise.sample()
                noisy_action = action.detach().cpu().numpy() + expl_noise.numpy().reshape(-1)
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
                agent.train(memory_buffer=memory, train_iteration=t, batch_size=batch_size,epochs=1)
            
            if done.item():
                episode_rewards.append(total_reward)
                print(f'Return for episode {len(episode_rewards)} is : {total_reward}')
                total_reward = 0
                obs, _ = env.reset()
            else:
                obs = obs_
        
        env.close()
        
        pi_loss_series = pd.Series(agent.pi_loss)
        policy_losses[f'cycle {cycles}'] = pi_loss_series
        q_loss_series = pd.Series(agent.q_loss)
        value_losses[f'cycle {cycles}'] = q_loss_series
        r = pd.Series(episode_rewards)
        training_returns[f'cycle {cycles}'] = r

    policy_losses['label'] = 'policy'
    value_losses['label'] = 'value'
    training_returns['label'] = 'returns'

    train_losses = pd.concat([policy_losses,value_losses, training_returns], ignore_index=True)
    train_losses.to_csv(f'Data/CSVs/{NOISE}Noise.csv')

    if PLOTTING:
        oh_ddpg = DDPGMetrics(file_path=f'Data/CSVs/{NOISE}Noise.csv', show=True, title=f'n ~ {NOISE}')
        oh_ddpg.plot_losses()
    