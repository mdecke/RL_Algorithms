import torch

import numpy as np
import pandas as pd
import gymnasium as gym

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.CustomPendulum import PendulumEnv
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
    
    env = PendulumEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    

    training_steps = 15000
    warm_up = 1
    discount_gamma = 0.99
    buffer_length = 15000
    batch_size = 100

    list_of_all_the_data = []

    for cycles in range(10):
        print(f'\n... training cycle {cycles+1}/{NB_TRAINING_CYCLES} ...')

        behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
        target_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
          
        behavior_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
        target_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)

        models = [behavior_policy, behavior_q]
        for model in models:
            init_model_weights(model)

        target_policy.load_state_dict(behavior_policy.state_dict())
        target_q.load_state_dict(behavior_q.state_dict())

        
        agent = DDPG(policy_network=behavior_policy, target_policy=target_policy, env=env,
                    value_network=behavior_q, target_value_function=target_q,
                    discount_factor=discount_gamma, total_training_time=training_steps, device=device)
        
        memory = DDPGMemory(buffer_length=buffer_length)


        obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
        episodic_returns = []
        cumulative_reward = 0

        for t in range(training_steps):
            if t <= warm_up:
                clipped_action = env.action_space.sample()
            else:
                action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                expl_noise = noise.sample()
                noisy_action = action + expl_noise
                clipped_action = np.clip(noisy_action.detach().cpu(), env.action_space.low, env.action_space.high)
            
            obs_, reward, termination, truncation, _ = env.step(clipped_action)
            done = termination + truncation
            cumulative_reward += reward
            
            state_tensor = torch.tensor(obs,dtype=torch.float32,device=device)
            action_tensor = torch.tensor(clipped_action,dtype=torch.float32, device=device)
            reward_tensor = torch.tensor(reward,dtype=torch.float32, device=device)
            next_state_tensor = torch.tensor(obs_, dtype=torch.float32, device=device)
            done = torch.tensor(done, dtype=torch.int, device=device)
            
            memory.add_sample(state=state_tensor, action=action_tensor, reward=reward_tensor,
                            next_state=next_state_tensor, done=done)
            
            if t>=warm_up and len(memory.states) >= batch_size:
                agent.train(memory_buffer=memory, train_iteration=t, batch_size=batch_size,epochs=1)
            
            if done.item():
                episodic_returns.append(cumulative_reward)
                print(f'Return for episode {len(episodic_returns)} is : {cumulative_reward}')
                cumulative_reward = 0
                obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
            else:
                obs = obs_.copy()
        list_of_all_the_data.append([cycles,
                                     agent.pi_loss,
                                     agent.q_loss,
                                     episodic_returns])
        env.close()
        
    col_names = ['cycle', 'policy_loss', 'q_loss', 'return']
    df = pd.DataFrame(list_of_all_the_data, columns=col_names)

    DATA_FOLDER = 'Data/CSVs/Metrics'
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    PLOTS_FOLDER = 'Data/Plots'
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)

    df.to_csv(f'{DATA_FOLDER}/{NOISE}.csv', index=False)

    if PLOTTING:
        oh_ddpg = DDPGMetrics(file_path=f'{DATA_FOLDER}/{NOISE}Noise.csv', show=True, title=f'n ~ {NOISE}')
        oh_ddpg.plot_losses()
    