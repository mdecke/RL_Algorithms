import os
import torch
import gymnasium as gym

import numpy as np
import pandas as pd
from tqdm import tqdm

from Algorithms.DDPG import *
from MiscFunctions.Plotting import *




# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
device = 'cpu'
NB_TRAINING_CYCLES = 1
NOISE = 'Gaussian' # 'Gaussian' or 'OrnsteinUhlenbeck'
PLOTTING = True
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Metrics")


if __name__ == '__main__':

    if NOISE == 'Gaussian':
        noise = Normal(loc=0, scale=0.2)
    elif NOISE == 'OrnsteinUhlenbeck':
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=0.1)
    else:
        raise ValueError('Noise must be either Gaussian or OrnsteinUhlenbeck')

    env = gym.make("Ant-v4")
    obs_, _ = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_low = env.action_space.low
    action_high = env.action_space.high


    training_steps = int(1e6)
    warm_up = int(1e4)
    discount_gamma = 0.98 #0.98
    buffer_length = int(2e5)
    batch_size = 100 #100

    list_of_all_the_data = []

    for cycles in range(NB_TRAINING_CYCLES):
        seed_torch = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(seed_torch)
        seed_np = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed_np)
        print(f'\nUsing seed {seed_np} for numpy and {seed_torch} for torch')

        behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
        target_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
          
        behavior_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
        target_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)

        models = [behavior_policy, behavior_q]
        for model in models:
            init_model_weights(model, seed=seed_torch)

        target_policy.load_state_dict(behavior_policy.state_dict())
        target_q.load_state_dict(behavior_q.state_dict())

        
        agent = DDPG(policy_network=behavior_policy, target_policy=target_policy, env=env,
                    value_network=behavior_q, target_value_function=target_q,
                    discount_factor=discount_gamma, total_training_time=training_steps,seed=seed_torch, device=device)
        
        memory = DDPGMemory(state_dim=state_dim, action_dim=action_dim, buffer_length=buffer_length)


        obs, _ = env.reset()
        episodic_returns = []
        cumulative_reward = 0

        for t in tqdm(range(training_steps), desc=f"Cycle {cycles+1}", unit="step"):
            with torch.no_grad():
                if t <= warm_up:
                    clipped_action = env.action_space.sample()
                else:
                    action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                    expl_noise = noise.sample(action.shape)
                    noisy_action = action.cpu().numpy() + expl_noise.cpu().numpy()
                    clipped_action = np.clip(noisy_action,
                                                a_min=action_low,
                                                a_max=action_high)
                
                obs_, reward, termination, truncation, _ = env.step(clipped_action)
                done = termination or truncation
                cumulative_reward += reward
                
                memory.add_sample(state=obs, action=clipped_action, reward=reward, next_state=obs_, done=done)
            
            if t>=warm_up and len(memory.states) >= batch_size:
                agent.train(memory_buffer=memory, train_iteration=t, batch_size=batch_size,epochs=1)
            
            if done:
                episodic_returns.append(cumulative_reward)
                cumulative_reward = 0
                obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
            else:
                obs = obs_.copy()
        
        for i in range(len(agent.pi_loss)):
            list_of_all_the_data.append({
                'cycle': cycles + 1,
                'policy_loss': agent.pi_loss[i],
                'q_loss': agent.q_loss[i],
                'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
            })
        
        env.close()
    
    print('... Saving data ...')
   
    df = pd.DataFrame(list_of_all_the_data)
    os.makedirs(DATA_FOLDER, exist_ok=True)    
    df.to_csv(f'{DATA_FOLDER}/{NOISE}.csv', index=False)

    print('... Saved data to CSV ...')
    
    # Plotting
    if PLOTTING:
        print('Plotting...')
        fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))
        plotter = DDPGMetrics(data=df, show=False, title=f'Ant V4 DDPG', smooth=2)
        plotter.plot_losses(ax=ax)
        plt.tight_layout()
        plt.show()
    