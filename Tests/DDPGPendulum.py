import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.CustomPendulum import PendulumEnv
from Algorithms.DDPG import *
from MiscFunctions.Plotting import *


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
device = 'cpu'
NB_TRAINING_CYCLES = 5
NOISE = 'OrnsteinUhlenbeck' # 'Gaussian' or 'OrnsteinUhlenbeck'
PLOTTING = False


if __name__ == '__main__':
    
    if NOISE == 'Gaussian':
        noise = Normal(loc=0, scale=0.2)
    elif NOISE == 'OrnsteinUhlenbeck':
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=0.1)
    else:
        raise ValueError('Noise must be either Gaussian or OrnsteinUhlenbeck')
    
    env = PendulumEnv(dt=0.05,max_epsiode_steps=200, reward='dense')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

    training_steps = 20000
    warm_up = 1
    discount_gamma = 0.99
    buffer_length = 15000
    batch_size = 100

    list_of_all_the_data = []

    for cycles in range(NB_TRAINING_CYCLES):
        
        rng_state = np.random.get_state()
        seed_np = rng_state[1][0]
        seed_torch = torch.initial_seed()
        print(f'\nUsing seed {seed_np} for numpy and {seed_torch} for torch')

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

        for t in tqdm(range(training_steps), desc=f"Cycle {cycles+1}", unit="step"):
            with torch.no_grad():
                if t <= warm_up:
                    dist = torch.distributions.Uniform(low=action_low, high=action_high)
                    clipped_action = dist.sample()
                else:
                    action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                    expl_noise = noise.sample(action.shape)
                    noisy_action = action + expl_noise
                    clipped_action = torch.clip(noisy_action,
                                                min=action_low,
                                                max=action_high)
                
                obs_, reward, termination, truncation, _ = env.step(clipped_action)
                done = termination or truncation
                cumulative_reward += reward
                
                state_tensor = torch.tensor(obs,dtype=torch.float32,device=device)
                action_tensor = clipped_action.clone()
                reward_tensor = reward.clone()
                next_state_tensor = torch.tensor(obs_, dtype=torch.float32, device=device)
                done = torch.tensor(done, dtype=torch.int, device=device)
                
                memory.add_sample(state=state_tensor, action=action_tensor, reward=reward_tensor,
                                next_state=next_state_tensor, done=done)
            
            if t>=warm_up and len(memory.states) >= batch_size:
                agent.train(memory_buffer=memory, train_iteration=t, batch_size=batch_size,epochs=1)
            
            if done.item():
                episodic_returns.append(cumulative_reward.detach().cpu().numpy())
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
        
    df = pd.DataFrame(list_of_all_the_data)

    DATA_FOLDER = 'Data/CSVs/Metrics'
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    PLOTS_FOLDER = 'Data/Plots'
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)

    df.to_csv(f'{DATA_FOLDER}/{NOISE}.csv', index=False)

    print('Saved data to CSV')
    
    # Plotting
    if PLOTTING:
        print('Plotting...')
        plotter = DDPGMetrics(data=df, show=True, title=f'{NOISE} added noise', smooth=20)
    