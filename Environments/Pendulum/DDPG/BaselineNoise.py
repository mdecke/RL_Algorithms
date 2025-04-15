import gymnasium as gym
import torch
import gymnasium as gym

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from Algorithms.DDPG import *
from Algorithms.CustomRewards import SparsePendulumRewardWrapper
from MiscFunctions.Plotting import *
import datetime


def train(args):
    """
    Train a DDPG agent on Pendulum-v1.
    """
    np.random.seed(42)
    # Unpack all arguments from args
    env_name           = args.env_name
    NB_TRAINING_CYCLES = args.NB_TRAINING_CYCLES
    NOISE              = args.NOISE
    REWARD_TYPE        = args.REWARD_TYPE
    PLOTTING           = args.PLOTTING
    device             = args.device
    training_steps     = args.training_steps
    warm_up            = args.warm_up
    discount_gamma     = args.discount_gamma
    buffer_length      = args.buffer_length
    batch_size         = args.batch_size

    data_folder        = f"Metrics/{datetime.date.today()}"
    best_model_folder  = f"TrainedModels/{datetime.date.today()}"

    BEST_SO_FAR = -np.inf

#__________________________________________________ Creating data folders ____________________________________________________
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, data_folder)
    best_model_folder = os.path.join(current_dir, best_model_folder)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(best_model_folder, exist_ok=True)

#______________________________________________________ Environment setup ____________________________________________________
    
    env = gym.make(env_name) #, render_mode='human')
    # env.unwrapped.m = 5.0
    

    if REWARD_TYPE == 'sparse':
        env = SparsePendulumRewardWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    if NOISE == 'Gaussian':
        noise = Normal(loc=0, scale=0.2)
    elif NOISE == 'OrnsteinUhlenbeck':
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.1)
    else:
        raise ValueError('Invalid noise type. Choose "Gaussian" or "OrnsteinUhlenbeck".')
    
    
    seeds = np.random.randint(0, 2**32 - 1, size=NB_TRAINING_CYCLES, )
    print(f"[INFO] Cycle seeds: {seeds}")

#______________________________________________________ Training loop ____________________________________________________
    
    list_of_all_the_data = []

    for cycle_idx in range(NB_TRAINING_CYCLES):
        torch.manual_seed(seeds[cycle_idx])

        behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim)
        target_policy = Policy(state_dim=state_dim, action_dim=action_dim)
        behavior_q = Value(state_dim=state_dim, action_dim=action_dim)
        target_q = Value(state_dim=state_dim, action_dim=action_dim)

        # New behaviorPolicy and behaviorQ network weights for each cycle: N(0, 0.1)
        init_model_weights(behavior_policy, seed=seeds[cycle_idx])
        init_model_weights(behavior_q, seed=seeds[cycle_idx])

        # PolicyWeights_t <-- PolicyWeights_b | QWeights_t <-- QWeights_b
        target_policy.load_state_dict(behavior_policy.state_dict())
        target_q.load_state_dict(behavior_q.state_dict())

        agent = DDPG(
            policy_network=behavior_policy,
            target_policy=target_policy,
            value_network=behavior_q,
            target_value_function=target_q,
            discount_factor=discount_gamma,
            seed=seeds[cycle_idx],
            device=device,
        )

        memory = DDPGMemory(state_dim=state_dim, action_dim=action_dim, buffer_length=buffer_length)

        obs, _ = env.reset(seed=int(seeds[cycle_idx]),options={'x_init': np.pi, 'y_init': 8.0})
        episodic_returns = []
        cumulative_reward = 0

        progress_bar = tqdm(range(training_steps), desc=f"Cycle {cycle_idx+1}/{NB_TRAINING_CYCLES}", unit="step")
        for t in progress_bar:
            if t < warm_up:
                clipped_action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                    expl_noise = noise.sample(action.shape).cpu().numpy()
                noisy_action = action.cpu().numpy() + expl_noise
                clipped_action = np.clip(noisy_action, a_min=action_low, a_max=action_high)

            if REWARD_TYPE == 'sparse':
                obs_, reward, terminated, truncated, _ = env.step(obs, clipped_action)
            else:
                obs_, reward, terminated, truncated, _ = env.step(clipped_action)
            
            done = terminated or truncated

            cumulative_reward += reward
            memory.add_sample(state=obs, action=clipped_action, reward=reward, next_state=obs_, done=done)


            if (t >= warm_up) and (len(memory.states) >= batch_size):
                agent.train(memory_buffer=memory, batch_size=batch_size, epochs=1)

            if done:
                episodic_returns.append(cumulative_reward)
                if cumulative_reward > BEST_SO_FAR:
                    BEST_SO_FAR = cumulative_reward
                    torch.save(behavior_policy.state_dict(), f"{best_model_folder}/{NOISE}_{REWARD_TYPE}.pth")

                cumulative_reward = 0
                obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
            else:
                obs = obs_.copy()

        # Collect stats
        for i in range(len(agent.pi_loss)):
            list_of_all_the_data.append({
                'cycle': cycle_idx + 1,
                'step': i,
                'policy_loss': agent.pi_loss[i],
                'q_loss': agent.q_loss[i],
                'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
            })

    env.close()

#______________________________________________________ Save data ____________________________________________________

    df = pd.DataFrame(list_of_all_the_data)
    csv_filename = f"{NOISE}_{REWARD_TYPE}.csv"
    df.to_csv(os.path.join(data_folder, csv_filename), index=False)
    print(f"[INFO] Metrics saved to: {os.path.join(data_folder, csv_filename)}")

    
    if PLOTTING:
        print("[INFO] Plotting metrics...")
        fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))
        plotter = DDPGMetrics(data=df, show=False, title=f"{NOISE} Noise", smooth=2)
        plotter.plot_losses(ax=ax)
        plt.tight_layout()
        plt.show()
