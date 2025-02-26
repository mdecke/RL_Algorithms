import argparse
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from Algorithms.DDPG import *
from MiscFunctions.Plotting import *
from MiscFunctions.NoiseModeling import MLESampler

def train(args):
    """
    Train a DDPG agent on Pendulum-v1.
    """

    # Unpack all arguments from args
    env_name           = args.env_name
    NB_TRAINING_CYCLES = args.NB_TRAINING_CYCLES
    EXPERT             = args.EXPERT
    REWARD_TYPE        = args.REWARD_TYPE
    PLOTTING           = args.PLOTTING
    device             = args.device
    training_steps     = args.training_steps
    warm_up            = args.warm_up
    discount_gamma     = args.discount_gamma
    buffer_length      = args.buffer_length
    batch_size         = args.batch_size

    data_folder        = "Metrics"
    best_model_folder  = "TrainedModels"

    BEST_SO_FAR = -np.inf

#__________________________________________________ Creating data folders ____________________________________________________
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, data_folder)
    best_model_folder = os.path.join(current_dir, best_model_folder)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(best_model_folder, exist_ok=True)

#______________________________________________________ Environment setup ____________________________________________________
    
    env = gym.make(env_name) #, render_mode='human')
    
    if REWARD_TYPE == 'sparse':
        env = SparseRewardWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    noise = MLESampler(weight_files='../Data/Models/Noise/P(a|s).pth',
                       input_dim=state_dim-1, output_dim=action_dim, device=device)
    assert isinstance(noise, MLESampler)
    
    seeds = np.random.randint(0, 2**32 - 1, size=NB_TRAINING_CYCLES)

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

        obs, _ = env.reset()
        episodic_returns = []
        cumulative_reward = 0

        progress_bar = tqdm(range(training_steps), desc=f"Cycle {cycle_idx+1}/{NB_TRAINING_CYCLES}", unit="step")
        for t in progress_bar:
            if t <= warm_up:
                clipped_action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                    noise.get_input(obs=obs)
                    expl_noise = noise.sample(action.shape).cpu().numpy()
                
                if EXPERT == 'Expert':
                    action_bh = np.clip(action.cpu().numpy(), a_min=action_low, a_max=action_high)
                    if REWARD_TYPE == 'sparse':
                        _, r_bpol, _, _, _ = env.step(obs, action_bh)
                    else:
                        _, r_bpol, _, _, _ = env.step(action_bh)
                    noisy_action = expl_noise
                else:
                    noisy_action = action.cpu().numpy() + expl_noise
                
                clipped_action = np.clip(noisy_action, a_min=action_low, a_max=action_high)

            if REWARD_TYPE == 'sparse':
                obs_, reward, done, truncated, _ = env.step(obs, clipped_action)
            else:
                obs_, reward, done, truncated, _ = env.step(clipped_action)
            
            done = done or truncated
            
            if t <= warm_up or EXPERT == 'Noise':
                cumulative_reward += reward
            else:
                cumulative_reward += r_bpol
            
            memory.add_sample(state=obs, action=clipped_action, reward=reward, next_state=obs_, done=done)


            if (t >= warm_up) and (len(memory.states) >= batch_size):
                agent.train(memory_buffer=memory, batch_size=batch_size, epochs=1)

            if done:
                episodic_returns.append(cumulative_reward)
                if cumulative_reward > BEST_SO_FAR:
                    BEST_SO_FAR = cumulative_reward
                    torch.save(behavior_policy.state_dict(), f"{best_model_folder}/Custom{EXPERT}_{REWARD_TYPE}.pth")

                cumulative_reward = 0
                obs, _ = env.reset()
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
    csv_filename = f"Custom{EXPERT}_{REWARD_TYPE}.csv"
    df.to_csv(os.path.join(data_folder, csv_filename), index=False)
    print(f"[INFO] Metrics saved to: {os.path.join(data_folder, csv_filename)}")

    
    if PLOTTING:
        print("[INFO] Plotting metrics...")
        fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))
        plotter = DDPGMetrics(data=df, show=False, title=f"Custom {EXPERT}", smooth=2)
        plotter.plot_losses(ax=ax)
        plt.tight_layout()
        plt.show()


# import torch

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# import os
# from Algorithms.DDPG import *
# from MiscFunctions.Plotting import *
# from MiscFunctions.NoiseModeling import MLESampler


# # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# # print(f"Using device: {device}")
# device = 'cpu'

# NB_TRAINING_CYCLES = 5
# BEST_SO_FAR = -np.inf

# REWARD_TYPE = 'dense' # 'sparse' or 'dense'
# TYPE = 'Expert' # 'Expert' or 'Noise'

# DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Metrics")
# BEST_MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainedModels")

# PLOTTING = True


# if __name__ == '__main__':
#     GRAVITY = 10.0
#     os.makedirs(DATA_FOLDER, exist_ok=True) 
#     os.makedirs(BEST_MODEL_FOLDER, exist_ok=True) 

#     env = gym.make("Pendulum-v1") #, render_mode = 'human')

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
    
#     action_low = env.action_space.low
#     action_high = env.action_space.high

#     training_steps = 15000
#     warm_up = 1
#     discount_gamma = 0.99
#     buffer_length = 15000
#     batch_size = 100
    
    
#     noise = MLESampler(weight_files='Data/Models/Noise/P(a|s).pth',
#                        input_dim=state_dim-1, output_dim=action_dim, device=device)
#     assert isinstance(noise, MLESampler)
    
    
#     if REWARD_TYPE == 'sparse':
#         env = SparseRewardWrapper(env)


#     list_of_all_the_data = []
#     seeds = np.random.randint(0, 2**32 - 1, size=NB_TRAINING_CYCLES)
#     print(f'\nUsing seeds {seeds} for training')

#     for cycles in range(NB_TRAINING_CYCLES):
#         torch.manual_seed(seeds[cycles])

#         behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
#         target_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
          
#         behavior_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
#         target_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)

#         models = [behavior_policy, behavior_q]
#         for model in models:
#             init_model_weights(model, seed=seeds[cycles])

#         target_policy.load_state_dict(behavior_policy.state_dict())
#         target_q.load_state_dict(behavior_q.state_dict())

        
#         agent = DDPG(policy_network=behavior_policy, target_policy=target_policy,
#                     value_network=behavior_q, target_value_function=target_q,
#                     discount_factor=discount_gamma, seed=seeds[cycles], device=device)
        
#         memory = DDPGMemory(state_dim=state_dim, action_dim=action_dim, buffer_length=buffer_length)


#         obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
#         episodic_returns = []
#         cumulative_reward = 0

#         for t in tqdm(range(training_steps), desc=f"Cycle {cycles+1}", unit="step"):       
#             if t <= warm_up:
#                 clipped_action = env.action_space.sample()
#             else:
#                 with torch.no_grad():
#                     action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
#                     if EXPERT:
#                         assert isinstance(noise, MLESampler)
#                         noise.get_input(obs=obs)
#                         noisy_action = noise.sample(shape=action_dim)
#                         noisy_action = noisy_action.cpu().numpy()
#                         action_bh = np.clip(action,
#                                                 a_min=action_low,
#                                                 a_max=action_high)
#                         _,r_bpol,_,_,_ = env.step(obs, action_bh)
#                     else:
#                         noise.get_input(obs=obs)
#                         expl_noise = noise.sample(action.shape)
#                         noisy_action = action.cpu().numpy() + expl_noise.cpu().numpy()

#                 clipped_action = np.clip(noisy_action,
#                                                 a_min=action_low,
#                                                 a_max=action_high)
            
#             obs_, reward, termination, truncation, _ = env.step(clipped_action) #obs,clipped_action
#             done = termination or truncation
            
#             if t<=warm_up or not EXPERT:
#                 cumulative_reward += reward
#             else:
#                 cumulative_reward += r_bpol
            
#             memory.add_sample(state=obs, action=clipped_action, reward=reward, next_state=obs_, done=done)
            
#             if t>=warm_up and len(memory.states) >= batch_size:
#                 agent.train(memory_buffer=memory, batch_size=batch_size, epochs=1)
            
#             if done:
#                 episodic_returns.append(cumulative_reward)
#                 if cumulative_reward > BEST_SO_FAR:
#                     best_return_so_far = cumulative_reward
#                 torch.save(behavior_policy.state_dict(), f'{BEST_MODEL_FOLDER}/Expert_policy_{REWARD_TYPE}.pth')

#                 cumulative_reward = 0
#                 obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
#             else:
#                 obs = obs_.copy()
        
#         for i in range(len(agent.pi_loss)):
#             list_of_all_the_data.append({
#                 'cycle': cycles + 1,
#                 'policy_loss': agent.pi_loss[i],
#                 'q_loss': agent.q_loss[i],
#                 'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
#             })
        
#     env.close()

# print(f'\n Saving data ...')
# df = pd.DataFrame(list_of_all_the_data)
# os.makedirs(DATA_FOLDER, exist_ok=True)    
# df.to_csv(f'{DATA_FOLDER}/Custom{TYPE}_{REWARD_TYPE}.csv', index=False)
# print('... Done')

# # Plotting
# if PLOTTING:
#     print('Plotting...')
#     fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))
#     plotter = DDPGMetrics(data=df, show=False, title=f'Custom {TYPE}', smooth=2)
#     plotter.plot_losses(ax=ax)
#     plt.tight_layout()
#     plt.show()
    