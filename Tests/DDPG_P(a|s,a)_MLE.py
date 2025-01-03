import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd 
import gym 

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

from Algorithms.DDPG import *
from MiscFunctions.Plotting import *
from MiscFunctions.DataProcessing import *
from MiscFunctions.SimulationData import get_state


FILE_PATH = 'Data/data_1000.csv'
NB_CLUSTERS = 50
NB_TRIALS = 100
NB_TRAINING_CYCLES = 10
DATA_TYPE = 'state_action' 
PLOTTING = False
training_steps = 15000
warm_up = 2000
discount_gamma = 0.99
buffer_length = 6000
batch_size = 128

epochs = 100
nb_of_batches = 32
nb_trajectories = 20


if __name__ == '__main__':
    np.random.seed(42)
    
    print('... fitting P(a|s,a-1) ...\n')

    clusterer = Cluster(file_path=FILE_PATH, nb_clusters=5)
    extended_data = clusterer._add_previous_action(return_df=True)
    extended_data['state'] = extended_data['state'].apply(lambda x: np.array(ast.literal_eval(x)))
    episode_idx = extended_data['episode'].to_numpy()
    episode_idx = torch.tensor(episode_idx, dtype=torch.float32)

    env = gym.make('Pendulum-v1')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0] - 1 + action_dim

    model = ContinuousActionNN(state_dim, action_dim)

    train_nn(model, epochs=epochs,nb_batch=nb_of_batches, nb_trajectories=nb_trajectories, data=extended_data, data_type=DATA_TYPE)

    model.eval()

    print('... Distribution fitted ... \n')
    

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy_losses = pd.DataFrame()
    value_losses = pd.DataFrame()
    training_returns = pd.DataFrame()
    
    for cycles in range(NB_TRAINING_CYCLES):
        print(f'... training cycle {cycles}/{NB_TRAINING_CYCLES} ...')

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
        exploration_noise = []
        warm_up_actions = []
        prev_action = 0

        for t in range(training_steps):
            
            state = get_state(obs)
            augemented_state = np.append(state, prev_action)
            obs = torch.tensor(obs).to(device=device)
            input = torch.tensor(augemented_state, dtype=torch.float32).to(device=device)
            expected_action,_ =  model(input)
            if t < warm_up:
                noisy_action = env.action_space.sample()
            else:
                action = policy.forward(obs)
                cpu_action = action.detach().cpu().numpy()
                noisy_action = cpu_action + expected_action.detach().cpu().numpy()
  
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
            obs_, reward, termination, truncation, _ = env.step(noisy_action)
            done = termination + truncation
            total_reward += reward
            
            state_tensor = obs.clone().detach().requires_grad_(True).to(device=device)
            action_tensor = torch.tensor(noisy_action,dtype=torch.float32, device=device)
            reward_tensor = torch.tensor(reward,dtype=torch.float32, device=device)
            next_state_tensor = torch.tensor(obs_, dtype=torch.float32, device=device)
            done = torch.tensor(done, dtype=torch.int, device=device)
            
            memory.add_sample(state=state_tensor, action=action_tensor, reward=reward_tensor,
                            next_state=next_state_tensor, done=done)
            
            if t>warm_up and len(memory.states) >= batch_size:
                agent.train(memory_buffer=memory, train_iteration=t, batch_size=batch_size,epochs=1)
                prev_action = noisy_action

            
            if done.item():
                episode_rewards.append(total_reward)
                print(f'Return for episode {len(episode_rewards)} is : {total_reward}')
                total_reward = 0
                obs, _ = env.reset()
                prev_action = 0
            else:
                obs = obs_
        
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
    train_losses.to_csv('Data/P(a|s,a)NoiseTraining_MLE.csv')

    if PLOTTING:
        pasa_ddpg = DDPGMetrics(file_path='Data/P(a|s,a)NoiseTraining_MLE.csv', show=True, title='P(a[k]|s[k],a[k-1]) Noise')
        pasa_ddpg.plot_losses()


