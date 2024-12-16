import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CUSTOM_NOISE_FILE_PATH = '/Users/mathieudecker/PersonalProjects/RL_Algorithms/Action_dist_train_losses.csv'
NORMAL_FILE_PATH = '/Users/mathieudecker/PersonalProjects/RL_Algorithms/Train_losses.csv'


custom_noise_df = pd.read_csv(CUSTOM_NOISE_FILE_PATH)
normal_noise_df = pd.read_csv(NORMAL_FILE_PATH)

custom_noise_policy_losses = custom_noise_df[custom_noise_df['label'] == 'policy'].iloc[:,1:-1]
custom_noise_value_losses = custom_noise_df[custom_noise_df['label'] == 'value'].iloc[:,1:-1]
custom_returns = custom_noise_df[custom_noise_df['label'] == 'returns'].iloc[:,1:-1]

normal_noise_policy_losses = normal_noise_df[normal_noise_df['label'] == 'policy'].iloc[:,1:-1]
normal_noise_value_losses = normal_noise_df[normal_noise_df['label'] == 'value'].iloc[:,1:-1]
normal_returns = normal_noise_df[normal_noise_df['label'] == 'returns'].iloc[:,1:-1]


custom_noise_policy_losses['min'] = custom_noise_policy_losses.min(axis=1)
custom_noise_policy_losses['max'] = custom_noise_policy_losses.max(axis=1)
custom_noise_policy_losses['mean'] = custom_noise_policy_losses.mean(axis=1)

custom_noise_value_losses['min'] = custom_noise_value_losses.min(axis=1)
custom_noise_value_losses['max'] = custom_noise_value_losses.max(axis=1)
custom_noise_value_losses['mean'] = custom_noise_value_losses.mean(axis=1)

custom_returns['min'] = custom_returns.min(axis=1)
custom_returns['max'] = custom_returns.max(axis=1)
custom_returns['mean'] = custom_returns.mean(axis=1)


normal_noise_policy_losses['min'] = normal_noise_policy_losses.min(axis=1)
normal_noise_policy_losses['max'] = normal_noise_policy_losses.max(axis=1)
normal_noise_policy_losses['mean'] = normal_noise_policy_losses.mean(axis=1)

normal_noise_value_losses['min'] = normal_noise_value_losses.min(axis=1)
normal_noise_value_losses['max'] = normal_noise_value_losses.max(axis=1)
normal_noise_value_losses['mean'] = normal_noise_value_losses.mean(axis=1)

normal_returns['min'] = normal_returns.min(axis=1)
normal_returns['max'] = normal_returns.max(axis=1)
normal_returns['mean'] = normal_returns.mean(axis=1)


plt.figure(figsize=(15, 8))

# Q-function loss subplot
plt.subplot(3, 1, 1)
plt.plot(range(13000),custom_noise_value_losses['mean'], label='custom noise Q-value Loss avg', color='blue')
plt.fill_between(x=range(len(custom_noise_value_losses)),
                y1=custom_noise_value_losses['min'],
                y2=custom_noise_value_losses['max'],
                color='blue', alpha=0.2)
plt.plot(range(13000),normal_noise_value_losses['mean'], label='normal noise Q-value Loss avg', color='red')
plt.fill_between(x=range(len(normal_noise_value_losses)),
                y1=normal_noise_value_losses['min'],
                y2=normal_noise_value_losses['max'],
                color='red', alpha=0.2)
plt.ylabel('Value Loss')
plt.xlabel('Training Steps')
plt.title('Q-function Loss')
plt.grid(True)
plt.legend()

# Policy loss subplot
plt.subplot(3, 1, 2)
plt.plot(custom_noise_policy_losses['mean'], label='custom noise Policy Loss avg', color='blue')
plt.fill_between(x=range(len(custom_noise_policy_losses['min'])),
                y1=custom_noise_policy_losses['min'],
                y2=custom_noise_policy_losses['max'],
                color='blue', alpha=0.2)
plt.plot(normal_noise_policy_losses['mean'], label='normal noise Policy Loss avg', color='red')
plt.fill_between(x=range(len(normal_noise_policy_losses['min'])),
                y1=normal_noise_policy_losses['min'],
                y2=normal_noise_policy_losses['max'],
                color='red', alpha=0.2)
plt.ylabel('policy Loss')
plt.xlabel('Training Steps')
plt.title('Policy Loss')
plt.grid(True)
plt.legend()

# Episodic return subplot (no smoothing here, just raw)
plt.subplot(3, 1, 3)
plt.plot(range(75),custom_returns['mean'], label='avg episodic return custom', color='blue')
plt.fill_between(x=range(75),
                y1=custom_returns['min'],
                y2=custom_returns['max'],
                color='blue', alpha=0.2)
plt.plot(range(75),normal_returns['mean'], label='normal avg episodic retrun', color='red')
plt.fill_between(x=range(75),
                y1=normal_returns['min'],
                y2=normal_returns['max'],
                color='red', alpha=0.2)
plt.ylabel('retrun')
plt.xlabel('episodes')
plt.title('Episodes')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()


