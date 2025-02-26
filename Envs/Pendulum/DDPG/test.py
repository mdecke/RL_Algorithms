import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt

import torch
import os

from numpy.random import seed
from Algorithms.DDPG import Policy

seed(42)

BEST_MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainedModels")
N_EPISODES = 10

env = gym.make("Pendulum-v1", render_mode='human')
obs, _ = env.reset()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

test_policy = Policy(state_dim=obs_dim, action_dim=action_dim)
test_policy.load_state_dict(torch.load(f=f'{BEST_MODEL_FOLDER}/Expert_policy.pth',
                                       map_location="cpu", weights_only=True))
test_policy.eval()

done = False
episodic_return = []
seeds = np.random.randint(0, 2**32 - 1, size=N_EPISODES)

for i in range(N_EPISODES):
    # print(f'Episode {i}')
    obs, _ = env.reset(seed=int(seeds[i]), options={'x_init': np.pi, 'y_init': 8.0})
    done = False
    cumulative_reward = 0

    while not done:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = test_policy(obs_tensor).cpu().numpy()

        obs_,r,trunc ,term, _ = env.step(action)
        cumulative_reward += r
        done = trunc or term
        obs = obs_.copy()
        if done:
            episodic_return.append(cumulative_reward) 
            # print(f'Episode {i} finished with return {cumulative_reward}')                

print("done")
env.close()
quit()
avged_returns = []
for i in range(0, len(episodic_return)-1):
    avged_returns.append(np.mean(episodic_return[0:i+1]))
avg_return = np.mean(episodic_return)

plt.plot(avged_returns, label='Avg Returns')
plt.axhline(y=avg_return, color='r', linestyle='--', label=f'Avg Return: {avg_return:.2f}')
plt.xlabel('Nb of episodes averaged')
plt.ylabel('Average Return')
plt.title('Average Return vs Episode')
plt.legend()
plt.grid()
plt.show()

