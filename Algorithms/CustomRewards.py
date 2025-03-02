import numpy as np
import gymnasium as gym

class SparsePendulumRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, obs, action):
        # Take a step using the underlying environment
        cos_theta, sin_theta, thdot = obs[0], obs[1], obs[2]
        th = np.arctan2(sin_theta, cos_theta) 
        th = angle_normalize(th)
        cost = - (10*np.tanh(10*th**2) + 0.1*thdot**2 + 0.001*action**2)
        obs_, _, terminated, truncated, info = self.env.step(action)
        
        return obs_, cost.squeeze(), terminated, truncated, info

def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi