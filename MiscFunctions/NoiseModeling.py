import torch
import torch.nn as nn

import numpy as np


class ContinuousActionNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousActionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(32, action_dim)
        self.log_sigma_head = nn.Linear(32, action_dim)  

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma

class MLESampler:
    def __init__(self, weight_files:str, input_dim:int, output_dim:int, device: str=None):
        self.weight_file = weight_files
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_input = None
        self.device = device

        self.model_NN = ContinuousActionNN(state_dim=self.input_dim,
                                           action_dim=self.output_dim).to(self.device)
        self.model_NN.load_state_dict(torch.load(self.weight_file, map_location=self.device, weights_only=True,))
        self.model_NN.eval()

    def get_input(self, obs:np.ndarray, prev_action:np.ndarray=None, prev_obs:np.ndarray=None):
        cos_th, sin_th, theta_dot = obs[0], obs[1], obs[2]
        theta = np.arctan2(sin_th, cos_th)
        new_state = np.array([theta, theta_dot]).reshape(1, -1)
        if prev_action == None and prev_obs == None:
            input_mle = new_state
        elif prev_action.all() != None and prev_obs == None:
            input_mle = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
        elif prev_action.all() != None and prev_obs.all() != None:
            theta_prev = np.arctan2(prev_obs[0,1], prev_obs[0,0])
            theta_dot_prev = prev_obs[0,2]
            new_prev_state = np.array([theta_prev, theta_dot_prev]).reshape(1, -1)
            intermed = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
            input_mle = np.concatenate((intermed, new_prev_state), axis=1)
        else:
            raise ValueError("Invalid input type. Must be 'state' or 'state_action' or 'prev_state_action'")
        
        self.model_input = torch.tensor(input_mle, dtype=torch.float32, device=self.device)
    
    def sample(self, shape:int):
        if self.model_input is None:
            raise ValueError("Input not set. Please call get_input() method first.")
        # print(self.model_input)
        mu, log_sigma = self.model_NN(self.model_input)
        # print(mu, log_sigma)
        sigma = torch.exp(log_sigma) + 1e-5
        dist = torch.distributions.Normal(mu, sigma) # Add small value to avoid case where sigma is zero
        action = dist.sample()
        return action.reshape(shape)