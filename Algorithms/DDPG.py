import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class DDPGMemory:
    def __init__(self, buffer_length:int):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.memory_buffer_length = buffer_length
    
    def add_sample(self, state:torch.Tensor, action:torch.Tensor,
                   reward:torch.Tensor, next_state:torch.Tensor, done:torch.Tensor):
        
        if len(self.states) == self.memory_buffer_length:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample_memory(self, batch_size):
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))
        batch = [(self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i]) for i in indices]
        return zip(*batch)
    

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, policy_lr):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear_layer_1 = nn.Linear(self.state_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), policy_lr)
        self.to(device)
    
    def preprocess(self, policy_input, train=True):
        return policy_input
        
    def forward(self, inputs,preprocess=False):
        if preprocess:
            inputs = self.preprocess(inputs)
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
        
        return 2 * torch.tanh(self.action_layer(x))


class Value(nn.Module):
    def __init__(self, state_dim, action_dim, value_lr):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear_layer_1 = nn.Linear(self.state_dim + self.action_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), value_lr)
        self.to(device)
    
    def preprocess(self, value_input, train=True):
        return value_input
    
    def forward(self, inputs, preprocess=False):
        if preprocess:
            inputs = self.preprocess(inputs)
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x).squeeze()


class DDPG:
    def __init__(self, policy_network:Policy, target_policy:Policy,env:gym.Env,
                 value_network:Value, target_value_function:Value, discount_factor:float,
                 total_training_time:int):
        
        self.pi = policy_network.to(device=device)
        self.pi_t = target_policy.to(device=device)
        self.q = value_network.to(device=device)
        self.q_t = target_value_function.to(device=device)
        self.gamma = discount_factor
        self.T = total_training_time
        self.env = env
        self.pi_loss = []
        self.q_loss = []

    
    def soft_update(self,target_network, network, tau):
        for target_param, source_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    
    def train(self,memory_buffer:DDPGMemory, train_iteration:int, batch_size:int, epochs:int):
        for epoch in range(epochs):
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = memory_buffer.sample_memory(batch_size)

            #batch to device
            sampled_states = torch.stack(sampled_states).to(device)
            sampled_actions = torch.stack(sampled_actions).to(device)
            sampled_rewards = torch.stack(sampled_rewards).to(device)
            sampled_next_states = torch.stack(sampled_next_states).to(device)
            sampled_dones = torch.stack(sampled_dones).to(device)
            
            # compute target values
            with torch.no_grad():
                next_actions = self.pi_t.forward(sampled_next_states)
                next_state_action_pairs = torch.cat([sampled_next_states, next_actions], dim=1)
                target_q_values = self.q_t.forward(next_state_action_pairs)
                y = sampled_rewards + self.gamma * (1 - sampled_dones) * target_q_values

            # compute critic loss
            state_action_pairs = torch.cat([sampled_states, sampled_actions], dim=1)
            critic_values = self.q.forward(state_action_pairs)

            critic_loss = functional.mse_loss(critic_values, y)

            # optimization step (critic)
            self.q.optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.q.optimizer.step()

            # compute policy (actor) loss
            actions = self.pi.forward(sampled_states)
            state_action_pairs = torch.cat([sampled_states, actions], dim=1)
            critic_values = self.q.forward(state_action_pairs)

            policy_loss = -critic_values.mean()

            # optimization step (policy)
            self.pi.optimizer.zero_grad()
            policy_loss.backward()
            # nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            self.pi.optimizer.step()

            # update target networks
            self.soft_update(self.pi_t, self.pi, tau=0.005)
            self.soft_update(self.q_t, self.q, tau=0.005)

            # record data
            # self.track_data("Loss / Policy loss", policy_loss.item())
            # self.track_data("Loss / Critic loss", critic_loss.item())

            # self.track_data("Q-network / Q1 (max)", torch.max(critic_values).item())
            # self.track_data("Q-network / Q1 (min)", torch.min(critic_values).item())
            # self.track_data("Q-network / Q1 (mean)", torch.mean(critic_values).item())

            # self.track_data("Target / Target (max)", torch.max(y).item())
            # self.track_data("Target / Target (min)", torch.min(y).item())
            # self.track_data("Target / Target (mean)", torch.mean(y).item())
        
        self.pi_loss.append(policy_loss)
        self.q_loss.append(critic_loss)
        
        if train_iteration % 100 == 0:
            print(f'timestep {train_iteration}/{self.T}: Policy loss = {self.pi_loss[-1]} || Value loss = {self.q_loss[-1]}')
        
        


if __name__ == '__main__':
    np.random.seed(42)

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
    policy_tilde = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
    policy_tilde.load_state_dict(policy.state_dict())
    
    value = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
    value_tilde = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
    value_tilde.load_state_dict(value.state_dict())

    training_steps = 15000
    discount_gamma = 0.99
    buffer_length = 1000
    batch_size = 128
    
    agent = DDPG(policy_network=policy, target_policy=policy_tilde, env=env,
                 value_network=value, target_value_function=value_tilde,
                 discount_factor=discount_gamma, total_training_time=training_steps)
    
    memory = DDPGMemory(buffer_length=buffer_length)


    noise = OUNoise(action_dim=action_dim)

    obs, _ = env.reset()
    episode_rewards = []
    total_reward = 0

    for t in range(training_steps):
        obs = torch.tensor(obs).to(device=device)
        action = policy.forward(obs)
        noisy_action = action.detach().cpu().numpy() + noise.sample()
        noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
        obs_, reward, termination, truncation, _ = env.step(noisy_action)
        done = termination + truncation
        total_reward += reward
        
        state_tensor = torch.tensor(obs,dtype=torch.float32)
        action_tensor = torch.tensor(noisy_action,dtype=torch.float32)
        reward_tensor = torch.tensor(reward,dtype=torch.float32)
        next_state_tensor = torch.tensor(obs_, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.int)
        
        memory.add_sample(state=state_tensor, action=action_tensor, reward=reward_tensor,
                          next_state=next_state_tensor, done=done)
        
        if len(memory.states) >= batch_size:
            agent.train(memory_buffer=memory, train_iteration=t, batch_size=batch_size,epochs=1)
        
        if done.item():
            episode_rewards.append(total_reward)
            print(f'Return for episode {len(episode_rewards)} is : {total_reward}')
            total_reward = 0
            obs, _ = env.reset()
        else:
            obs = obs_
        
plt.plot(agent.q_loss.cpu())
plt.ylabel('Value function approximator loss')
plt.xlabel('training steps')
plt.grid()
plt.show() 

