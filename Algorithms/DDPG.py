import gymnasium as gym

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.distributions import Normal


class OrnsteinUhlenbeckNoise:
    def __init__(self,theta: float,sigma: float,base_scale: float,mean: float = 0,std: float = 1) -> None:
        super().__init__()
        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.distribution = Normal(loc=torch.tensor(mean, dtype=torch.float32),
                                   scale=torch.tensor(std, dtype=torch.float32))

    def sample(self, size:torch.Size = torch.Size([1,1])) -> torch.Tensor:
        if hasattr(self.state, "shape") and self.state.shape != torch.Size(size):
            self.state = 0
        self.state += -self.state * self.theta + self.sigma * self.distribution.sample(size)

        return self.base_scale * self.state
    

class DDPGMemory:
    def __init__(self, buffer_length:int):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.memory_buffer_length = buffer_length
    
    def add_sample(self, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor,
                   next_state:torch.Tensor, done:torch.Tensor):
        
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
        batch = [(self.states[i], self.actions[i], self.rewards[i], self.next_states[i], 
                  self.dones[i]) for i in indices]
        return zip(*batch)
    

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, policy_lr, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    
        self.linear_layer_1 = nn.Linear(self.state_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), policy_lr)
        self.to(device)
        
    def forward(self, inputs):
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
        
        return 2 * torch.tanh(self.action_layer(x))


class Value(nn.Module):
    def __init__(self, state_dim, action_dim, value_lr, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear_layer_1 = nn.Linear(self.state_dim + self.action_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), value_lr)
        self.to(device)
    
    def forward(self, inputs):
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x).squeeze()

def init_model_weights(model:nn.Module, mean=0.0, std=0.1):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "weight" in name:
                nn.init.normal_(param, mean=mean, std=std)
            elif "bias" in name:
                nn.init.normal_(param, mean=mean, std=std)

class DDPG:
    def __init__(self, policy_network:Policy, target_policy:Policy,env:gym.Env,
                 value_network:Value, target_value_function:Value, discount_factor:float,
                 total_training_time:int, device='cpu'):
        
        self.pi = policy_network.to(device=device)
        self.pi_t = target_policy.to(device=device)
        self.q = value_network.to(device=device)
        self.q_t = target_value_function.to(device=device)
        self.gamma = discount_factor
        self.T = total_training_time
        self.env = env
        self.pi_loss = []
        self.q_loss = []

        self.device = device

    
    def soft_update(self,target_network, network, tau):
        for target_param, source_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


    def train(self,memory_buffer:DDPGMemory, train_iteration:int, batch_size:int, epochs:int):
        models = [self.pi, self.pi_t, self.q, self.q_t]
        for model in models:
            model.train()
        
        for epoch in range(epochs):
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = memory_buffer.sample_memory(batch_size)

            #batch to device
            sampled_states = torch.stack(sampled_states).to(self.device)
            sampled_actions = torch.stack(sampled_actions).to(self.device)
            sampled_rewards = torch.stack(sampled_rewards).to(self.device)
            sampled_next_states = torch.stack(sampled_next_states).to(self.device)
            sampled_dones = torch.stack(sampled_dones).to(self.device)
            
            
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
            # nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.q.optimizer.step()

            # compute policy (actor) loss
            actions = self.pi.forward(sampled_states)
            state_action_pairs = torch.cat([sampled_states, actions], dim=1)
            critic_values = self.q.forward(state_action_pairs)

            policy_loss = -critic_values.mean()

            # optimization step (policy)
            self.pi.optimizer.zero_grad()
            policy_loss.backward()
            # nn.utils.clip_grad_norm_(self.pi.parameters(), 5.0)
            self.pi.optimizer.step()

            # update target networks
            self.soft_update(self.pi_t, self.pi, tau=0.005)
            self.soft_update(self.q_t, self.q, tau=0.005)

        
        self.pi_loss.append(policy_loss.detach().cpu().numpy().item())
        self.q_loss.append(critic_loss.detach().cpu().numpy().item())
        
        if train_iteration % 100 == 0:
            print(f'timestep {train_iteration}/{self.T}: Policy loss = {self.pi_loss[-1]} || Value loss = {self.q_loss[-1]}')