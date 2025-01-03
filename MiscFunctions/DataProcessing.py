import os
import sys
import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from statsmodels.distributions.empirical_distribution import ECDF

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiscFunctions.EMAlgo import EM_Algorithm, EMOptunaOptimizer


class Cluster:
    def __init__(self, file_path: str, nb_clusters: int):
        self.dataset = pd.read_csv(file_path)
        self.nb_clusters = nb_clusters
        self.kmeans = None
        self.actions_per_state = {}
        self.fitted_models = {}

    def _add_previous_action(self, return_df: bool = False):
        self.dataset['previous_action'] = self.dataset['actions'].shift(1, fill_value=0)
        self.dataset.groupby('episode').first().reset_index()['previous_action'] = 0
        if return_df:
            extended_data = self.dataset
            return extended_data

    def _extract_states(self):
        states = self.dataset['state'].to_list()
        states_np = np.array([np.fromstring(state.strip('[]'), sep=',') for state in states])
        return states_np

    def cluster_data(self, what_to_cluster: str = 'states'):
        if what_to_cluster == 'states':
            data_to_cluster = self._extract_states()
        elif what_to_cluster == 'state_action':
            states_np = self._extract_states()
            self._add_previous_action()
            prev_actions = self.dataset['previous_action'].to_numpy().reshape(-1, 1)
            data_to_cluster = np.concatenate((states_np, prev_actions), axis=1)
        else:
            raise ValueError("Invalid value for 'what_to_cluster'. Must be 'states' or 'state_action'")
            
        self.kmeans = KMeans(n_clusters=self.nb_clusters)
        self.kmeans.fit(data_to_cluster)
        self.dataset['cluster_label'] = self.kmeans.labels_
        
        for i in range(self.nb_clusters):
            self.actions_per_state[f'label {i}'] = self.dataset[self.dataset['cluster_label'] == i]['actions'].to_numpy()
        return self.dataset


class FitGMM:
    def __init__(self, data=None, file_path=None, clustered_data=True, nb_trials=10, components_range=(2, 5), lambda_reg_range=(1e-5, 1e-3)):
        if data is not None:
            self.dataset = data
        elif file_path is not None:
            self.dataset = pd.read_csv(file_path)
        else:
            raise ValueError("Either 'data' or 'file_path' must be provided.")
        
        self.clustered_data = clustered_data
        
        if clustered_data:
            self.nb_clusters = self.dataset['cluster_label'].max() + 1
       
        self.fitted_model = None
        self.nb_trails = nb_trials
        self.components_range = components_range
        self.lambda_reg_range = lambda_reg_range
        self.fitted_models = {}
        

    def fit_gmm(self):
        """Fits a GMM to each cluster's action distribution using EM algorithm"""
        if self.clustered_data:
            for i in range(self.nb_clusters):
                print(f" ... Fitting GMM for cluster {i}/{self.nb_clusters} ... ")
                actions = self.dataset[self.dataset['cluster_label'] == i]['actions'].to_numpy()
                optimizer = EMOptunaOptimizer(
                    data=actions,
                    n_trials=self.nb_trails,
                    n_components_range=self.components_range,
                    lambda_reg_range=self.lambda_reg_range
                )
                optimizer.optimize()
                optimizer.print_best_results()
            
                # Fit the GMM with optimized parameters
                best_params = optimizer.get_best_params()
                
                em_model = EM_Algorithm(
                    data=actions,
                    n_distributions=best_params['n_components'],
                    regularization_constant=best_params['lambda_reg'])
                
                em_model.fit(regularized=True)
                self.fitted_models[f'cluster {i}'] = em_model
            return self.fitted_models
        else:
            actions = self.dataset['actions'].to_numpy()
            optimizer = EMOptunaOptimizer(
                data=actions,
                n_trials=self.nb_trails,
                n_components_range=self.components_range,
                lambda_reg_range=self.lambda_reg_range
            )
            optimizer.optimize()
            optimizer.print_best_results()
        
            # Fit the GMM with optimized parameters
            best_params = optimizer.get_best_params()
            
            em_model = EM_Algorithm(
                data=actions,
                n_distributions=best_params['n_components'],
                regularization_constant=best_params['lambda_reg'])
            
            em_model.fit(regularized=True)
            self.fitted_models['all actions'] = em_model
            return self.fitted_models

    
    def sample_cluster(self, cluster_label: int, n_samples: int = 1) -> np.ndarray:
        if not self.fitted_models:
            raise ValueError("No fitted models available. Run fit_gmm() first.")
            
        if f'cluster {cluster_label}' not in self.fitted_models:
            raise ValueError(f"No fitted model for cluster {cluster_label}")
            
        model = self.fitted_models[f'cluster {cluster_label}']
        return model.sample(n_samples)


class FitKDE:
    def __init__(self, data=None, file_path=None, nb_bins=10):
        if data is not None:
            self.data = data
        elif file_path is not None:
            self.data = pd.read_csv(file_path)
        else:
            raise ValueError("Either 'data' or 'file_path' must be provided.")
        self.nb_bins = nb_bins
        self.actions = self.data['actions'].to_numpy()
        self.action_space = np.linspace(min(self.actions), max(self.actions), len(self.actions))

    def compute_sse(self, kde, bin_edges):
        hist_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        kde_density = np.exp(kde.score_samples(hist_bin_centers[:, None]))
        sse = np.sum((kde_density - self.action_hist) ** 2)
        return sse
    
    def fit_action_dist(self, kernels:list):
        bin_edges = np.histogram_bin_edges(self.actions, bins=self.nb_bins)
        self.action_hist,_= np.histogram(self.actions, bins=bin_edges, density=True)
        kernels = kernels
        best_kernel = None
        best_sse = float('inf')

        for kernel in kernels:
            print(f" ... fitting KDE with {kernel} kernel ...")
            for bandwidth in np.arange(0.01, 0.5, 0.02):
                kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
                kde.fit(self.actions[:, None])
                sse = self.compute_sse(kde, bin_edges)
                if sse < best_sse:
                    best_sse = sse
                    best_bdw = bandwidth
                    if best_kernel != kernel:
                        best_kernel = kernel
        
        self.kde = KernelDensity(kernel=best_kernel,bandwidth=best_bdw)
        self.kde.fit(self.actions[:, None])
        self.pdf = np.exp(self.kde.score_samples(self.action_space[:, None]))
        self.cdf = np.cumsum(self.pdf) * (self.action_space[1] - self.action_space[0])

    def save_best_model(self, filename="Data/Models/P(a)_KDE.pkl"):
        if hasattr(self, 'kde'):
            joblib.dump(self.kde, filename)
            print(f"Best KDE model saved to {filename}.")
        else:
            print("No fitted model found. Please fit the model before saving.")
    
    def load_best_model(self, filename="Data/Models/P(a)_KDE.pkl"):
        print('... loading model ...')
        self.kde = joblib.load(filename)
        self.kde.fit(self.actions[:, None])
        self.pdf = np.exp(self.kde.score_samples(self.action_space[:, None]))
        self.cdf = np.cumsum(self.pdf) * (self.action_space[1] - self.action_space[0])


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
        # Outputs the mean
        self.log_sigma_head = nn.Linear(32, action_dim) # Outputs log(sigma)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma
    
    
def train_nn(model:ContinuousActionNN, epochs:int, nb_batch:int,
             nb_trajectories:int, data:pd.DataFrame, data_type:str):

    episode_idx = data['episode'].to_numpy()
    episode_idx = torch.tensor(episode_idx, dtype=torch.float32)
    nb_episodes = episode_idx.unique().cpu()

    training_loss_list = []
    
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.
        last_loss = 0.
        for batch_idx in range(nb_batch):
            # Zero your gradients for every batch!
            model.optimizer.zero_grad()
            batch_loss = 0.0
            for _ in range(nb_trajectories):
                trajectory_idx = np.random.choice(nb_episodes, size=1)
                states = data[data['episode'].isin(trajectory_idx)]['state'].to_numpy()
                states_list = []
                for i in range(len(states)):
                    states_list.append(states[i])
                actions = data[data['episode'].isin(trajectory_idx)]['actions'].to_numpy()
                previous_actions = data[data['episode'].isin(trajectory_idx)]['previous_action'].to_numpy()
                states = np.array(states_list)
                batch_states = torch.tensor(states, dtype=torch.float32)
                batch_actions = torch.tensor(actions, dtype=torch.float32).reshape(-1, 1)
                if data_type == 'states':
                    inputs = batch_states
                elif data_type == 'state_action':
                    batch_previous_actions = torch.tensor(previous_actions, dtype=torch.float32).reshape(-1, 1)
                    inputs = torch.cat((batch_states, batch_previous_actions), dim=1)
                else:
                    raise ValueError("Invalid data type. Must be 'states' or 'states_action'")
                
                mu, log_sigma = model(inputs) # model(all_states)
                sigma = torch.exp(log_sigma)

                # Compute NLL loss
                diff = batch_actions - mu
                nll_loss = (log_sigma + 0.5 * (diff**2) / (sigma**2)).mean() + 0.5 * np.log(2 * np.pi)
                batch_loss += nll_loss
            
            batch_loss /= nb_trajectories
            # Compute the loss and its gradients
            batch_loss.backward()
            running_loss += batch_loss.item()

            # Adjust learning weights
            model.optimizer.step()

            if batch_idx  % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
                running_loss = 0.

        print(f"Epoch {epoch+1}/{epochs} avg loss: {last_loss}")
        training_loss_list.append(last_loss)
