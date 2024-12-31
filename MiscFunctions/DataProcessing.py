import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiscFunctions.EMAlgo import EM_Algorithm, EMOptunaOptimizer


class Cluster:
    def __init__(self, file_path: str, nb_clusters: int):
        self.dataset = pd.read_csv(file_path)
        self.nb_clusters = nb_clusters
        self.kmeans = None
        self.actions_per_state = {}
        self.fitted_models = {}

    def _add_previous_action(self):
        self.dataset['previous_action'] = self.dataset['actions'].shift(1, fill_value=0)
        self.dataset.groupby('episode').first().reset_index()['previous_action'] = 0

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
    def __init__(self, data=None, file_path=None, nb_trials=10, components_range=(2, 5), lambda_reg_range=(1e-5, 1e-3)):
        if data is not None:
            self.dataset = data
        elif file_path is not None:
            self.dataset = pd.read_csv(file_path)
        else:
            raise ValueError("Either 'data' or 'file_path' must be provided.")
        self.fitted_model = None
        self.nb_clusters = self.dataset['cluster_label'].max() + 1
        self.nb_trails = nb_trials
        self.components_range = components_range
        self.lambda_reg_range = lambda_reg_range
        self.fitted_models = {} 
    
    def fit_gmm(self):
        """Fits a GMM to each cluster's action distribution using EM algorithm"""
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
    
    def sample_cluster(self, cluster_label: int, n_samples: int = 1) -> np.ndarray:
        if not self.fitted_models:
            raise ValueError("No fitted models available. Run fit_gmm() first.")
            
        if f'cluster {cluster_label}' not in self.fitted_models:
            raise ValueError(f"No fitted model for cluster {cluster_label}")
            
        model = self.fitted_models[f'cluster {cluster_label}']
        return model.sample(n_samples)


