import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def get_state(x: np.array):
    theta = np.arctan2(x[1], x[0])
    theta_dot = x[2]
    return np.array([theta, theta_dot])

def augment_data(data:pd.DataFrame):
    prev_act =[]
    prev_state = []
    current_episode = None
    for idx, row in data.iterrows():
        if current_episode != row['episode']:
            # New episode starts, use 0 and [0,0]
            prev_act.append(np.array(0))
            prev_state.append("[0, 0]")
            current_episode = row['episode']
        else:
            prev_act.append(data.loc[idx-1, 'actions'])
            prev_state.append(data.loc[idx-1, 'states'])
    data['previous_actions'] = prev_act
    data['previous_states'] = prev_state
    
    return data

def get_nparray(x):
    x_list = []
    for i in range(len(x)):
        x_list.append(x[i])
    x_np = np.array(x_list)
    return x_np

def make_input(obs:np.ndarray, prev_action:np.ndarray=None, prev_obs:np.ndarray=None):
        theta = np.arctan2(obs[0,1], obs[0,0])
        theta_dot = obs[0,2]
        new_state = np.array([theta, theta_dot]).reshape(1, -1)
        if prev_action == None and prev_obs == None:
            input_model = new_state
        elif prev_action.all() != None and prev_obs == None:
            input_model = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
        elif prev_action.all() != None and prev_obs.all() != None:
            theta_prev = np.arctan2(prev_obs[0,1], prev_obs[0,0])
            theta_dot_prev = prev_obs[0,2]
            new_prev_state = np.array([theta_prev, theta_dot_prev]).reshape(1, -1)
            intermed = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
            input_model = np.concatenate((intermed, new_prev_state), axis=1)
        else:
            raise ValueError("Invalid input type. Must be 'state' or 'state_action' or 'prev_state_action'")
        return input_model
        

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


