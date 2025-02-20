import numpy as np
import pandas as pd


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
        
