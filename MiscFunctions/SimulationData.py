import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from numpy.random import seed
from collections import deque
import scipy.linalg

seed(42)

class LQRController:
    def __init__(self, g=10.0, m=1.0, l=1.0, b=0.1, Q=np.diag([15, 1]), R=np.array([[0.5]])):
        """
        LQR Controller for the Pendulum environment.
        Parameters:
        g: gravity
        m: mass of the pendulum
        l: length of the pendulum
        b: damping coefficient
        Q: state cost matrix
        R: control effort cost
        """
        self.g = g
        self.m = m
        self.l = l
        self.b = b
        self.Q = Q
        self.R = R

        self.A = np.array([[0, 1], [self.g / self.l, -self.b / (self.m * self.l**2)]])
        self.B = np.array([[0], [1 / (self.m * self.l**2)]])

        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.dot(np.linalg.inv(self.R), np.dot(self.B.T, P))

    def compute_control(self, theta, theta_dot):
        """
        Compute the control action using the LQR feedback law.
        """
        x = np.array([theta, theta_dot]).flatten()
        u = -np.dot(self.K, x)
        return np.clip(u, -2.0, 2.0)  # Clip to action limits of Pendulum-v1
    

class EnergyShapingController:
    def __init__(self, mass, rod_length, gravity, kp):
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2)
        self.kp = kp

    def get_action(self, state):
        target_energy = self.m * self.g * self.l
        kinetic_energy = 0.5 * self.I * (state[1] ** 2)
        potential_energy = -target_energy * np.cos(state[0])
        total_energy = kinetic_energy + potential_energy

        energy_error = total_energy - target_energy
        # u = self.kp * state[1] * energy_error
        if abs(state[1]) < 0.1: # to adjust for faster convergence
            u = self.kp * 0.5 * np.sign(state[0])
        else:
            u = self.kp * state[1] * energy_error
        action = np.clip(u, a_min=-2, a_max=2)
        return action

def get_state(x: np.array):
    theta = np.arctan2(x[1], x[0])
    theta_dot = x[2]
    return np.array([theta, theta_dot])

if __name__ == "__main__":

    env = gym.make("Pendulum-v1", g=9.81, render_mode='rgb_array') #render_mode = 'human', if want to see video

    pendulum_params = {"mass": env.unwrapped.m, "rod_length": env.unwrapped.l, "gravity": 9.81}
    energy_controller = EnergyShapingController(kp=0.3, **pendulum_params)
    lqr_controller = LQRController(g=9.81, m=env.unwrapped.m, l=env.unwrapped.l, b=0.1)

    n_episodes = 1000
    assert n_episodes >= 10 # Somehow, because of plotting, a smaller version crashes
    switch_threshold = 1.7
    done_threshold = 0.01
    state_buffer = deque(maxlen=9)
    df = pd.DataFrame(columns=["episode", "actions", "state", "next_state"])

    for i in range(n_episodes):
        print('Episode:', i)
        obs, _ = env.reset()
        done = False
        state0 = get_state(obs)
        state = state0.copy()
        state_buffer.clear()

        while not done:
            if abs(state[0]) > switch_threshold:
                action = energy_controller.get_action(state)
            else:
                action = lqr_controller.compute_control(state[0], state[1])

            obs_ ,_ ,_ ,_, _ = env.step(action.reshape(1,-1))
            next_state = get_state(obs_.squeeze())
            state_buffer.append(next_state)
 
            if len(state_buffer) == 9 and all(abs(s[0]) < done_threshold for s in state_buffer):
                done = True

            state_list = state.tolist() # for pd saving.
            next_state_list = next_state.tolist() # for pd saving.

            df2 = pd.DataFrame([[i, action.squeeze(), state_list, next_state_list]],columns=['episode', 'actions', 'state', 'next_state'])
            df = pd.concat([df, df2], ignore_index=True)
            state = next_state.copy() # use .copy() for arrays because of the shared memory issues
        
    env.close()
    df.to_csv(f'Data/CSVs/data_{n_episodes}.csv', index=False)
    # Plot state.
    print('Plotting...')
    fig, ax = plt.subplots(3, 1, sharex=True)

    for i in range(n_episodes):
        episode_data = df[df['episode'] == i]
        angles = [state[0] for state in episode_data["state"]]
        angular_vel = [state[1] for state in episode_data["state"]]
        actions = [action for action in episode_data["actions"]]
        ax[0].plot(angles, label=f"Episode {i}")
        ax[1].plot(angular_vel, label=f"Episode {i}")
        ax[2].plot(actions, label=f"Episode {i}")
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title("Performance of Energy Shaping Controller + LQR on Inverted Pendulum")
        ax[0].set_ylabel("Angle [rad]")
        ax[1].set_ylabel("Angular Velocity [rad/s]")
        ax[2].set_ylabel("Actions [Nm]")

    plt.tight_layout()
    plt.savefig(f'Data/Plots/ModelBasedControlData_{n_episodes}Episodes.svg')
    plt.show()