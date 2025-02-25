import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DDPGMetrics:
    def __init__(self, data:pd.DataFrame=None, file_path:str=None, show:bool=True, title:str=None, smooth:int=None):
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        elif type(data) is not None:
            self.df = data

        self.show = show
        self.title = title
        self.smooth = smooth 

    def split_losses(self):
        cycles = self.df['cycle'].unique()
        self.policy_losses = pd.DataFrame()
        self.q_losses = pd.DataFrame()
        self.episodic_returns = pd.DataFrame()
        for cycle in cycles:
            cycle_policy_loss = self.df[self.df['cycle'] == cycle]['policy_loss'].reset_index(drop=True)
            cycle_q_loss = self.df[self.df['cycle'] == cycle]['q_loss'].reset_index(drop=True)
            cycle_returns = self.df[self.df['cycle'] == cycle]['return'].dropna().reset_index(drop=True)

            self.policy_losses[f'cycle {cycle}'] = cycle_policy_loss
            self.q_losses[f'cycle {cycle}'] = cycle_q_loss
            self.episodic_returns[f'cycle {cycle}'] = cycle_returns

        self.policy_losses['min'] = self.policy_losses.min(axis=1)
        self.policy_losses['max'] = self.policy_losses.max(axis=1)
        self.policy_losses['mean'] = self.policy_losses.mean(axis=1)

        self.q_losses['min'] = self.q_losses.min(axis=1)
        self.q_losses['max'] = self.q_losses.max(axis=1)
        self.q_losses['mean'] = self.q_losses.mean(axis=1)

        self.episodic_returns['min'] = self.episodic_returns.min(axis=1)
        self.episodic_returns['max'] = self.episodic_returns.max(axis=1)
        self.episodic_returns['mean'] = self.episodic_returns.mean(axis=1)

    def smooth_data(self,data, window=50):
            return np.convolve(data, np.ones(window)/window, mode='same')
          
    def plot_losses(self, ax=None):
        self.split_losses()
        trainings_steps = np.linspace(0,len(self.policy_losses['mean']), len(self.policy_losses['mean']))
        n_episodes = np.linspace(0,len(self.episodic_returns['mean']), len(self.episodic_returns['mean']))
        
        if ax is None:
            fig, ax = plt.subplots(3, 1, sharex=False, figsize=(10, 8))
            
        if self.smooth:
            window = self.smooth  # You can adjust this value
            self.policy_losses['mean'] = self.smooth_data(self.policy_losses['mean'].to_numpy(), window)
            self.policy_losses['min'] = self.smooth_data(self.policy_losses['min'].to_numpy(), window)
            self.policy_losses['max'] = self.smooth_data(self.policy_losses['max'].to_numpy(), window)
            self.q_losses['mean'] = self.smooth_data(self.q_losses['mean'].to_numpy(), window)
            self.q_losses['min'] = self.smooth_data(self.q_losses['min'].to_numpy(), window)
            self.q_losses['max'] = self.smooth_data(self.q_losses['max'].to_numpy(), window)
            self.episodic_returns['mean'] = self.smooth_data(self.episodic_returns['mean'].to_numpy(), window)
            self.episodic_returns['min'] = self.smooth_data(self.episodic_returns['min'].to_numpy(), window)
            self.episodic_returns['max'] = self.smooth_data(self.episodic_returns['max'].to_numpy(), window)
            
            # Adjust training steps and episodes arrays to match the new length
            trainings_steps = np.linspace(0, len(self.policy_losses['mean']), len(self.policy_losses['mean']))
            n_episodes = np.linspace(0, len(self.episodic_returns['mean']), len(self.episodic_returns['mean']))

        # Q-function loss subplot
        ax[0].plot(trainings_steps, self.q_losses['mean'], label=f'{self.title} Mean Q-value Loss')
        ax[0].fill_between(x=range(len(trainings_steps)),
                        y1=self.q_losses['min'],
                        y2=self.q_losses['max'], alpha=0.2)
        ax[0].set_ylabel('Value Loss')
        ax[0].set_xlabel('Training Steps')
        ax[0].set_title(f'Q-function Loss (smoothed = {self.smooth})')
        

        # Policy loss subplot
        ax[1].plot(trainings_steps, self.policy_losses['mean'], label=f'{self.title} Mean Policy Loss')
        ax[1].fill_between(x=range(len(trainings_steps)),
                        y1=self.policy_losses['min'],
                        y2=self.policy_losses['max'],
                        alpha=0.2)
        ax[1].set_ylabel('Policy Loss')
        ax[1].set_xlabel('Training Steps')
        ax[1].set_title(f'Policy Loss (smoothed = {self.smooth})')

        # Episodic return subplot (no smoothing here, just raw)
        ax[2].plot(n_episodes, self.episodic_returns['mean'], label=f'{self.title} Mean Episodic Return')
        ax[2].fill_between(x = range(len(n_episodes)),
                        y1=self.episodic_returns['min'],
                        y2=self.episodic_returns['max'],
                        alpha=0.2)
        ax[2].set_ylabel('Return')
        ax[2].set_xlabel('Training Episodes')
        ax[2].set_title(f'Episodic Return (smoothed = {self.smooth})')

        ax[2].axhline(y=0, color='k', linewidth=2)
        ax[2].set_ylim(top=50)

        for i in range(3):
            ax[i].set_facecolor('lightgrey')
            ax[i].grid(True)
            ax[i].legend()

        if self.show and ax is None:
            plt.tight_layout()
            plt.show()
