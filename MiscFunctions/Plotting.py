import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF


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
          
    def plot_losses(self):
        self.split_losses()
        trainings_steps = np.linspace(0,len(self.policy_losses['mean']), len(self.policy_losses['mean']))
        n_episodes = np.linspace(0,len(self.episodic_returns['mean']), len(self.episodic_returns['mean']))
        if self.show:
            fig,ax = plt.subplots(3, 1, sharex=False, figsize=(10, 8))
            

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
        
        for i in range(3):
            ax[i].set_facecolor('lightgrey')
            ax[i].grid(True)
            ax[i].legend()

        if self.show:
            plt.tight_layout()
            plt.show()
        
    
class ClusterPlotting:
    def __init__(self, data=None, file_path=None):
        
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        elif type(data) is not None:
            self.df = data

    def plot_clusters(self, data_type: str, save=False):
        states = self.df['state'].to_list()
        states_np = np.array([np.fromstring(state.strip('[]'), sep=',') for state in states])
        if data_type == "polar":
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(states_np[:,0], states_np[:,1], c=self.df['cluster_label'], cmap='viridis')
            plt.title('Clustered Data')
            plt.xlabel('Feature 1 - angle')
            plt.ylabel('Feature 2 - angular velocity')

        
        elif data_type == "cartesian":
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection ='3d')
            scatter = ax.scatter(states_np[:,0], states_np[:,1], states_np[:,2], c=self.df['cluster_label'], cmap='viridis')
            ax.set_title('Clustered Data')
            ax.set_xlabel('Feature 1 - x')
            ax.set_ylabel('Feature 2 - y')
            ax.set_zlabel('Feature 3 - omega dot')
            
        
        elif data_type == "state_action":
            prev_actions = self.df['previous_action'].to_numpy().reshape(-1, 1)
            state_prev_actions = np.concatenate((states_np, prev_actions), axis=1)
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection ='3d')
            scatter = ax.scatter(state_prev_actions[:,0], state_prev_actions[:,1], state_prev_actions[:,2], c=self.df['cluster_label'], cmap='viridis')
            ax.set_title('Clustered Data')
            ax.set_xlabel('Feature 1 - theta')
            ax.set_ylabel('Feature 2 - theta dot')
            ax.set_zlabel('Feature 3 - previous action')
        
        plt.colorbar(scatter, label='Cluster Label')
        plt.grid(True)
        if save:
            plt.savefig(f'Data/Plots/Clusters.svg')
        plt.tight_layout()
        plt.show()
    
    def plot_fitted_cluster_gmm(self, fitted_models:dict, save=False):
        nb_clusters = self.df['cluster_label'].max() + 1
        subplots_per_figure = 10
        n_figures = (nb_clusters + subplots_per_figure - 1) // subplots_per_figure

        for fig_num in range(n_figures):
            start_idx = fig_num * subplots_per_figure
            end_idx = min(start_idx + subplots_per_figure, nb_clusters)
            n_plots = end_idx - start_idx
            
            n_cols = min(4, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_plots == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for i in range(start_idx, end_idx):
                ax_idx = i - start_idx
                actions = self.df[self.df['cluster_label'] == i]['actions'].to_numpy()
                em_model = fitted_models[f'cluster {i}']
                
                # Plot histogram
                axes[ax_idx].hist(actions, bins=30, density=True, alpha=0.6, 
                                color='gray', label="Actions")
                
                # Generate x values for plotting
                x = np.linspace(min(actions), max(actions), 1000)
                
                # Plot individual components
                for j in range(em_model.n_dist):
                    component = em_model.weights[j] * em_model.gaussian(x, 
                                                                    em_model.means[j], 
                                                                    em_model.stds[j])
                    axes[ax_idx].plot(x, component, '--', alpha=0.7,
                                    label=f'G{j+1} (μ={em_model.means[j]:.2f}, σ={em_model.stds[j]:.2f})')
                
                # Plot combined PDF
                combined_pdf = sum(em_model.weights[j] * em_model.gaussian(x, 
                                                                        em_model.means[j], 
                                                                        em_model.stds[j])
                                for j in range(em_model.n_dist))
                axes[ax_idx].plot(x, combined_pdf, 'r-', label="Combined GMM")
                
                # Set labels and title
                axes[ax_idx].set_xlabel('Action Values')
                axes[ax_idx].set_ylabel('Density')
                axes[ax_idx].set_title(f'Cluster {i} GMM Fit')
                axes[ax_idx].legend(loc='upper right', prop={'size': 6})
                axes[ax_idx].grid(True)

            # Hide empty subplots if any
            for j in range(ax_idx + 1, len(axes)):
                axes[j].set_visible(False)
            if save:
                plt.savefig(f'Data/Plots/GMMClusters_{fig_num}.svg')
            plt.tight_layout()
            plt.show()


class KDEPlotting:
    def __init__(self, actions, nb_bins, kde_fitter):
        self.actions = actions
        self.nb_bins = nb_bins
        self.action_space = np.linspace(min(self.actions), max(self.actions), len(self.actions))
        self.kde = kde_fitter.kde
        self.kde.fit(self.actions[:, None])
        self.pdf = np.exp(self.kde.score_samples(self.action_space[:, None]))
        self.cdf = np.cumsum(self.pdf) * (self.action_space[1] - self.action_space[0])

    def plot_action_dist(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.actions, bins=self.nb_bins, density=True, alpha=0.6, color='red', label='Histogram')
        plt.grid()
        plt.xlabel('Torque [N/m]')
        plt.ylabel('Density')
        plt.title('Action distribution over action space')
        # plt.savefig(f'InvertedPendulum/Data/Plots/P(a)_Histogram_{datetime.now().date()}')
        plt.show()
    
    def plot_hist_vs_fitted_pdf(self):
        params = self.kde.get_params()
        kernel = params['kernel']
        bandwidth = params['bandwidth']

        plt.figure(figsize=(10, 6))
        plt.fill_between(self.action_space, self.pdf, alpha=0.3, color='green',label=f"KDE: {kernel} (bw={bandwidth:.2f})")
        plt.hist(self.actions, bins=self.nb_bins, density=True, alpha=0.6, color='red', label='Histogram')
        plt.grid()
        plt.xlabel('Action Values')
        plt.ylabel('Density')
        plt.title(f'Best KDE Fit: Kernel={kernel}, Bandwidth={bandwidth:.2f}')
        plt.legend()
        # plt.savefig(f"InvertedPendulum/Data/Plots/P(a)_BestDensityFit_{datetime.now().date()}")
        plt.show()
    
    def plot_cdf_vs_ECDF(self):
        ecdf = ECDF(self.actions)
        y_vals = ecdf(self.action_space)

        plt.plot(self.action_space, y_vals,'r-',label='ECDF')
        plt.plot(self.action_space, self.cdf, 'b-', label='KDE cdf')
        plt.xlabel('torque')
        plt.ylabel('cumulative probability')
        plt.grid()
        plt.legend()
        plt.title('Comparison of KDE cdf and Empirical CDF')
        # plt.savefig(f'InvertedPendulum/Data/Plots/P(a)_CDFComp_{datetime.now().date()}')
        plt.show()