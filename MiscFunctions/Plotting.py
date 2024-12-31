import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DDPGMetrics:
    def __init__(self, data=None, file_path=None, show=True, title=None):
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        elif type(data) is not None:
            self.df = data

        self.show = show
        self.title = title
    def split_losses(self):
        self.policy_losses = self.df[self.df['label'] == 'policy'].iloc[:,1:-1]
        self.value_losses = self.df[self.df['label'] == 'value'].iloc[:,1:-1]
        self.returns = self.df[self.df['label'] == 'returns'].iloc[:,1:-1]

        self.policy_losses['min'] = self.policy_losses.min(axis=1)
        self.policy_losses['max'] = self.policy_losses.max(axis=1)
        self.policy_losses['mean'] = self.policy_losses.mean(axis=1)

        self.value_losses['min'] = self.value_losses.min(axis=1)
        self.value_losses['max'] = self.value_losses.max(axis=1)
        self.value_losses['mean'] = self.value_losses.mean(axis=1)

        self.returns['min'] = self.returns.min(axis=1)
        self.returns['max'] = self.returns.max(axis=1)
        self.returns['mean'] = self.returns.mean(axis=1)
    
    def plot_losses(self):
        self.split_losses()
        trainings_steps = np.linspace(0,len(self.policy_losses['mean']), len(self.policy_losses['mean']))
        n_episodes = np.linspace(0,len(self.returns['mean']), len(self.returns['mean']))
        if self.show:
            plt.figure(figsize=(10, 8))
        if self.title == "P(a[k]) Noise":
            color = 'blue'
        elif self.title == "P(a[k]|s[k],a[k-1]) Noise":
            color = 'orange'
        elif self.title == "P(a[k]|s[k]) Noise":
            color = 'green'
        elif self.title == "n ~ OH() Noise":
            color = 'red'
        # Q-function loss subplot
        plt.subplot(3, 1, 1)
        plt.plot(trainings_steps, self.value_losses['mean'], label=f'{self.title} Mean Q-value Loss', color=color)
        plt.fill_between(x=range(len(trainings_steps)),
                        y1=self.value_losses['min'],
                        y2=self.value_losses['max'],
                        color=color, alpha=0.2)
        plt.ylabel('Value Loss')
        plt.xlabel('Training Steps')
        plt.title('Q-function Loss')
        plt.grid(True)
        plt.legend()

        # Policy loss subplot
        plt.subplot(3, 1, 2)
        plt.plot(trainings_steps, self.policy_losses['mean'], label=f'{self.title} Mean Policy Loss', color=color)
        plt.fill_between(x=range(len(trainings_steps)),
                        y1=self.policy_losses['min'],
                        y2=self.policy_losses['max'],
                        color=color, alpha=0.2)
        plt.ylabel('Policy Loss')
        plt.xlabel('Training Steps')
        plt.title('Policy Loss')
        plt.grid(True)
        plt.legend()

        # Episodic return subplot (no smoothing here, just raw)
        plt.subplot(3, 1, 3)
        plt.plot(n_episodes, self.returns['mean'], label=f'{self.title} Mean Episodic Return', color=color)
        plt.fill_between(x = range(len(n_episodes)),
                        y1=self.returns['min'],
                        y2=self.returns['max'],
                        color=color, alpha=0.2)
        plt.ylabel('Return')
        plt.xlabel('Episodes')
        plt.title('Episodic Return')
        plt.grid(True)
        plt.legend()
        

        if self.show:
            plt.tight_layout()
            plt.show()
        
    
class ClusterPlotting:
    def __init__(self, data=None, file_path=None):
        
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        elif type(data) is not None:
            self.df = data

    def plot_clusters(self, data_type: str):
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
        plt.show()
    
    def plot_fitted_cluster_gmm(self, fitted_models:dict):
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

            plt.tight_layout()
            plt.show()
