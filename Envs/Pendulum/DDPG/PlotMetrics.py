import matplotlib.pyplot as plt

from MiscFunctions.Plotting import *

fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))
# ax[2].set_ylim(-2500)
SMOOTHING_WINDOW = 10
REWARD_TYPE = 'sparse' #'dense' or 'sparse' 
gauss_plotter = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Gaussian_{REWARD_TYPE}.csv',
                            show=False, title='Gaussian added Noise', smooth=SMOOTHING_WINDOW)
gauss_plotter.plot_losses(ax=ax)

ou_plotter = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/OrnsteinUhlenbeck_{REWARD_TYPE}.csv',
                         show=False, title='OU added Noise', smooth=SMOOTHING_WINDOW)
ou_plotter.plot_losses(ax=ax)

custom_plotter = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/CustomNoise_{REWARD_TYPE}.csv',
                             show=False, title='Custom added Noise', smooth=SMOOTHING_WINDOW)
custom_plotter.plot_losses(ax=ax)

expert_plotter = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/CustomExpert_{REWARD_TYPE}.csv',
                             show=False, title='EXPERT Policy', smooth=SMOOTHING_WINDOW)
expert_plotter.plot_losses(ax=ax)


plt.tight_layout()
# plt.savefig('Data/Plots/ALLTrainingComparison.svg')
plt.show()


