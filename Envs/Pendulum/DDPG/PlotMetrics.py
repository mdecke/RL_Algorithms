import matplotlib.pyplot as plt

from MiscFunctions.Plotting import *

fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))
ax[2].set_ylim(-2500)
SMOOTHING_WINDOW = 1

# gauss_plotter = DDPGMetrics(file_path='Envs/Pendulum/DDPG/Metrics/Gaussian_sparse_single.csv',
#                             show=False, title='Gaussian added Noise', smooth=SMOOTHING_WINDOW)
# gauss_plotter.plot_losses(ax=ax)

# ou_plotter = DDPGMetrics(file_path='Envs/Pendulum/DDPG/Metrics/OrnsteinUhlenbeck_sparse_single.csv',
#                          show=False, title='OU added Noise', smooth=SMOOTHING_WINDOW)
# ou_plotter.plot_losses(ax=ax)

# custom_plotter = DDPGMetrics(file_path='Envs/Pendulum/DDPG/Metrics/Custom_sparse_single.csv',
#                              show=False, title='Custom added Noise', smooth=SMOOTHING_WINDOW)
# custom_plotter.plot_losses(ax=ax)

# expert_plotter = DDPGMetrics(file_path='Envs/Pendulum/DDPG/Metrics/EXPERT_single.csv',
#                              show=False, title='EXPERT added Noise', smooth=SMOOTHING_WINDOW)
# expert_plotter.plot_losses(ax=ax)

expert_plotter = DDPGMetrics(file_path='Envs/Pendulum/DDPG/Metrics/EXPERT_single.csv',
                         show=False, title='learned EXPERT', smooth=SMOOTHING_WINDOW)
expert_plotter.plot_losses(ax=ax)

other_plotter = DDPGMetrics(file_path='Envs/Pendulum/DDPG/Metrics/EXPERT_sparse_single.csv',
                         show=False, title='learned Other', smooth=SMOOTHING_WINDOW)
other_plotter.plot_losses(ax=ax)

custom_plotter = DDPGMetrics(file_path='Envs/Pendulum/DDPG/Metrics/Custom_single.csv',
                                show=False, title='learned Custom', smooth=SMOOTHING_WINDOW)
custom_plotter.plot_losses(ax=ax)


plt.tight_layout()
# plt.savefig('Data/Plots/ALLTrainingComparison.svg')
plt.show()


