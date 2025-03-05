import matplotlib.pyplot as plt

from MiscFunctions.Plotting import *
import datetime

SMOOTHING_WINDOW = 5
REWARD_TYPE = 'sparse' #'dense' or 'sparse' 
plot_p_q = False

if plot_p_q:
    fig, ax = plt.subplots(3, 1, sharex=False, figsize=(8, 8))
    ax[2].set_ylim(-2500)
else:
    fig, ax = plt.subplots(1, 1, sharex=False, figsize=(8, 8))
    ax.set_ylim(-2500)

# ___________________________________________________________ Ploting dense reward - short training __________________________________________________________________

gauss_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/Gaussian_dense.csv',
                            show=False, title='Dense Reward - Gaussian', smooth=SMOOTHING_WINDOW)
gauss_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

ou_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/OrnsteinUhlenbeck_dense.csv',
                            show=False, title='Dense Reward - OU', smooth=SMOOTHING_WINDOW)
ou_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

custom_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomNoise_dense.csv',
                            show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
custom_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

expert_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomExpert_dense.csv',
                            show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
expert_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

plt.tight_layout()
# plt.savefig('Data/Plots/ALLTrainingComparison.svg')
plt.show()

# # ___________________________________________________________ Ploting dense reward - long training __________________________________________________________________

# gauss_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/Gaussian_dense.csv',
#                             show=False, title='Dense Reward - Gaussian', smooth=SMOOTHING_WINDOW)
# gauss_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# ou_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/OrnsteinUhlenbeck_dense.csv',
#                             show=False, title='Dense Reward - OU', smooth=SMOOTHING_WINDOW)
# ou_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# custom_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomNoise_dense.csv',
#                             show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
# custom_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# expert_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomExpert_dense.csv',
#                             show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
# expert_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# plt.tight_layout()
# # plt.savefig('Data/Plots/ALLTrainingComparison.svg')
# plt.show()

# # ___________________________________________________________ Ploting sparse reward - short training __________________________________________________________________

# gauss_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/Gaussian_dense.csv',
#                             show=False, title='Dense Reward - Gaussian', smooth=SMOOTHING_WINDOW)
# gauss_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# ou_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/OrnsteinUhlenbeck_dense.csv',
#                             show=False, title='Dense Reward - OU', smooth=SMOOTHING_WINDOW)
# ou_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# custom_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomNoise_dense.csv',
#                             show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
# custom_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# expert_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomExpert_dense.csv',
#                             show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
# expert_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# plt.tight_layout()
# # plt.savefig('Data/Plots/ALLTrainingComparison.svg')
# plt.show()

# # ___________________________________________________________ Ploting sparse reward - long training __________________________________________________________________

# gauss_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/Gaussian_dense.csv',
#                             show=False, title='Dense Reward - Gaussian', smooth=SMOOTHING_WINDOW)
# gauss_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# ou_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/OrnsteinUhlenbeck_dense.csv',
#                             show=False, title='Dense Reward - OU', smooth=SMOOTHING_WINDOW)
# ou_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# custom_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomNoise_dense.csv',
#                             show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
# custom_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# expert_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomExpert_dense.csv',
#                             show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
# expert_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)



# plt.tight_layout()
# # plt.savefig('Data/Plots/ALLTrainingComparison.svg')
# plt.show()


