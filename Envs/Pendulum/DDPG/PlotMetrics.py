import matplotlib.pyplot as plt

from MiscFunctions.Plotting import *
import datetime

SMOOTHING_WINDOW = 5
# REWARD_TYPE = 'sparse' #'dense' or 'sparse' 
plot_p_q = False

if plot_p_q:
    fig, ax = plt.subplots(3, 1, sharex=False, figsize=(10, 6))
    ax[2].set_ylim(-3000)
else:
    fig, ax = plt.subplots(1, 1, sharex=False, figsize=(10, 6))
    ax.set_ylim(-3000)

# ___________________________________________________________ Ploting dense reward - short training __________________________________________________________________

# # gauss_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/Gaussian_dense.csv',
# #                             show=False, title='Dense Reward - Gaussian', smooth=SMOOTHING_WINDOW)
# # gauss_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # ou_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/OrnsteinUhlenbeck_dense.csv',
# #                             show=False, title='Dense Reward - OU', smooth=SMOOTHING_WINDOW)
# # ou_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # custom_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomNoise_dense.csv',
# #                             show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
# # custom_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # expert_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomExpert_dense.csv',
# #                             show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
# # expert_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # addition_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/CustomNoise_dense.csv',
# #                                 show=False, title='Dense Reward - Addition Expert', smooth=SMOOTHING_WINDOW)
# # addition_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # plt.tight_layout()
# # plt.savefig('Data/Plots/ALLTrainingComparison.svg')
# # plt.show()

# # # ___________________________________________________________ Ploting dense reward - long training __________________________________________________________________

# gauss_plotter_dense_long = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/Gaussian_dense.csv',
#                             show=False, title='Dense Reward - Gaussian', smooth=SMOOTHING_WINDOW)
# gauss_plotter_dense_long.plot_losses(ax=ax, plot_p_q=plot_p_q)

# ou_plotter_dense_long = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/OrnsteinUhlenbeck_dense.csv',
#                             show=False, title='Dense Reward - OU', smooth=SMOOTHING_WINDOW)
# ou_plotter_dense_long.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # custom_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomNoise_dense.csv',
# #                             show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
# # custom_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # expert_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomExpert_dense.csv',
# #                             show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
# # expert_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # plt.tight_layout()
# # # plt.savefig('Data/Plots/ALLTrainingComparison.svg')
# # plt.show()

# # # ___________________________________________________________ Ploting sparse reward - short training __________________________________________________________________

# # gauss_plotter_sparse_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/Gaussian_sparse.csv',
# #                             show=False, title='Sparse Reward - Gaussian', smooth=SMOOTHING_WINDOW)
# # gauss_plotter_sparse_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # ou_plotter_sparse_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/OrnsteinUhlenbeck_sparse.csv',
# #                             show=False, title='Sparse Reward - OU', smooth=SMOOTHING_WINDOW)
# # ou_plotter_sparse_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # custom_plotter_sparse_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomNoise_sparse.csv',
# #                             show=False, title='Sparse Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
# # custom_plotter_sparse_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # expert_plotter_sparse_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/NewCustomExpert_sparse.csv',
# #                             show=False, title='Sparse Reward - Expert', smooth=SMOOTHING_WINDOW)
# # expert_plotter_sparse_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # addition_plotter_sparse_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Short_{datetime.date.today()}/CustomNoise_sparse.csv',
# #                                 show=False, title='Sparse Reward - Addition Expert', smooth=SMOOTHING_WINDOW)
# # addition_plotter_sparse_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # plt.tight_layout()
# # # plt.savefig('Data/Plots/ALLTrainingComparison.svg')
# # plt.show()

# # # ___________________________________________________________ Ploting sparse reward - long training __________________________________________________________________

# gauss_plotter_sparse_long = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/Gaussian_sparse.csv',
#                             show=False, title='Sparse Reward - Gaussian', smooth=SMOOTHING_WINDOW)
# gauss_plotter_sparse_long.plot_losses(ax=ax, plot_p_q=plot_p_q)

# ou_plotter_sparse_long = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/OrnsteinUhlenbeck_sparse.csv',
#                             show=False, title='Sparse Reward - OU', smooth=SMOOTHING_WINDOW)
# ou_plotter_sparse_long.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # custom_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomNoise_dense.csv',
# #                             show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
# # custom_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)

# # expert_plotter_dense_short = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/Long_{datetime.date.today()}/NewCustomExpert_dense.csv',
# #                             show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
# # expert_plotter_dense_short.plot_losses(ax=ax, plot_p_q=plot_p_q)




test_plotter1 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5gm_{datetime.date.today()}/Gaussian_sparse.csv',
                            show=False, title='Sparse Reward - Gaussian', smooth=SMOOTHING_WINDOW)
test_plotter1.plot_losses(ax=ax, plot_p_q=plot_p_q)

test_plotter2 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5gm_{datetime.date.today()}/OrnsteinUhlenbeck_sparse.csv',
                            show=False, title='Sparse Reward - OU', smooth=SMOOTHING_WINDOW)
test_plotter2.plot_losses(ax=ax, plot_p_q=plot_p_q)

test_plotter3 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5gm_{datetime.date.today()}/Gaussian_dense.csv',
                            show=False, title='Dense Reward - Gaussian', smooth=SMOOTHING_WINDOW)
test_plotter3.plot_losses(ax=ax, plot_p_q=plot_p_q)

test_plotter4 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5gm_{datetime.date.today()}/OrnsteinUhlenbeck_dense.csv',
                            show=False, title='Dense Reward - OU', smooth=SMOOTHING_WINDOW)
test_plotter4.plot_losses(ax=ax, plot_p_q=plot_p_q)

test_plotter5 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5m_{datetime.date.today()}/NewCustomNoise_sparse.csv',
                            show=False, title='Sparse Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
test_plotter5.plot_losses(ax=ax, plot_p_q=plot_p_q)

# test_plotter6 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5m_{datetime.date.today()}/NewCustomExpert_sparse.csv',
#                             show=False, title='Sparse Reward - Expert', smooth=SMOOTHING_WINDOW)
# test_plotter6.plot_losses(ax=ax, plot_p_q=plot_p_q)

test_plotter7 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5m_{datetime.date.today()}/NewCustomNoise_dense.csv',
                            show=False, title='Dense Reward - OUR METHOD', smooth=SMOOTHING_WINDOW)
test_plotter7.plot_losses(ax=ax, plot_p_q=plot_p_q)

# test_plotter8 = DDPGMetrics(file_path=f'Envs/Pendulum/DDPG/Metrics/5m_{datetime.date.today()}/NewCustomExpert_dense.csv',
#                             show=False, title='Dense Reward - Expert', smooth=SMOOTHING_WINDOW)
# test_plotter8.plot_losses(ax=ax, plot_p_q=plot_p_q)

plt.tight_layout()
plt.legend(loc='best', fontsize='medium')
ax = plt.gca()
ax.set_facecolor('0.95')
# plt.savefig('Data/Plots/ALLTrainingComparison.svg')
plt.show()


