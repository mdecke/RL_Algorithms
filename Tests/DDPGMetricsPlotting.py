import os
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.DDPG import *
from MiscFunctions.Plotting import *

plt.figure(figsize=(15, 8))
action_dist_noise = DDPGMetrics(file_path='Action_dist_train_losses.csv', show=False, title='P(a[k]) Noise')
action_dist_noise.plot_losses()

prev_action_ddpg = DDPGMetrics(file_path='PrevActionLoss.csv', show=False, title='P(a[k]|s[k],a[k-1]) Noise')
prev_action_ddpg.plot_losses()

state_action_ddpg = DDPGMetrics(file_path='StateActionLosses.csv', show=False, title='P(a[k]|s[k]) Noise')
state_action_ddpg.plot_losses()

oh_ddpg = DDPGMetrics(file_path='Train_losses.csv', show=False, title='n ~ OH() Noise')
oh_ddpg.plot_losses()

plt.tight_layout()
plt.show()


