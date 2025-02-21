import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiscFunctions.Plotting import *

fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))

gauss_plotter = DDPGMetrics(file_path='Data/CSVs/Metrics/tests/Gaussian.csv', show=False, title='Gaussian added Noise')
gauss_plotter.plot_losses(ax=ax)

ou_plotter = DDPGMetrics(file_path='Data/CSVs/Metrics/tests/OrnsteinUhlenbeck.csv', show=False, title='Ornstein-Uhlenbeck added Noise')
ou_plotter.plot_losses(ax=ax)

plt.tight_layout()
# plt.savefig('Data/Plots/ALLTrainingComparison.svg')
plt.show()


