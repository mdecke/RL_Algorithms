import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiscFunctions.Plotting import *

plt.figure(figsize=(15, 8))

oh_ddpg = DDPGMetrics(file_path='Data/CSVs/OHNoiseTraining.csv', show=False, title='OH Noise', smooth=True)
oh_ddpg.plot_losses()

pa_ddpg = DDPGMetrics(file_path='Data/CSVs/P(a)NoiseTraining.csv', show=False, title='P(a) Noise', smooth=True)
pa_ddpg.plot_losses()

pas_ddpg = DDPGMetrics(file_path='Data/CSVs/P(a|s)NoiseTraining.csv', show=False, title='P(a[k]|s[k]) Noise', smooth=True)
pas_ddpg.plot_losses()

pasa_ddpg = DDPGMetrics(file_path='Data/CSVs/P(a|s,a)NoiseTraining.csv', show=False, title='P(a[k]|s[k],a[k-1]) Noise', smooth=True)
pasa_ddpg.plot_losses()

pas_MLE_ddpg = DDPGMetrics(file_path='Data/CSVs/P(a|s)NoiseTraining_MLE.csv', show=False, title='P(a[k]|s[k])MLE Noise', smooth=True)
pas_MLE_ddpg.plot_losses()

pasa_MLE_ddpg = DDPGMetrics(file_path='Data/CSVs/P(a|s,a)NoiseTraining_MLE.csv', show=False, title='P(a[k]|s[k],a[k-1])MLE Noise', smooth=True)
pasa_MLE_ddpg.plot_losses()

plt.tight_layout()
plt.savefig('Data/Plots/ALLTrainingComparison.svg')
plt.show()


