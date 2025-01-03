import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MiscFunctions.DataProcessing import *
from MiscFunctions.Plotting import *

FILE_PATH = 'Data/CSVs/data_1000.csv' 
NB_CLUSTERS = 50
NB_TRIALS = 100


clusterer = Cluster(file_path=FILE_PATH, nb_clusters=NB_CLUSTERS)
augemented_data = clusterer.cluster_data(what_to_cluster='states')

plotter = ClusterPlotting(data=augemented_data)
plotter.plot_clusters(data_type='polar', save=True)

fitter = FitGMM(data=augemented_data, nb_trials=NB_TRIALS)
gmm_dict = fitter.fit_gmm()

plotter.plot_fitted_cluster_gmm(gmm_dict, save=True)