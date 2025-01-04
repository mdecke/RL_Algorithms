# Bidep_MT

This repo is a reinforcement learning project focusing on solving the classic pendulum environment using various algorithms and tools such as energy shaping controllers, LQR, DDPG, and more. It mainly focuses on changing the usual aproach of inserting gaussian or OH noise for exploration while training to some sort of guided guess based on some distribution of so called good actions.

---

## Table of Contents

- [Project Structure](#project-structure)
  - [Algorithms](#algorithms)
  - [Data/Plots](#dataplots)
  - [Description](#description)
  - [Misc.Functions](#miscfunctions)
  - [Test](#test)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Running the Simulations](#running-the-simulations)
- [Usage Notes](#usage-notes)
  - [Parameters to Tune](#parameters-to-tune)
  - [Data Management](#data-management)
- [References](#references)

---

## Project Structure

### Algorithms
- **Purpose**: Contains scripts for reinforcement learning algorithms.
- **Current content**:  
  - Only the **DDPG** algorithm is implemented.  
    It is inspired by [skrl](https://skrl.readthedocs.io/en/latest/) and the DDPG paper [[1]](#references).

### Data/Plots
- **Purpose**: Contains all data, plots, and saved models.
- **Current content**:  
  - Only the `Plots` subfolder is kept, storing all generated plots in `.svg` format.
    - `.svg` files can be opened in a browser or imported into documents.

### Description
- **Purpose**: Contains LaTeX scripts that explain how each script in the repository was written.

### Misc.Functions
- **Purpose**: Utility scripts for simulations, data processing, and plotting.
  - **SimulationData.py**: Solves the pendulum environment using energy shaping and LQR.
  - **data_processing.py**: Contains classes for processing data.
  - **EMAlgo.py**: Implements a custom Expectation-Maximization algorithm and hyperparameter tuning with Optuna.
  - **Plotting.py**: Includes classes for generating plots.

### Test
- **Purpose**: Contains executable scripts for testing algorithms and generating results.
  - Each script outputs a `.csv` file containing policy value losses and episodic returns.
  - A dedicated script plots the results from these tests.

---

## Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/YourRepositoryName.git
   cd YourRepositoryName

### Running The Simulation

1. **Generate the necessary data**: Run the `SimulationData.py` script to create a `.csv` file:
    ```bash
    python Misc.Functions/SimulationData.py

2. **Run test scripts**: Navigate to the Tests folder and execute the desired test script. For example:
    ```bash
    python Tests/DDPG_P(a|s)_MLE.py
    ```
    Each test generates a `.csv` file with policy value losses and episodic returns. These files will be saved in the Data folder in a sub folder called CSVs

3. **Plot Results**: In the Tests folder, execute the script called `DDPGMetricsPlotting.py`. This file generates different plots based on the selected data (the `.csv` files
   produced from step 3.
   ```bash
   python Tests/DDPGMetricsPlotting.py
   ```
   These plots will be saved in the Data folder, in the Plots sub folder. 
   
---

## References

1. Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). **OpenAI Gym**. *arXiv preprint arXiv:1606.01540*. [Link to paper](https://arxiv.org/abs/1606.01540)
2. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2019). **Continuous control with deep reinforcement learning**. *arXiv preprint arXiv:1509.02971*. [Link to paper](https://arxiv.org/abs/1509.02971)
