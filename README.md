## RL Algos

This repo is made of personal implementations of classical reinforcement learning agents to solve classic RL environment. These algorithms are inspired by the foundational papers as well as implementations of these algorithms found online. Please see the sources for more details.

---

## Table of Contents

- [Project Structure](#project-structure)
  - [Algorithms](#algorithms)
  - [Envs](#envs)
  - [Misc.Functions](#miscfunctions)
  - [scripts](#scripts)
- [Usage](#getting-started)
- [References](#references)

## Usage

To set up the environment, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mdecke/<your-repo>.git
   cd <your-repo>
   ```
2. **Install all dependencies**
   ```bash
   pip install --ugrade pip
   pip install -e .
   ```
3. **Run Training**
   Modfiy the scripts/run_DDPG.sh file to match your desired hyperparameters (number of training steps, cycles, individual plotting ...)
   ```bash
   scripts/run_DDPG.sh 
   ```
   The training script will:
    - Run multiple training cycles.
    - Save training metrics in the Envs/Pendulum/DDPG/Metrics/ folder.
    - Save trained models in the Envs/Pendulum/DDPG/TrainedModels/ folder.
  Go to Envs/Pendulum/DDPG/BaselineNoise.py to change saving folders if needed.

4. **Visualize Training**
   Once training is complete, you can visualize the performance metrics using:
   ```bash
   python Envs/Pendulu/DDPG/PlotMetrics.py
   ```
   This script will generate plots showing the training progress.



## References

1. Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). **OpenAI Gym**. *arXiv preprint arXiv:1606.01540*. [Link to paper](https://arxiv.org/abs/1606.01540)
2. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2019). **Continuous control with deep reinforcement learning**. *arXiv preprint arXiv:1509.02971*. [Link to paper](https://arxiv.org/abs/1509.02971)
3. Phil Tarbor youtube page: https://www.youtube.com/@MachineLearningwithPhil
4. @article{serrano2023skrl,
  author  = {Antonio Serrano-Muñoz and Dimitrios Chrysostomou and Simon Bøgh and Nestor Arana-Arexolaleiba},
  title   = {skrl: Modular and Flexible Library for Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {254},
  pages   = {1--9},
  url     = {http://jmlr.org/papers/v24/23-0112.html}
}

##License


