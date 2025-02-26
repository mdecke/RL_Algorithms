#!/bin/bash
# run_experiments.sh
# This script will run multiple experiments by calling train.py with different parameters.

# Arrays (or lists) of hyperparameters to try:
NOISES=("Gaussian" "OrnsteinUhlenbeck")
REWARDS=("dense" "sparse")

# Loop over each combination of NOISE and REWARD
for NOISE in "${NOISES[@]}"; do
  for REWARD_TYPE in "${REWARDS[@]}"; do
    echo "--------------------------------------------------"
    echo "Running training with:"
    echo "  NOISE       = $NOISE"
    echo "  REWARD_TYPE = $REWARD_TYPE"
    echo "--------------------------------------------------"

    # Because train.py is in ../DDDG/ relative to the scripts folder:
    python ../Envs/Pendulum/DDPG/train.py \
      --NB_TRAINING_CYCLES 1 \
      --NOISE "$NOISE" \
      --REWARD_TYPE "$REWARD_TYPE" \
      --training_steps 15000 \
      --batch_size 100 
  done
done

# --PLOTTING \ add this to the end of the command if you want to plot the each result