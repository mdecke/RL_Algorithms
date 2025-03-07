#!/bin/bash
# run_experiments.sh
# This script will run multiple experiments by calling train.py with different parameters.

# Arrays (or lists) of hyperparameters to try:
EXPERTS=("Noise" "Expert")
REWARDS=("dense" "sparse")

# Loop over each combination of EXPERT and REWARD
for EXPERT in "${EXPERTS[@]}"; do
  for REWARD_TYPE in "${REWARDS[@]}"; do
    echo "--------------------------------------------------"
    echo "Running training with:"
    echo "  Expert used as = $EXPERT"
    echo "  REWARD_TYPE    = $REWARD_TYPE"
    echo "--------------------------------------------------"

    # Because train.py is in ../DDDG/ relative to the scripts folder:
    python ../Envs/Pendulum/DDPG/train_new_expert_method.py \
      --NB_TRAINING_CYCLES 5 \
      --EXPERT "$EXPERT" \
      --REWARD_TYPE "$REWARD_TYPE" \
      --training_steps 30000 \
      --warm_up 0 \
      --batch_size 100 \
      
  done
done

# --PLOTTING \ add this to the end of the command if you want to plot the each result