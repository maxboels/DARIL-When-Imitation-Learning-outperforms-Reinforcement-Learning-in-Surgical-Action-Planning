#!/bin/bash

USER="mboels"
CONFIG_NAME=$1

# Define the directory where the project is located
PROJECT_DIR="/nfs/home/$USER/projects/surl"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Change to the project directory
cd $PROJECT_DIR

# Run the training script with the specified configuration
# we removed the config flag
# python run_exp_real_data.py --config $CONFIG_NAME
python run_exp_real_data.py