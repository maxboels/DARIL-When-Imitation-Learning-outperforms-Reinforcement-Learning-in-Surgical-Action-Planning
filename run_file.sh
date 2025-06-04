#!/bin/bash

USER="mboels"
CONFIG_NAME=$1

# Define the directory where the project is located
PROJECT_DIR="/nfs/home/$USER/projects/surl"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Change to the project directory
cd $PROJECT_DIR


# Fix OpenCV issue on DGX server
echo "🔧 Fixing OpenCV compatibility issue on DGX server..."

# Option 1: Reinstall opencv-python
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python==4.8.1.78

# Option 2: If that doesn't work, try downgrading
# pip install opencv-python==4.5.5.64

# Option 3: Install specific versions
# pip install opencv-python==4.7.1.72

echo "✅ OpenCV fix attempted. Try running your script again."
echo "If this doesn't work, use the fallback solution below."

# Run the training script with the specified configuration
python run_experiment_v2.py