#!/bin/bash
# run_experiments.sh

# 1. Autoregressive baseline
echo "Running autoregressive baseline..."
python train.py --mode single --config configs/ar_config.yaml

# 2. Parallel decoder
echo "Running parallel decoder..."
python train.py --mode single --config configs/parallel_config.yaml

# 3. Diffusion decoder
echo "Running diffusion decoder..."
python train.py --mode single --config configs/diffusion_config.yaml

# Optional: Run full ablation
# python train.py --mode ablation