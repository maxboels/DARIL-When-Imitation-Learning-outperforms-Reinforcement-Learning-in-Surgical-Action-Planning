import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import yaml
import os
from datetime import datetime

# Import custom packages
from models import *
from train import *
from utils import *
from datasets import *

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main function to run the experiment with CholecT50 data
def run_cholect50_experiment(cfg):
    """Run the experiment with CholecT50 data."""
    print("Starting CholecT50 experiment for surgical video analysis")

    # Set outputs to None
    best_model_path = None
    next_frame_model = None
    reward_model = None
    policy_model = None
    action_weights = None
    results = None
    analysis = None

    # Init logger
    logger = SimpleLogger(log_dir="logs", name="loggings")
    logger.info("Starting CholecT50 experiment for Surgical Actions Prediction")

    cfg_exp = cfg['experiment']
    logger.info(f"Experiment configuration: {cfg_exp}")
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 1: Load data
    logger.info("Loading CholecT50 data...")
    train_data = load_cholect50_data(cfg, logger, split='train', max_videos=cfg['experiment']['max_videos'])
    test_data = load_cholect50_data(cfg, logger, split='test', max_videos=cfg['experiment']['max_videos'])
    
    # Create dataloaders
    train_dataset = NextFramePredictionDataset(cfg['data'], train_data)
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    test_video_loaders = create_video_dataloaders(cfg, test_data, batch_size=16, shuffle=False)

    if cfg['experiment']['recognition']['train'] or cfg['experiment']['recognition']['inference']:
        # Step I: Recognition Head
        # Create model
        model = RecognitionHead(**cfg['models']['recognition']['transformer']).to(device)
        
        # Check if we should train or load a pre-trained model
        if cfg['experiment']['recognition']['train']:
            logger.info("Starting model training...")
            model = train_recognition_head(cfg, logger, model, train_loader, val_loader=test_video_loaders, device=device)
            # Save the trained model
            save_dir = cfg['models']['recognition'].get('save_dir', 'saved_models')
            model_path = model.save_model(save_dir)
            logger.info(f"Saved trained model to {model_path}")
        elif 'best_model_path' in cfg['experiment']['recognition']:
            # Load pre-trained model
            model_path = cfg['experiment']['recognition']['best_model_path']
            logger.info(f"Loading pre-trained model from {model_path}")
            model.load_model(model_path)

        # Run inference and evaluation
        if cfg['experiment']['recognition']['inference']:
            logger.info("Running inference and evaluation...")
            results = run_recognition_inference(cfg, logger, model, test_video_loaders, device)
            logger.info(f"Results: {results}")
    
    # Step II: World Model - Next Frame Prediction
    # 1. Pre-train next frame prediction model
    if cfg_exp['world_model']['train']:       
        print("\n[WORLD MODEL] Training next frame prediction model...")
        world_model = WorldModel(**cfg['models']['world_model']).to(device)
        best_model_path = train_world_model(cfg, logger, world_model, train_loader, test_video_loaders, device=device)  # Reduced epochs for demonstration
        logger.info(f"[WORLD MODEL] Best model saved at: {best_model_path}")
    
    # 2. Run inference for world model
    if cfg_exp['world_model']['inference']:
        logger.info("\n[WORLD MODEL] Running inference...")
        if best_model_path is None:
            best_model_path = cfg_exp['world_model']['best_model_path']
            logger.info(f"[WORLD MODEL] Using best model from pre existing path: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        world_model = WorldModel(**cfg['models']['world_model']).to(device)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        world_model.eval()
        
        # For inference
        final_metrics = evaluate_world_model(
            cfg, logger, world_model, test_video_loaders, device,
            eval_mode='full', save_results=True
        )
        logger.info(f"[WORLD MODEL] Inference evaluation completed!")

    # Step 3: Train reward prediction model
    if cfg_exp['pretrain_reward_model']:
        print("\nTraining reward prediction model...")
        reward_model = RewardPredictor(**cfg['models']['reward']).to(device)
        train_reward_model(cfg['reward_model'], train_data, device)
    
    # Calculate action rewards
    if cfg_exp['calculate_action_rewards']:
        print("\nCalculating action rewards...")
        avg_action_rewards = calculate_action_rewards(train_data, next_frame_model, reward_model, device)
    
    # Train action policy model with reward weighting
    if cfg_exp['train_action_policy']:
        print("\nTraining action policy model...")
        policy_model, action_weights = train_action_policy(cfg, train_data, avg_action_rewards, device)
    
    # Run TD-MPC2 to evaluate the model
    if cfg_exp['run_tdmpc']:
        print("\nRunning TD-MPC2...")
        results = run_tdmpc(data, next_frame_model, reward_model, policy_model, action_weights, device)
    
    # Analyze and visualize results
    if cfg_exp['analyze_results']:
        print("\nAnalyzing results...")
        analysis = analyze_results(results, action_weights)
    
    return next_frame_model, reward_model, policy_model, action_weights, results, analysis

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Load config
    config_path = 'config.yaml'
    print(f"Loading configuration from {os.path.abspath(config_path)}")
    cfg = load_config(config_path)
    print("\nConfiguration loaded successfully!")
    # Run the experiment
    next_frame_model, reward_model, policy_model, action_weights, results, analysis = run_cholect50_experiment(cfg)    
    print("\nExperiment completed!")
    if analysis:
        print(f"Model performance: {analysis['percent_improvement']:.2f}% improvement in action quality")

    print("Done!")