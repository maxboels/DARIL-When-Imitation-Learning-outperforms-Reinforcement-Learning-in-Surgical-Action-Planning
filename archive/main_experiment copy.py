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
from typing import Dict, List, Any, Optional, Tuple

# Import your existing modules
from datasets.cholect50 import load_cholect50_data, NextFramePredictionDataset, create_video_dataloaders
from utils.logger import SimpleLogger

# Import new components
from models.dual_world_model import DualWorldModel
from trainer.dual_trainer import DualTrainer, train_dual_world_model
from evaluation.dual_evaluator import DualModelEvaluator
from rl.environment import SurgicalWorldModelEnv, MultiVideoSurgicalEnv

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_dual_world_model_experiment(cfg):
    """
    Run the complete dual world model experiment supporting both supervised and RL training.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary containing experiment results
    """
    print("Starting Dual World Model experiment for surgical video analysis")
    
    # Initialize logger
    logger = SimpleLogger(log_dir="logs", name="dual_world_model")
    logger.info("Starting Dual World Model experiment")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize results container
    experiment_results = {
        'supervised_training': None,
        'rl_training': None,
        'evaluation_results': None,
        'model_paths': {},
        'config': cfg
    }
    
    # Step 1: Load data
    logger.info("Loading CholecT50 data...")
    train_data = load_cholect50_data(cfg, logger, split='train', max_videos=cfg['experiment']['train']['max_videos'])
    test_data = load_cholect50_data(cfg, logger, split='test', max_videos=cfg['experiment']['test']['max_videos'])
    
    # Create datasets and dataloaders
    train_dataset = NextFramePredictionDataset(cfg['data'], train_data)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory']
    )
    
    test_video_loaders = create_video_dataloaders(cfg, test_data, batch_size=16, shuffle=False)
    
    logger.info(f"Max training videos: {cfg['experiment']['train']['max_videos']}")
    logger.info(f"Max test videos: {cfg['experiment']['test']['max_videos']}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test videos: {len(test_video_loaders)}")
    
    # Step 2: Initialize Dual World Model
    logger.info("Initializing Dual World Model...")
    model_config = cfg['models']['dual_world_model']
    model = DualWorldModel(**model_config).to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Step 3: Training Phase
    training_mode = cfg.get('training_mode', 'supervised')
    logger.info(f"Training mode: {training_mode}")
    
    if training_mode in ['supervised', 'mixed']:
        # Supervised training for autoregressive action prediction
        logger.info("Starting supervised training...")
        cfg['training_mode'] = 'supervised'
        
        try:
            supervised_model_path = train_dual_world_model(cfg, logger, model, train_loader, test_video_loaders, device)
            experiment_results['model_paths']['supervised'] = supervised_model_path
            experiment_results['supervised_training'] = {'status': 'completed', 'model_path': supervised_model_path}
            logger.info(f"Supervised training completed. Model saved: {supervised_model_path}")
        except Exception as e:
            logger.error(f"Supervised training failed: {str(e)}")
            experiment_results['supervised_training'] = {'status': 'failed', 'error': str(e)}
            return experiment_results
    
    if training_mode in ['rl', 'mixed']:
        # RL training for state and reward prediction
        logger.info("Starting RL training...")
        
        # If mixed mode, load the supervised model first
        if training_mode == 'mixed' and 'supervised' in experiment_results['model_paths']:
            logger.info("Loading supervised model for RL fine-tuning...")
            model = DualWorldModel.load_model(experiment_results['model_paths']['supervised'], device)
        
        cfg['training_mode'] = 'rl'
        
        try:
            rl_model_path = train_dual_world_model(cfg, logger, model, train_loader, test_video_loaders, device)
            experiment_results['model_paths']['rl'] = rl_model_path
            experiment_results['rl_training'] = {'status': 'completed', 'model_path': rl_model_path}
            logger.info(f"RL training completed. Model saved: {rl_model_path}")
        except Exception as e:
            logger.error(f"RL training failed: {str(e)}")
            experiment_results['rl_training'] = {'status': 'failed', 'error': str(e)}
            if training_mode == 'rl':  # If pure RL mode failed, return
                return experiment_results
    
    # Step 4: Evaluation
    logger.info("Starting model evaluation...")
    
    # Determine which model to evaluate
    if 'rl' in experiment_results['model_paths']:
        eval_model_path = experiment_results['model_paths']['rl']
        logger.info("Evaluating RL-trained model")
    elif 'supervised' in experiment_results['model_paths']:
        eval_model_path = experiment_results['model_paths']['supervised']
        logger.info("Evaluating supervised-trained model")
    else:
        logger.error("No trained model available for evaluation")
        return experiment_results
    
    try:
        # Load model for evaluation
        eval_model = DualWorldModel.load_model(eval_model_path, device)
        
        # Create evaluator
        evaluator = DualModelEvaluator(eval_model, cfg, device, logger)
        
        # Run comprehensive evaluation
        eval_results = evaluator.evaluate_both_modes(
            test_video_loaders, 
            save_results=True, 
            save_dir=os.path.join(logger.log_dir, "evaluation_results")
        )
        
        experiment_results['evaluation_results'] = eval_results
        logger.info("Evaluation completed successfully")
        
        # Log key metrics
        if 'supervised' in eval_results:
            sup_results = eval_results['supervised']
            if 'action_prediction' in sup_results:
                action_acc = sup_results['action_prediction'].get('single_step_action_exact_match', {}).get('mean', 0)
                logger.info(f"Action prediction accuracy: {action_acc:.4f}")
            
            if 'state_prediction' in sup_results:
                state_mse = sup_results['state_prediction'].get('single_step_state_mse', {}).get('mean', 0)
                logger.info(f"State prediction MSE: {state_mse:.4f}")
        
        if 'rl' in eval_results:
            rl_results = eval_results['rl']
            if 'state_prediction' in rl_results:
                rl_state_mse = rl_results['state_prediction'].get('state_mse', {}).get('mean', 0)
                logger.info(f"RL state prediction MSE: {rl_state_mse:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        experiment_results['evaluation_results'] = {'status': 'failed', 'error': str(e)}
    
    # Step 5: RL Environment Testing (if RL experiments are enabled)
    if False:  # PATCHED: Temporarily disable RL experiments
    # if cfg.get('experiment', {}).get('rl_experiments', {}).get('enabled', False):
        logger.info("Starting RL environment experiments...")
        
        try:
            rl_env_results = run_rl_environment_experiments(cfg, logger, eval_model, train_data, device)
            experiment_results['rl_environment'] = rl_env_results
            logger.info("RL environment experiments completed")
        except Exception as e:
            logger.error(f"RL environment experiments failed: {str(e)}")
            experiment_results['rl_environment'] = {'status': 'failed', 'error': str(e)}
    
    # Step 6: Generate Summary Report
    logger.info("Generating summary report...")
    summary_report = generate_experiment_summary(experiment_results, logger)
    
    # Save summary report
    summary_path = os.path.join(logger.log_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    logger.info(f"Experiment completed! Summary saved to: {summary_path}")
    
    return experiment_results

def run_rl_environment_experiments(cfg, logger, model, train_data, device):
    """
    Run RL environment experiments using trained world model.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        model: Trained DualWorldModel
        train_data: Training data for environment context
        device: Device to run on
        
    Returns:
        Dictionary of RL experiment results
    """
    logger.info("Setting up RL environment experiments")
    
    rl_config = cfg.get('rl_training', {})
    results = {}
    
    # Create environment
    env_config = {
        'rl_horizon': rl_config.get('rl_horizon', 50),
        'context_length': cfg['data']['context_length'],
        'reward_weights': rl_config.get('reward_weights', {}),
        'normalize_rewards': rl_config.get('normalize_rewards', True),
        'early_termination': rl_config.get('early_termination', True)
    }
    
    # Use multi-video environment for diversity
    env = MultiVideoSurgicalEnv(model, env_config, train_data, device)
    
    # Test 1: Random Policy Baseline
    logger.info("Testing random policy baseline...")
    random_results = test_random_policy(env, num_episodes=10)
    results['random_policy'] = random_results
    
    # Test 2: Behavioral Cloning (using model's action predictions)
    logger.info("Testing behavioral cloning policy...")
    bc_results = test_behavioral_cloning_policy(env, model, num_episodes=10)
    results['behavioral_cloning'] = bc_results
    
    # Test 3: RL Training (if algorithms are specified)
    algorithms = cfg.get('experiment', {}).get('rl_experiments', {}).get('algorithms', [])
    
    if algorithms:
        try:
            from trainer.rl_trainer import RLExperimentRunner
            
            # Create RL trainer
            rl_trainer = RLExperimentRunner(cfg, logger)
            rl_trainer.load_world_model(cfg['experiment']['dual_world_model']['best_model_path'])
            
            # Run RL experiments
            rl_training_results = rl_trainer.run_rl_experiments(train_data, algorithms)
            results['rl_algorithms'] = rl_training_results
            
        except ImportError:
            logger.warning("RL training dependencies not available, skipping RL algorithm experiments")
            results['rl_algorithms'] = {'status': 'skipped', 'reason': 'dependencies_not_available'}
        except Exception as e:
            logger.error(f"RL algorithm experiments failed: {str(e)}")
            results['rl_algorithms'] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_random_policy(env, num_episodes=10):
    """Test a random policy in the environment."""
    episode_rewards = []
    episode_lengths = []
    episode_metrics = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                episode_metrics.append(env.base_env.get_episode_metrics())
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'episode_metrics': episode_metrics
    }

def test_behavioral_cloning_policy(env, model, num_episodes=10):
    """Test behavioral cloning policy using model's action predictions."""
    episode_rewards = []
    episode_lengths = []
    episode_metrics = []
    
    model.eval()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Use model to predict action
            with torch.no_grad():
                # Convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device)
                obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
                
                # Generate action prediction
                generation_output = model.autoregressive_action_prediction(
                    initial_states=obs_tensor,
                    horizon=1,
                    temperature=0.8
                )
                
                # Extract action
                if 'predicted_actions' in generation_output:
                    action_probs = generation_output['predicted_actions'][0, 0].cpu().numpy()
                    action = (action_probs > 0.5).astype(np.float32)
                else:
                    # Fallback to random action
                    action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                episode_metrics.append(env.base_env.get_episode_metrics())
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'episode_metrics': episode_metrics
    }

def generate_experiment_summary(experiment_results, logger):
    """
    Generate a comprehensive summary of the experiment results.
    
    Args:
        experiment_results: Dictionary of experiment results
        logger: Logger instance
        
    Returns:
        Dictionary containing experiment summary
    """
    # PATCHED: Handle None experiment_results
    if experiment_results is None:
        logger.warning("Experiment results are None, creating empty summary")
        return {
            'experiment_timestamp': datetime.now().isoformat(),
            'training_summary': {},
            'evaluation_summary': {},
            'rl_summary': {},
            'recommendations': ['⚠ Experiment results were None - check for errors']
        }
    
    summary = {
        'experiment_timestamp': datetime.now().isoformat(),
        'training_summary': {},
        'evaluation_summary': {},
        'rl_summary': {},
        'recommendations': []
    }
    
    # Training summary
    if experiment_results.get('supervised_training', {}):
        summary['training_summary']['supervised'] = experiment_results['supervised_training']['status']
    
    if experiment_results.get('rl_training', {}):
        summary['training_summary']['rl'] = experiment_results['rl_training']['status']
    
    # Evaluation summary
    eval_results = experiment_results.get('evaluation_results')
    if eval_results and 'comparison' in eval_results:
        comparison = eval_results['comparison']
        
        if 'state_prediction' in comparison:
            state_comp = comparison['state_prediction']
            summary['evaluation_summary']['state_prediction'] = {
                'better_mode': state_comp.get('better_mode'),
                'improvement': state_comp.get('improvement', 0),
                'supervised_mse': state_comp.get('supervised_mse', 0),
                'rl_mse': state_comp.get('rl_mse', 0)
            }
    
    # RL environment summary
    rl_env_results = experiment_results.get('rl_environment')
    if rl_env_results:
        summary['rl_summary'] = {
            'random_policy_reward': rl_env_results.get('random_policy', {}).get('avg_reward', 0),
            'behavioral_cloning_reward': rl_env_results.get('behavioral_cloning', {}).get('avg_reward', 0),
            'rl_algorithms_status': rl_env_results.get('rl_algorithms', {}).get('status', 'not_run')
        }
    
    # Generate recommendations
    recommendations = []
    
    # Training recommendations
    supervised_training = experiment_results.get('supervised_training') or {}
    if supervised_training.get('status') == 'completed':
        recommendations.append("✓ Supervised training completed successfully")
    else:
        recommendations.append("⚠ Consider debugging supervised training issues")
    
    if experiment_results.get('rl_training', {}).get('status') == 'completed':
        recommendations.append("✓ RL training completed successfully")
    
    # Evaluation recommendations
    if eval_results:
        if 'supervised' in eval_results:
            sup_action_acc = eval_results['supervised'].get('action_prediction', {}).get('single_step_action_exact_match', {}).get('mean', 0)
            if sup_action_acc > 0.7:
                recommendations.append("✓ Strong action prediction performance")
            elif sup_action_acc > 0.5:
                recommendations.append("→ Moderate action prediction performance, consider more training")
            else:
                recommendations.append("⚠ Low action prediction performance, check data quality and model architecture")
        
        if 'comparison' in eval_results and 'state_prediction' in eval_results['comparison']:
            better_mode = eval_results['comparison']['state_prediction'].get('better_mode')
            if better_mode:
                recommendations.append(f"→ {better_mode.title()} mode performs better for state prediction")
    
    # RL environment recommendations
    if rl_env_results:
        random_reward = rl_env_results.get('random_policy', {}).get('avg_reward', 0)
        bc_reward = rl_env_results.get('behavioral_cloning', {}).get('avg_reward', 0)
        
        if bc_reward > random_reward * 1.5:
            recommendations.append("✓ Behavioral cloning shows significant improvement over random policy")
        else:
            recommendations.append("⚠ Behavioral cloning performance is poor, consider model improvements")
    
    summary['recommendations'] = recommendations
    
    # Log summary
    logger.info("=== EXPERIMENT SUMMARY ===")
    for recommendation in recommendations:
        logger.info(recommendation)
    
    return summary

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to run the dual world model experiment."""
    # Load configuration
    config_path = 'config_dual.yaml'  # Use the new dual config
    if not os.path.exists(config_path):
        config_path = 'config.yaml'  # Fallback to original config
    
    print(f"Loading configuration from {os.path.abspath(config_path)}")
    cfg = load_config(config_path)
    
    # Set up paths and create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Run the experiment
    try:
        results = run_dual_world_model_experiment(cfg)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Print key results
        if results.get('evaluation_results'):
            eval_results = results['evaluation_results']
            
            if 'supervised' in eval_results:
                sup_results = eval_results['supervised']
                action_acc = sup_results.get('action_prediction', {}).get('single_step_action_exact_match', {}).get('mean', 0)
                state_mse = sup_results.get('state_prediction', {}).get('single_step_state_mse', {}).get('mean', 0)
                print(f"Supervised Mode - Action Accuracy: {action_acc:.4f}, State MSE: {state_mse:.4f}")
            
            if 'rl' in eval_results:
                rl_results = eval_results['rl']
                rl_state_mse = rl_results.get('state_prediction', {}).get('state_mse', {}).get('mean', 0)
                print(f"RL Mode - State MSE: {rl_state_mse:.4f}")
        
        if results.get('rl_environment'):
            rl_env = results['rl_environment']
            random_reward = rl_env.get('random_policy', {}).get('avg_reward', 0)
            bc_reward = rl_env.get('behavioral_cloning', {}).get('avg_reward', 0)
            print(f"RL Environment - Random Policy: {random_reward:.3f}, Behavioral Cloning: {bc_reward:.3f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())