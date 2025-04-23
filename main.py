import os
import torch
import argparse
import yaml
import logging
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

# Import custom modules
from datasets import (
    load_cholect50_data,
    NextFramePredictionDataset,
    NextFramePredictionDataset, 
    EnhancedRewardModelDataset,
    SurgicalDataPreprocessor
)
from world_model_improvements import SurgicalWorldModel, SurgicalRewardModel
from training_functions import SurgicalTrainer
from evaluation_functions import SurgicalEvaluator

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file and console output."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def create_output_dirs(config):
    """Create necessary output directories."""
    dirs = [
        config['training']['log_dir'],
        config['training']['checkpoint_dir'],
        config['evaluation']['save_dir'],
        'preprocessed',
        'visualizations'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def run_experiment(config, args):
    """
    Run the full surgical world model experiment.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"surgical_world_model_{timestamp}"
    
    # Create output directories
    create_output_dirs(config)
    
    # Setup logger
    log_file = os.path.join(config['training']['log_dir'], f"{run_name}.log")
    logger = setup_logger(run_name, log_file)
    
    logger.info(f"Starting experiment: {run_name}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Mode: {args.mode}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load data
    logger.info("Loading data...")
    train_data = load_cholect50_data(config['data'], split='train', max_videos=config['experiment']['max_videos'])
    test_data = load_cholect50_data(config['data'], split='test', max_videos=config['experiment']['max_videos'])
    
    logger.info(f"Loaded {len(train_data)} training videos and {len(test_data)} test videos")
    
    # Create surgical preprocessor
    preprocessor = SurgicalDataPreprocessor(config['data'])
    
    if args.mode in ['train', 'full']:
        # Identify essential steps and analyze workflow
        logger.info("Analyzing surgical workflow...")
        preprocessor.identify_essential_steps(train_data)
        preprocessor.build_transition_matrix(train_data)
        preprocessor.calculate_step_importance()
        
        # Save preprocessor
        preprocessor.save('preprocessed/surgical_preprocessor.json')
        logger.info("Surgical workflow analysis completed and saved")
    else:
        # Load preprocessor for evaluation/inference
        logger.info("Loading surgical workflow analysis...")
        if os.path.exists('preprocessed/surgical_preprocessor.json'):
            preprocessor = SurgicalDataPreprocessor.load('preprocessed/surgical_preprocessor.json', config['data'])
            logger.info("Loaded surgical workflow analysis")
        else:
            logger.warning("No preprocessor found. Will proceed without surgical workflow context.")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    # Create training dataset
    train_dataset = NextFramePredictionDataset(
        config['data'], 
        train_data,
        surgical_preprocessor=preprocessor,
        split='train'
    )
    
    # Create validation dataset
    val_dataset = NextFramePredictionDataset(
        config['data'], 
        test_data,
        surgical_preprocessor=preprocessor,
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create trainer
    trainer = SurgicalTrainer(config, logger)
    
    # Create evaluator
    evaluator = SurgicalEvaluator(config, logger)
    
    # Initialize models
    world_model = None
    reward_model = None
    policy_model = None
    
    #===========================
    # World Model Training
    #===========================
    if args.mode in ['train', 'train-world', 'full']:
        logger.info("Initializing surgical world model...")
        
        # Create world model
        world_model = SurgicalWorldModel(
            hidden_dim=config['models']['world_model']['hidden_dim'],
            embedding_dim=config['models']['world_model']['embedding_dim'],
            action_embedding_dim=config['models']['world_model']['action_embedding_dim'],
            n_layer=config['models']['world_model']['n_layer'],
            max_length=config['models']['world_model']['max_length'],
            use_head=config['models']['world_model']['use_head'],
            targets_dims=config['models']['world_model']['targets_dims'],
            target_heads=config['models']['world_model']['target_heads'],
            loss_weights=config['models']['world_model']['loss_weights'],
            num_action_classes=config['models']['world_model']['num_action_classes'],
            num_steps=config['models']['world_model'].get('num_steps', 15),
            imitation_learning=config['models']['world_model']['imitation_learning']
        ).to(device)
        
        logger.info("Starting world model training...")
        best_model_path = trainer.train_world_model(world_model, train_loader, val_loader)
        logger.info(f"World model training completed. Best model saved at: {best_model_path}")
        
        # Evaluate surgical awareness
        logger.info("Evaluating surgical workflow awareness...")
        surgical_metrics = evaluator.evaluate_surgical_awareness(world_model, val_loader, preprocessor)
        logger.info(f"Surgical awareness evaluation completed: {surgical_metrics}")
        
        # Evaluate multi-step prediction
        logger.info("Evaluating multi-step prediction...")
        prediction_metrics = evaluator.evaluate_multi_step_prediction(world_model, val_loader)
        logger.info(f"Multi-step evaluation completed: {prediction_metrics}")
    
    #===========================
    # Reward Model Training
    #===========================
    if args.mode in ['train', 'train-reward', 'full']:
        # Initialize reward model
        logger.info("Initializing surgical reward model...")
        
        # Create reward model
        reward_model = SurgicalRewardModel(
            embedding_dim=config['models']['reward_model']['embedding_dim'],
            hidden_dim=config['models']['reward_model']['hidden_dim'],
            num_heads=config['models']['reward_model']['num_heads'],
            num_layers=config['models']['reward_model']['num_layers'],
            dropout=config['models']['reward_model']['dropout'],
            num_action_classes=config['models']['reward_model']['num_action_classes'],
            num_steps=config['models']['reward_model'].get('num_steps', 15)
        ).to(device)
        
        # Create reward dataset
        reward_train_dataset = EnhancedRewardModelDataset(
            config['data'],
            train_data,
            preprocessor=preprocessor
        )
        
        reward_val_dataset = EnhancedRewardModelDataset(
            config['data'],
            test_data,
            preprocessor=preprocessor
        )
        
        # Create reward dataloaders
        reward_train_loader = DataLoader(
            reward_train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        reward_val_loader = DataLoader(
            reward_val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        # Train reward model
        logger.info("Starting reward model training...")
        best_reward_path = trainer.train_reward_model(reward_model, reward_train_loader, reward_val_loader)
        logger.info(f"Reward model training completed. Best model saved at: {best_reward_path}")
        
        # Load world model if needed
        if world_model is None and os.path.exists(config['experiment']['pretrain_world_model']['best_model_path']):
            logger.info("Loading pretrained world model...")
            world_model_path = config['experiment']['pretrain_world_model']['best_model_path']
            world_model = SurgicalWorldModel.load(world_model_path, device)
        
        # Evaluate risk-completeness balance
        if world_model is not None:
            logger.info("Evaluating risk-completeness balance...")
            balance_metrics = evaluator.evaluate_risk_completeness_balance(
                world_model, reward_model, val_loader
            )
            logger.info(f"Risk-completeness evaluation completed: {balance_metrics}")
    
    #===========================
    # Bi-level Optimization
    #===========================
    if args.mode in ['train', 'train-policy', 'full']:
        # Load models if needed
        if world_model is None and os.path.exists(config['experiment']['pretrain_world_model']['best_model_path']):
            logger.info("Loading pretrained world model...")
            world_model_path = config['experiment']['pretrain_world_model']['best_model_path']
            world_model = SurgicalWorldModel.load(world_model_path, device)
        
        if reward_model is None and os.path.exists(config['experiment']['pretrain_reward_model']['best_model_path']):
            logger.info("Loading pretrained reward model...")
            reward_model_path = config['experiment']['pretrain_reward_model']['best_model_path']
            reward_model = SurgicalRewardModel.load(reward_model_path, device)
        
        # Initialize policy model
        logger.info("Initializing policy model...")
        from torch import nn
        
        # Simple policy model
        policy_model = nn.Sequential(
            nn.Linear(config['data']['embedding_dim'], config['models']['policy_model']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['models']['policy_model']['dropout']),
            nn.Linear(config['models']['policy_model']['hidden_dim'], config['models']['policy_model']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['models']['policy_model']['dropout']),
            nn.Linear(config['models']['policy_model']['hidden_dim'], config['models']['policy_model']['action_dim'])
        ).to(device)
        
        # Run bi-level optimization
        logger.info("Starting bi-level optimization...")
        best_reward_path, best_policy_path = trainer.run_bi_level_optimization(
            world_model, reward_model, policy_model, train_loader, val_loader
        )
        logger.info(f"Bi-level optimization completed.")
        logger.info(f"Best reward model saved at: {best_reward_path}")
        logger.info(f"Best policy model saved at: {best_policy_path}")
    
    #===========================
    # Evaluation
    #===========================
    if args.mode in ['eval', 'full']:
        # Load models if needed
        if world_model is None:
            # Try to load from configuration or use latest
            model_path = config['experiment']['pretrain_world_model']['best_model_path']
            if model_path is None or not os.path.exists(model_path):
                # Try to find latest model
                model_dir = config['training']['checkpoint_dir']
                model_files = [f for f in os.listdir(model_dir) if f.startswith('world_model_') and f.endswith('.pt')]
                if model_files:
                    model_path = os.path.join(model_dir, sorted(model_files)[-1])
                
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading world model from: {model_path}")
                world_model = SurgicalWorldModel.load(model_path, device)
            else:
                logger.error("No world model found for evaluation")
                return
        
        if reward_model is None:
            # Try to load from configuration or use latest
            model_path = config['experiment']['pretrain_reward_model']['best_model_path']
            if model_path is None or not os.path.exists(model_path):
                # Try to find latest model
                model_dir = config['training']['checkpoint_dir']
                model_files = [f for f in os.listdir(model_dir) if f.startswith('reward_model_') and f.endswith('.pt')]
                if model_files:
                    model_path = os.path.join(model_dir, sorted(model_files)[-1])
                
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading reward model from: {model_path}")
                reward_model = SurgicalRewardModel.load(model_path, device)
        
        # Try to load policy model
        policy_model_path = config['experiment']['bi_level_optimization']['best_policy_path']
        policy_model = None
        if policy_model_path and os.path.exists(policy_model_path):
            logger.info(f"Loading policy model from: {policy_model_path}")
            policy_model = torch.load(policy_model_path, map_location=device)
        else:
            # Look for latest policy model
            model_dir = config['training']['checkpoint_dir']
            model_files = [f for f in os.listdir(model_dir) if f.startswith('policy_model_') and f.endswith('.pt')]
            if model_files:
                policy_model_path = os.path.join(model_dir, sorted(model_files)[-1])
                logger.info(f"Loading policy model from: {policy_model_path}")
                policy_model = torch.load(policy_model_path, map_location=device)
        
        # Comprehensive evaluation
        logger.info("Starting comprehensive evaluation...")
        
        # 1. Evaluate surgical awareness
        logger.info("Evaluating surgical workflow awareness...")
        surgical_metrics = evaluator.evaluate_surgical_awareness(world_model, val_loader, preprocessor)
        logger.info(f"Surgical awareness metrics: {surgical_metrics}")
        
        # 2. Evaluate multi-step prediction
        logger.info("Evaluating multi-step prediction...")
        prediction_metrics = evaluator.evaluate_multi_step_prediction(world_model, val_loader)
        logger.info(f"Multi-step prediction metrics: {prediction_metrics}")
        
        # 3. Analyze essential steps
        logger.info("Analyzing essential surgical steps...")
        step_analysis = evaluator.analyze_essential_steps(world_model, preprocessor, val_loader)
        logger.info(f"Step analysis completed")
        
        # 4. Evaluate risk-completeness balance
        if reward_model is not None:
            logger.info("Evaluating risk-completeness balance...")
            balance_metrics = evaluator.evaluate_risk_completeness_balance(
                world_model, reward_model, val_loader
            )
            logger.info(f"Risk-completeness balance metrics: {balance_metrics}")
        
        # 5. Compare policy vs expert
        if policy_model is not None and reward_model is not None:
            logger.info("Comparing policy trajectories with expert demonstrations...")
            comparison_metrics = evaluator.compare_policy_vs_expert(
                world_model, reward_model, policy_model, val_loader
            )
            logger.info(f"Policy vs expert comparison: {comparison_metrics}")
        
        # Generate comprehensive evaluation report
        logger.info("Generating comprehensive evaluation report...")
        report_path = os.path.join(config['evaluation']['save_dir'], f"{run_name}_evaluation_report.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Surgical World Model Evaluation Report\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 1. Surgical Workflow Awareness\n\n")
            f.write(f"```\n{surgical_metrics}\n```\n\n")
            
            f.write(f"## 2. Multi-step Prediction Accuracy\n\n")
            f.write(f"```\n{prediction_metrics}\n```\n\n")
            
            f.write(f"## 3. Essential Steps Analysis\n\n")
            f.write(f"See visualizations in the output directory\n\n")
            
            if reward_model is not None:
                f.write(f"## 4. Risk-Completeness Balance\n\n")
                f.write(f"```\n{balance_metrics}\n```\n\n")
            
            if policy_model is not None and reward_model is not None:
                f.write(f"## 5. Policy vs Expert Comparison\n\n")
                f.write(f"```\n{comparison_metrics}\n```\n\n")
        
        logger.info(f"Evaluation report saved to: {report_path}")
    
    logger.info("Experiment completed successfully!")
    return world_model, reward_model, policy_model

def main():
    parser = argparse.ArgumentParser(description='Surgical World Model Experiment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['train', 'train-world', 'train-reward', 'train-policy', 'eval', 'full'],
                        help='Experiment mode')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with minimal data')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Adjust configuration for debug mode
    if args.debug:
        print("Running in debug mode with minimal data")
        config['experiment']['max_videos'] = 2
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 4
        config['bi_level_optimization']['num_iterations'] = 2
    
    # Run experiment
    world_model, reward_model, policy_model = run_experiment(config, args)
    
    print("Experiment completed!")

if __name__ == "__main__":
    main()