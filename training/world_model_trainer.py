#!/usr/bin/env python3
"""
FIXED World Model Trainer for Method 2
Action-conditioned forward simulation training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import json


class WorldModelTrainer:
    """
    FIXED Trainer for Conditional World Model (Method 2).
    
    Focuses on:
    1. Action-conditioned state prediction
    2. Multi-type reward prediction  
    3. Forward simulation capability for RL
    """
    
    def __init__(self, 
                 model,
                 config: Dict[str, Any],
                 logger,
                 device: str = 'cuda'):
        """
        Initialize the world model trainer.
        
        Args:
            model: ConditionalWorldModel instance
            config: Configuration dictionary
            logger: Logger instance
            device: Device to train on
        """
        
        self.model = model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Training configuration
        self.train_config = config['training']
        self.epochs = self.train_config['epochs']
        self.lr = self.train_config['learning_rate']
        self.weight_decay = self.train_config['weight_decay']
        self.clip_grad = self.train_config.get('gradient_clip_val', 1.0)
        self.log_every_n_steps = self.train_config.get('log_every_n_steps', 100)
        
        # Setup logging directories
        self.log_dir = logger.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Tensorboard
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard', 'world_model')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Track metrics
        self.metrics_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        self.logger.info("🌍 World Model Trainer initialized")
        self.logger.info(f"   Device: {device}")
        self.logger.info(f"   Epochs: {self.epochs}")
        self.logger.info(f"   Learning rate: {self.lr}")
        self.logger.info(f"   Focus: Action-conditioned state-reward prediction")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        
        # Different learning rates for different components
        param_groups = []
        
        # Transformer backbone
        transformer_params = []
        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                transformer_params.append(param)
        
        if transformer_params:
            param_groups.append({
                'params': transformer_params,
                'lr': self.lr * 0.8,  # Slightly lower for backbone
                'weight_decay': self.weight_decay
            })
        
        # State prediction head (most important)
        state_params = []
        for name, param in self.model.named_parameters():
            if 'next_state_head' in name:
                state_params.append(param)
        
        if state_params:
            param_groups.append({
                'params': state_params,
                'lr': self.lr * 1.5,  # Higher LR for state prediction
                'weight_decay': self.weight_decay * 0.5
            })
        
        # Reward heads
        reward_params = []
        for name, param in self.model.named_parameters():
            if 'reward_heads' in name:
                reward_params.append(param)
        
        if reward_params:
            param_groups.append({
                'params': reward_params,
                'lr': self.lr * 1.2,  # Higher LR for rewards
                'weight_decay': self.weight_decay * 0.1
            })
        
        # Other parameters
        other_params = []
        for name, param in self.model.named_parameters():
            if not any(component in name for component in ['transformer', 'next_state_head', 'reward_heads']):
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            })
        
        self.optimizer = torch.optim.AdamW(param_groups)
        
        # Learning rate scheduler
        scheduler_config = self.train_config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.epochs,
                eta_min=self.lr * 0.01
            )
        else:
            # Step scheduler
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, self.epochs // 3),
                gamma=0.5
            )
    
    def train(self, train_loader, test_loaders: Dict[str, Any]) -> str:
        """
        FIXED: Main training function for world model.
        
        Args:
            train_loader: Training data loader
            test_loaders: Dictionary of test data loaders {video_id: DataLoader}
            
        Returns:
            Path to the best saved model
        """
        
        self.logger.info("🌍 Starting World Model Training...")
        self.logger.info("🎯 Goal: Learn action-conditioned state-reward prediction")
        
        for epoch in range(self.epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # FIXED: Validation phase with dictionary of test loaders
            val_metrics = self._validate_epoch_fixed(test_loaders, epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, epoch)
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_path = os.path.join(
                    self.checkpoint_dir, f"world_model_best_epoch_{epoch+1}.pt"
                )
                self.model.save_model(self.best_model_path)
                self.logger.info(f"✅ New best model saved: {self.best_model_path}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"world_model_checkpoint_epoch_{epoch+1}.pt"
                )
                self.model.save_model(checkpoint_path)
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"State MSE: {val_metrics.get('state_loss', 0):.4f} | "
                f"Reward MSE: {val_metrics.get('total_reward_loss', 0):.4f}"
            )
        
        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, "world_model_final.pt")
        self.model.save_model(final_model_path)
        
        # Save training plots
        # self.save_training_plots()
        
        self.logger.info("✅ World Model Training completed!")
        self.logger.info(f"📄 Best model: {self.best_model_path}")
        self.logger.info(f"📄 Final model: {final_model_path}")
        
        return self.best_model_path if self.best_model_path else final_model_path
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                current_states = batch['current_states'].to(self.device)
                next_actions = batch['next_actions'].to(self.device)
                next_states = batch['next_states'].to(self.device)
                current_phases = batch['current_phases'].to(self.device)
                next_phases = batch['next_phases'].to(self.device)
                
                # Extract rewards
                target_rewards = {}
                for key, value in batch['rewards'].items():
                    target_rewards[key] = value.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    current_states=current_states,
                    next_actions=next_actions,
                    target_next_states=next_states,
                    target_rewards=target_rewards,
                    target_phases=next_phases
                )
                
                loss = outputs['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
                self.optimizer.step()
                
                # Accumulate metrics
                for key, value in outputs.items():
                    if key.endswith('loss') and isinstance(value, torch.Tensor):
                        epoch_metrics[key] += value.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'state': f"{outputs.get('state_loss', 0):.4f}",
                    'reward': f"{outputs.get('total_reward_loss', 0):.4f}"
                })
                
                # Log to tensorboard
                global_step = epoch * num_batches + batch_idx
                if batch_idx % self.log_every_n_steps == 0:
                    for key, value in outputs.items():
                        if key.endswith('loss') and isinstance(value, torch.Tensor):
                            self.tb_writer.add_scalar(f"train/{key}_batch", value.item(), global_step)
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            self.metrics_history[f"train_{key}"].append(epoch_metrics[key])
        
        return dict(epoch_metrics)
    
    def _validate_epoch_fixed(self, test_loaders: Dict[str, Any], epoch: int) -> Dict[str, float]:
        """FIXED: Validate for one epoch with dictionary of test loaders."""
        
        self.model.eval()
        all_video_metrics = []
        
        with torch.no_grad():
            # Iterate over each video's test loader
            for video_id, test_loader in test_loaders.items():
                video_metrics = defaultdict(float)
                num_batches = len(test_loader)
                
                for batch in tqdm(test_loader, desc=f"Validating {video_id}"):
                    # Move data to device
                    current_states = batch['current_states'].to(self.device)
                    next_actions = batch['next_actions'].to(self.device)
                    next_states = batch['next_states'].to(self.device)
                    current_phases = batch['current_phases'].to(self.device)
                    next_phases = batch['next_phases'].to(self.device)
                    
                    # Extract rewards
                    target_rewards = {}
                    for key, value in batch['rewards'].items():
                        target_rewards[key] = value.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        current_states=current_states,
                        next_actions=next_actions,
                        target_next_states=next_states,
                        target_rewards=target_rewards,
                        target_phases=next_phases
                    )
                    
                    # Accumulate metrics
                    for key, value in outputs.items():
                        if key.endswith('loss') and isinstance(value, torch.Tensor):
                            video_metrics[key] += value.item()
                
                # Average over batches for this video
                for key in video_metrics:
                    video_metrics[key] /= num_batches
                
                all_video_metrics.append(dict(video_metrics))
        
        # Average over all videos
        final_metrics = {}
        if all_video_metrics:
            for key in all_video_metrics[0].keys():
                values = [vm[key] for vm in all_video_metrics]
                final_metrics[key] = np.mean(values)
                self.metrics_history[f"val_{key}"].append(final_metrics[key])
        
        return final_metrics
    
    def evaluate_model(self, test_loaders: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Comprehensive evaluation of the trained world model.
        
        Args:
            test_loaders: Dictionary of test data loaders
            
        Returns:
            Detailed evaluation results
        """
        
        self.logger.info("📊 Evaluating World Model...")
        
        self.model.eval()
        
        # Standard metrics
        val_metrics = self._validate_epoch_fixed(test_loaders, epoch=0)
        
        # Simulation quality evaluation
        simulation_results = self._evaluate_simulation_quality(test_loaders)
        
        evaluation_results = {
            'overall_metrics': val_metrics,
            'simulation_quality': simulation_results,
            'model_type': 'ConditionalWorldModel',
            'evaluation_summary': {
                'best_metric': 'state_loss',
                'best_value': val_metrics.get('state_loss', 0.0),
                'strength': 'Action-conditioned state-reward prediction',
                'architecture': 'ConditionalWorldModel with action conditioning'
            }
        }
        
        self.logger.info(f"✅ Evaluation completed")
        self.logger.info(f"📊 State Loss: {val_metrics.get('state_loss', 0):.4f}")
        self.logger.info(f"📊 Reward Loss: {val_metrics.get('total_reward_loss', 0):.4f}")
        
        return evaluation_results
    
    def _evaluate_simulation_quality(self, test_loaders: Dict[str, Any]) -> Dict[str, float]:
        """FIXED: Evaluate world model simulation quality."""
        
        simulation_metrics = defaultdict(list)
        
        # Test simulation on a few samples from each video
        for video_id, test_loader in list(test_loaders.items())[:2]:  # Test on first 2 videos
            batch = next(iter(test_loader))
            current_states = batch['current_states'][:2].to(self.device)  # First 2 samples
            next_actions = batch['next_actions'][:2].to(self.device)
            
            # Test single step simulation
            for i in range(current_states.size(0)):
                state = current_states[i, 0]  # First timestep
                action = next_actions[i, 0]  # First action
                
                try:
                    next_state, rewards, _ = self.model.simulate_step(state, action)
                    
                    # Check if simulation produces reasonable outputs
                    if not torch.isnan(next_state).any():
                        simulation_metrics['successful_simulations'].append(1.0)
                    else:
                        simulation_metrics['successful_simulations'].append(0.0)
                    
                    # Check reward consistency
                    reward_count = len(rewards)
                    simulation_metrics['reward_types_predicted'].append(reward_count)
                    
                except Exception as e:
                    self.logger.warning(f"Simulation failed: {e}")
                    simulation_metrics['successful_simulations'].append(0.0)
                    simulation_metrics['reward_types_predicted'].append(0.0)
        
        # Average metrics
        return {key: np.mean(values) if values else 0.0 for key, values in simulation_metrics.items()}
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to tensorboard and history."""
        
        # Log to tensorboard
        for key, value in train_metrics.items():
            self.tb_writer.add_scalar(f"train/{key}_epoch", value, epoch)
        
        for key, value in val_metrics.items():
            self.tb_writer.add_scalar(f"val/{key}_epoch", value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.tb_writer.add_scalar("train/learning_rate", current_lr, epoch)
    
    def save_training_plots(self):
        """Save training history plots."""
        
        if not self.metrics_history:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss curves
        if 'train_total_loss' in self.metrics_history:
            axes[0, 0].plot(self.metrics_history['train_total_loss'], label='Train', color='blue')
            axes[0, 0].plot(self.metrics_history.get('val_total_loss', []), label='Val', color='red')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # State prediction loss
        if 'train_state_loss' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['train_state_loss'], label='Train', color='blue')
            axes[0, 1].plot(self.metrics_history.get('val_state_loss', []), label='Val', color='red')
            axes[0, 1].set_title('State Prediction Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Reward prediction loss
        if 'train_total_reward_loss' in self.metrics_history:
            axes[1, 0].plot(self.metrics_history['train_total_reward_loss'], label='Train', color='blue')
            axes[1, 0].plot(self.metrics_history.get('val_total_reward_loss', []), label='Val', color='red')
            axes[1, 0].set_title('Reward Prediction Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Phase prediction loss
        if 'train_phase_loss' in self.metrics_history:
            axes[1, 1].plot(self.metrics_history['train_phase_loss'], label='Train', color='blue')
            axes[1, 1].plot(self.metrics_history.get('val_phase_loss', []), label='Val', color='red')
            axes[1, 1].set_title('Phase Prediction Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'world_model_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        metrics_path = os.path.join(self.log_dir, 'world_model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
        
        self.logger.info(f"📊 Training plots saved to: {plot_path}")
        self.logger.info(f"📊 Metrics saved to: {metrics_path}")