import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple

class DualTrainer:
    """
    Trainer that supports both supervised learning for autoregressive action prediction
    and RL-based training for state prediction.
    """
    
    def __init__(self, 
                 model,
                 config: Dict[str, Any],
                 logger,
                 device: str = 'cuda'):
        """
        Initialize the dual trainer.
        
        Args:
            model: DualWorldModel instance
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
        
        # Setup logging
        self.log_dir = logger.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Tensorboard
        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Training mode
        self.training_mode = config.get('training_mode', 'supervised')  # 'supervised', 'rl', or 'mixed'
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Track metrics
        self.metrics_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Different learning rates for different components
        param_groups = []
        
        # GPT-2 backbone (lower learning rate)
        gpt2_params = []
        for name, param in self.model.named_parameters():
            if 'gpt2' in name:
                gpt2_params.append(param)
        
        if gpt2_params:
            param_groups.append({
                'params': gpt2_params,
                'lr': self.lr * 0.1,  # Lower LR for pre-trained backbone
                'weight_decay': self.weight_decay
            })
        
        # Prediction heads (normal learning rate)
        head_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'heads' in name:
                head_params.append(param)
            elif 'gpt2' not in name:  # Projection layers, etc.
                other_params.append(param)
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay * 0.1  # Less regularization for heads
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            })
        
        self.optimizer = optim.AdamW(param_groups)
        
        # Learning rate scheduler
        scheduler_config = self.train_config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.epochs,
                eta_min=self.lr * 0.01
            )
        else:
            # Warmup + linear decay
            warmup_steps = scheduler_config.get('warmup_steps', 1000)
            total_steps = self.epochs * 100  # Approximate
            self.scheduler = self._get_linear_schedule_with_warmup(warmup_steps, total_steps)
    
    def _get_linear_schedule_with_warmup(self, num_warmup_steps: int, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, 
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_supervised_mode(self, 
                            train_loader: DataLoader, 
                            val_loaders: Dict[str, DataLoader]) -> str:
        """
        Train the model in supervised mode for autoregressive action prediction.
        
        Args:
            train_loader: Training data loader
            val_loaders: Validation data loaders
            
        Returns:
            Path to the best saved model
        """
        self.logger.info("Starting supervised training for autoregressive action prediction")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = defaultdict(float)
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                # Move batch to device
                current_states = batch['current_states'].to(self.device)
                next_states = batch['next_states'].to(self.device)
                next_actions = batch['next_actions'].to(self.device)
                next_phases = batch.get('next_phases', None)
                if next_phases is not None:
                    next_phases = next_phases.to(self.device)
                
                # Create action sequence (shift by one for autoregressive training)
                # Use current actions as input, predict next actions
                current_actions = batch.get('current_actions', None)
                if current_actions is not None:
                    current_actions = current_actions.to(self.device)
                    # Expand to match sequence length
                    if current_actions.dim() == 2:  # [batch_size, num_actions]
                        current_actions = current_actions.unsqueeze(1).expand(-1, current_states.size(1), -1)
                
                # Forward pass
                outputs = self.model(
                    current_states=current_states,
                    actions=current_actions,
                    next_states=next_states,
                    next_actions=next_actions,
                    next_phases=next_phases,
                    mode='supervised'
                )
                
                loss = outputs['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Log losses
                for key, value in outputs.items():
                    if key.endswith('loss') and isinstance(value, torch.Tensor):
                        train_losses[key] += value.item()
                        
                        # Log to tensorboard per batch
                        self.tb_writer.add_scalar(
                            f"train/{key}_per_batch", 
                            value.item(), 
                            epoch * len(train_loader) + batch_idx
                        )
                
                # Periodic logging
                if batch_idx % self.log_every_n_steps == 0:
                    self.logger.info(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # Calculate average losses
            for key in train_losses:
                train_losses[key] /= len(train_loader)
                self.metrics_history[f"train_{key}"].append(train_losses[key])
                self.tb_writer.add_scalar(f"train/{key}_per_epoch", train_losses[key], epoch)
            
            # Validation
            val_metrics = self._evaluate_supervised(val_loaders)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                self.metrics_history[f"val_{key}"].append(value)
                self.tb_writer.add_scalar(f"val/{key}_per_epoch", value, epoch)
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_path = os.path.join(
                    self.checkpoint_dir, f"supervised_best_epoch_{epoch+1}.pt"
                )
                self.model.save_model(self.best_model_path)
                self.logger.info(f"New best model saved: {self.best_model_path}")
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_losses['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val Action Acc: {val_metrics.get('action_accuracy', 0):.4f}"
            )
        
        return self.best_model_path
    
    def train_rl_mode(self, 
                     train_loader: DataLoader, 
                     val_loaders: Dict[str, DataLoader]) -> str:
        """
        Train the model in RL mode for state and reward prediction.
        
        Args:
            train_loader: Training data loader
            val_loaders: Validation data loaders
            
        Returns:
            Path to the best saved model
        """
        self.logger.info("Starting RL training for state and reward prediction")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = defaultdict(float)
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                # Move batch to device
                current_states = batch['current_states'].to(self.device)
                next_states = batch['next_states'].to(self.device)
                next_actions = batch['next_actions'].to(self.device)
                next_rewards = batch.get('next_rewards', {})
                next_rewards = {k: v.to(self.device) for k, v in next_rewards.items()}
                
                # Forward pass
                outputs = self.model(
                    current_states=current_states,
                    actions=next_actions,
                    next_states=next_states,
                    next_rewards=next_rewards,
                    mode='rl'
                )
                
                loss = outputs['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Log losses
                for key, value in outputs.items():
                    if key.endswith('loss') and isinstance(value, torch.Tensor):
                        train_losses[key] += value.item()
                        
                        # Log to tensorboard per batch
                        self.tb_writer.add_scalar(
                            f"train/{key}_per_batch", 
                            value.item(), 
                            epoch * len(train_loader) + batch_idx
                        )
                
                # Periodic logging
                if batch_idx % self.log_every_n_steps == 0:
                    self.logger.info(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # Calculate average losses
            for key in train_losses:
                train_losses[key] /= len(train_loader)
                self.metrics_history[f"train_{key}"].append(train_losses[key])
                self.tb_writer.add_scalar(f"train/{key}_per_epoch", train_losses[key], epoch)
            
            # Validation
            val_metrics = self._evaluate_rl(val_loaders)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                self.metrics_history[f"val_{key}"].append(value)
                self.tb_writer.add_scalar(f"val/{key}_per_epoch", value, epoch)
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_path = os.path.join(
                    self.checkpoint_dir, f"rl_best_epoch_{epoch+1}.pt"
                )
                self.model.save_model(self.best_model_path)
                self.logger.info(f"New best model saved: {self.best_model_path}")
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_losses['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val State MSE: {val_metrics.get('state_mse', 0):.4f}"
            )
        
        return self.best_model_path
    
    def _evaluate_supervised(self, val_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """Evaluate model in supervised mode with proper metrics."""
        self.model.eval()
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for video_id, val_loader in val_loaders.items():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Evaluating {video_id} ...")):
                    current_states = batch['current_states'].to(self.device)
                    next_states = batch['next_states'].to(self.device)
                    next_actions = batch['next_actions'].to(self.device)
                    
                    outputs = self.model(
                        current_states=current_states,
                        next_states=next_states,
                        next_actions=next_actions,
                        mode='supervised'
                    )
                    
                    # Calculate proper metrics
                    if 'action_pred' in outputs:
                        batch_metrics = calculate_proper_validation_metrics(
                            outputs['action_pred'], next_actions
                        )
                        
                        for key, value in batch_metrics.items():
                            all_metrics[key].append(value)
                    
                    # Other losses
                    for key, value in outputs.items():
                        if key.endswith('loss') and isinstance(value, torch.Tensor):
                            all_metrics[key].append(value.item())
        
        # Average all metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            final_metrics[key] = np.mean(values) if values else 0.0
        
        return final_metrics
    
    def _evaluate_rl(self, val_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """Evaluate model in RL mode."""
        self.model.eval()
        metrics = defaultdict(float)
        total_samples = 0
        
        with torch.no_grad():
            for video_id, val_loader in val_loaders.items():
                video_metrics = defaultdict(float)
                num_batches = 0
                
                for batch in val_loader:
                    # Move batch to device
                    current_states = batch['current_states'].to(self.device)
                    next_states = batch['next_states'].to(self.device)
                    next_actions = batch['next_actions'].to(self.device)
                    next_rewards = batch.get('next_rewards', {})
                    next_rewards = {k: v.to(self.device) for k, v in next_rewards.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        current_states=current_states,
                        actions=next_actions,
                        next_states=next_states,
                        next_rewards=next_rewards,
                        mode='rl'
                    )
                    
                    # Accumulate losses
                    for key, value in outputs.items():
                        if key.endswith('loss') and isinstance(value, torch.Tensor):
                            video_metrics[key] += value.item()
                    
                    # Calculate state MSE
                    if 'state_pred' in outputs:
                        state_mse = torch.mean((outputs['state_pred'] - next_states) ** 2).item()
                        video_metrics['state_mse'] += state_mse
                    
                    num_batches += 1
                
                # Average over batches
                for key in video_metrics:
                    video_metrics[key] /= num_batches
                    metrics[key] += video_metrics[key]
                
                total_samples += 1
        
        # Average over videos
        for key in metrics:
            metrics[key] /= max(total_samples, 1)
        
        return dict(metrics)
    
    def train(self, 
              train_loader: DataLoader, 
              val_loaders: Dict[str, DataLoader]) -> str:
        """
        Main training function that dispatches to appropriate training mode.
        
        Args:
            train_loader: Training data loader
            val_loaders: Validation data loaders
            
        Returns:
            Path to the best saved model
        """
        if self.training_mode == 'supervised':
            return self.train_supervised_mode(train_loader, val_loaders)
        elif self.training_mode == 'rl':
            return self.train_rl_mode(train_loader, val_loaders)
        elif self.training_mode == 'mixed':
            # First train in supervised mode, then fine-tune in RL mode
            self.logger.info("Starting mixed training: supervised first, then RL")
            
            # Supervised training
            self.epochs = self.epochs // 2  # Split epochs between modes
            supervised_path = self.train_supervised_mode(train_loader, val_loaders)
            
            # Reset for RL training
            self.best_val_loss = float('inf')
            self.best_model_path = None
            
            # RL training
            rl_path = self.train_rl_mode(train_loader, val_loaders)
            
            return rl_path
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def save_training_plots(self):
        """Save training history plots."""
        if not self.metrics_history:
            return
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if 'train_total_loss' in self.metrics_history:
            axes[0, 0].plot(self.metrics_history['train_total_loss'], label='Train')
            axes[0, 0].plot(self.metrics_history.get('val_total_loss', []), label='Val')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # State loss
        if 'train_state_loss' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['train_state_loss'], label='Train')
            axes[0, 1].plot(self.metrics_history.get('val_state_loss', []), label='Val')
            axes[0, 1].set_title('State Prediction Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Action accuracy (if available)
        if 'val_action_accuracy' in self.metrics_history:
            axes[1, 0].plot(self.metrics_history['val_action_accuracy'])
            axes[1, 0].set_title('Action Prediction Accuracy')
            axes[1, 0].grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()
        
        self.logger.info(f"Training plots saved to {self.log_dir}")

def calculate_proper_validation_metrics(action_pred, next_actions):
    """Calculate proper validation metrics for multi-label classification."""
    
    action_probs = torch.sigmoid(action_pred)
    action_pred_binary = (action_probs > 0.5).float()
    
    metrics = {}
    
    # 1. Element-wise accuracy (the misleading 99% metric)
    element_wise_acc = (action_pred_binary == next_actions).float().mean().item()
    metrics['element_wise_accuracy'] = element_wise_acc
    
    # 2. Exact match accuracy (much more meaningful)
    exact_match = (action_pred_binary == next_actions).all(dim=-1).float().mean().item()
    metrics['exact_match_accuracy'] = exact_match
    
    # 3. Hamming accuracy (accounts for class imbalance)
    hamming_acc = (action_pred_binary == next_actions).float().sum(dim=-1) / next_actions.size(-1)
    metrics['hamming_accuracy'] = hamming_acc.mean().item()
    
    # 4. Active action accuracy (only for active actions)
    active_mask = next_actions > 0.5
    if active_mask.sum() > 0:
        active_correct = (action_pred_binary[active_mask] == next_actions[active_mask]).float().mean()
        metrics['active_action_accuracy'] = active_correct.item()
    else:
        metrics['active_action_accuracy'] = 0.0
    
    # 5. Precision, Recall, F1 for active actions
    if active_mask.sum() > 0:
        predicted_active = action_pred_binary > 0.5
        
        # True positives, false positives, false negatives
        tp = (predicted_active & (next_actions > 0.5)).float().sum()
        fp = (predicted_active & (next_actions <= 0.5)).float().sum()
        fn = (~predicted_active & (next_actions > 0.5)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics['precision'] = precision.item()
        metrics['recall'] = recall.item()
        metrics['f1_score'] = f1.item()
    
    # 6. Mean Average Precision (the real metric)
    try:
        from sklearn.metrics import average_precision_score
        ap_scores = []
        for i in range(next_actions.size(-1)):
            if next_actions[:, i].sum() > 0:  # Only for classes that appear
                y_true = next_actions[:, i].cpu().numpy()
                y_scores = action_probs[:, i].cpu().numpy()
                ap = average_precision_score(y_true, y_scores)
                ap_scores.append(ap)
        
        metrics['mean_average_precision'] = np.mean(ap_scores) if ap_scores else 0.0
    except:
        metrics['mean_average_precision'] = 0.0
    
    return metrics

def train_dual_world_model(cfg, logger, model, train_loader, test_video_loaders, device='cuda'):
    """
    Main training function for the dual world model.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        model: DualWorldModel instance
        train_loader: Training data loader
        test_video_loaders: Test video loaders
        device: Device to train on
        
    Returns:
        Path to the best saved model
    """
    # Create trainer
    trainer = DualTrainer(model, cfg, logger, device)
    
    # Train the model
    best_model_path = trainer.train(train_loader, test_video_loaders)
    
    # Save training plots
    trainer.save_training_plots()
    
    # Save final model
    final_model_path = os.path.join(trainer.checkpoint_dir, "final_model.pt")
    model.save_model(final_model_path)
    
    logger.info(f"Training completed. Best model: {best_model_path}")
    logger.info(f"Final model: {final_model_path}")
    
    return best_model_path