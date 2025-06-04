#!/usr/bin/env python3
"""
Autoregressive IL Trainer for Method 1
Pure causal frame generation → action prediction training
"""

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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import json


class AutoregressiveILTrainer:
    """
    Trainer for Autoregressive Imitation Learning (Method 1).
    
    Focuses on:
    1. Causal frame generation (no action conditioning)
    2. Action prediction from generated frame representations
    3. Pure supervised learning on expert demonstrations
    """
    
    def __init__(self, 
                 model,
                 config: Dict[str, Any],
                 logger,
                 device: str = 'cuda'):
        """
        Initialize the autoregressive IL trainer.
        
        Args:
            model: AutoregressiveILModel instance
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
        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard', 'autoregressive_il')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Track metrics
        self.metrics_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        self.logger.info("🎓 Autoregressive IL Trainer initialized")
        self.logger.info(f"   Device: {device}")
        self.logger.info(f"   Epochs: {self.epochs}")
        self.logger.info(f"   Learning rate: {self.lr}")
        self.logger.info(f"   Focus: Causal frame generation → action prediction")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        
        # Different learning rates for different components
        param_groups = []
        
        # GPT-2 backbone (lower learning rate for pre-trained-like component)
        gpt2_params = []
        for name, param in self.model.named_parameters():
            if 'gpt2' in name:
                gpt2_params.append(param)
        
        if gpt2_params:
            param_groups.append({
                'params': gpt2_params,
                'lr': self.lr * 0.1,  # Lower LR for backbone
                'weight_decay': self.weight_decay
            })
        
        # Frame generation head (important for IL)
        frame_params = []
        for name, param in self.model.named_parameters():
            if 'next_frame_head' in name:
                frame_params.append(param)
        
        if frame_params:
            param_groups.append({
                'params': frame_params,
                'lr': self.lr * 1.5,  # Higher LR for frame generation
                'weight_decay': self.weight_decay * 0.5
            })
        
        # Action prediction head (most important for evaluation)
        action_params = []
        for name, param in self.model.named_parameters():
            if 'action_prediction_head' in name:
                action_params.append(param)
        
        if action_params:
            param_groups.append({
                'params': action_params,
                'lr': self.lr * 2.0,  # Highest LR for action prediction
                'weight_decay': self.weight_decay * 0.1
            })
        
        # Other parameters (projection layers, etc.)
        other_params = []
        for name, param in self.model.named_parameters():
            if not any(component in name for component in ['gpt2', 'next_frame_head', 'action_prediction_head']):
                other_params.append(param)
        
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
            # Step scheduler
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, self.epochs // 3),
                gamma=0.5
            )
    
    def train(self, 
              train_loader: DataLoader, 
              test_loaders: Dict[str, DataLoader]) -> str:
        """
        Main training function for autoregressive IL.
        
        Args:
            train_loader: Training data loader
            test_loaders: Dictionary of test data loaders
            
        Returns:
            Path to the best saved model
        """
        
        self.logger.info("🎓 Starting Autoregressive IL Training...")
        self.logger.info("🎯 Goal: Learn causal frame generation → action prediction")
        
        for epoch in range(self.epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(test_loaders, epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, epoch)
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_path = os.path.join(
                    self.checkpoint_dir, f"autoregressive_il_best_epoch_{epoch+1}.pt"
                )
                self.model.save_model(self.best_model_path)
                self.logger.info(f"✅ New best model saved: {self.best_model_path}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"autoregressive_il_checkpoint_epoch_{epoch+1}.pt"
                )
                self.model.save_model(checkpoint_path)
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Action mAP: {val_metrics.get('action_mAP', 0):.4f} | "
                f"Frame MSE: {val_metrics.get('frame_loss', 0):.4f}"
            )
        
        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, "autoregressive_il_final.pt")
        self.model.save_model(final_model_path)
        
        # Save training plots
        self.save_training_plots()
        
        self.logger.info("✅ Autoregressive IL Training completed!")
        self.logger.info(f"📄 Best model: {self.best_model_path}")
        self.logger.info(f"📄 Final model: {final_model_path}")
        
        return self.best_model_path if self.best_model_path else final_model_path
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                input_frames = batch['input_frames'].to(self.device)
                target_next_frames = batch['target_next_frames'].to(self.device)
                target_actions = batch['target_actions'].to(self.device)
                target_phases = batch['target_phases'].to(self.device)
                
                # Forward pass (no action conditioning!)
                outputs = self.model(
                    frame_embeddings=input_frames,
                    target_next_frames=target_next_frames,
                    target_actions=target_actions,
                    target_phases=target_phases
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
                    'frame': f"{outputs.get('frame_loss', 0):.4f}",
                    'action': f"{outputs.get('action_loss', 0):.4f}"
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
    
    def _validate_epoch(self, test_loaders: Dict[str, DataLoader], epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for video_id, test_loader in test_loaders.items():
                video_metrics = defaultdict(float)
                num_batches = len(test_loader)
                
                for batch in test_loader:
                    # Move data to device
                    input_frames = batch['input_frames'].to(self.device)
                    target_next_frames = batch['target_next_frames'].to(self.device)
                    target_actions = batch['target_actions'].to(self.device)
                    target_phases = batch['target_phases'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        frame_embeddings=input_frames,
                        target_next_frames=target_next_frames,
                        target_actions=target_actions,
                        target_phases=target_phases
                    )
                    
                    # Accumulate losses
                    for key, value in outputs.items():
                        if key.endswith('loss') and isinstance(value, torch.Tensor):
                            video_metrics[key] += value.item()
                
                # Average over batches for this video
                for key in video_metrics:
                    video_metrics[key] /= num_batches
                    all_metrics[key].append(video_metrics[key])
        
        # Calculate comprehensive action prediction metrics
        action_metrics = self._evaluate_action_prediction(test_loaders)
        all_metrics.update(action_metrics)
        
        # Average over all videos
        final_metrics = {}
        for key, values in all_metrics.items():
            final_metrics[key] = np.mean(values) if values else 0.0
            self.metrics_history[f"val_{key}"].append(final_metrics[key])
        
        return final_metrics
    
    def _evaluate_action_prediction(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """Evaluate action prediction performance with comprehensive metrics."""
        
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for video_id, test_loader in test_loaders.items():
                for batch in test_loader:
                    input_frames = batch['input_frames'].to(self.device)
                    target_actions = batch['target_actions'].to(self.device)
                    
                    # Get action predictions
                    outputs = self.model(frame_embeddings=input_frames)
                    action_probs = outputs['action_pred']
                    
                    all_predictions.append(action_probs.cpu().numpy())
                    all_targets.append(target_actions.cpu().numpy())
        
        if not all_predictions:
            return {'action_mAP': 0.0, 'action_accuracy': 0.0}
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten for metric calculation
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        binary_preds = (pred_flat > 0.5).astype(int)
        
        # Calculate mAP (most important metric for IL)
        ap_scores = []
        for i in range(target_flat.shape[1]):
            if np.sum(target_flat[:, i]) > 0:
                try:
                    ap = average_precision_score(target_flat[:, i], pred_flat[:, i])
                    ap_scores.append(ap)
                except:
                    ap_scores.append(0.0)
            else:
                # No positive examples for this action
                ap_scores.append(1.0 if np.sum(binary_preds[:, i]) == 0 else 0.0)
        
        mAP = np.mean(ap_scores) if ap_scores else 0.0
        
        # Calculate other metrics
        exact_match = np.mean(np.all(binary_preds == target_flat, axis=1))
        hamming_acc = np.mean(binary_preds == target_flat)
        
        # Top-k accuracy
        top_k_accs = {}
        for k in [1, 3, 5]:
            top_k_acc_list = []
            for i in range(pred_flat.shape[0]):
                target_indices = np.where(target_flat[i] > 0.5)[0]
                if len(target_indices) > 0:
                    top_k_pred = np.argsort(pred_flat[i])[-k:]
                    hit = len(np.intersect1d(target_indices, top_k_pred)) > 0
                    top_k_acc_list.append(hit)
            top_k_accs[f'top_{k}_accuracy'] = np.mean(top_k_acc_list) if top_k_acc_list else 0.0
        
        # Precision, Recall, F1
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_flat.flatten(), binary_preds.flatten(), average='binary', zero_division=0
            )
        except:
            precision = recall = f1 = 0.0
        
        return {
            'action_mAP': mAP,
            'action_exact_match': exact_match,
            'action_hamming_accuracy': hamming_acc,
            'action_precision': precision,
            'action_recall': recall,
            'action_f1': f1,
            **top_k_accs
        }
    
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
    
    def evaluate_model(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the trained model.
        
        Args:
            test_loaders: Dictionary of test data loaders
            
        Returns:
            Detailed evaluation results
        """
        
        self.logger.info("📊 Evaluating Autoregressive IL Model...")
        
        self.model.eval()
        
        # Standard metrics
        val_metrics = self._validate_epoch(test_loaders, epoch=0)
        
        # Generation evaluation
        generation_results = self._evaluate_generation_quality(test_loaders)
        
        # Video-level evaluation
        video_results = {}
        for video_id, test_loader in test_loaders.items():
            video_metrics = self._evaluate_single_video(test_loader, video_id)
            video_results[video_id] = video_metrics
        
        evaluation_results = {
            'overall_metrics': val_metrics,
            'generation_quality': generation_results,
            'video_results': video_results,
            'model_type': 'AutoregressiveIL',
            'evaluation_summary': {
                'best_metric': 'action_mAP',
                'best_value': val_metrics.get('action_mAP', 0.0),
                'strength': 'Causal frame generation and action prediction',
                'architecture': 'Autoregressive (no action conditioning)'
            }
        }
        
        self.logger.info(f"✅ Evaluation completed")
        self.logger.info(f"📊 Action mAP: {val_metrics.get('action_mAP', 0):.4f}")
        self.logger.info(f"📊 Frame MSE: {val_metrics.get('frame_loss', 0):.4f}")
        self.logger.info(f"📊 Exact Match: {val_metrics.get('action_exact_match', 0):.4f}")
        
        return evaluation_results
    
    def _evaluate_generation_quality(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """Evaluate autoregressive generation quality."""
        
        generation_metrics = defaultdict(list)
        
        # Test generation on a few samples
        for video_id, test_loader in list(test_loaders.items())[:3]:  # Test on 3 videos
            batch = next(iter(test_loader))
            input_frames = batch['input_frames'][:2].to(self.device)  # First 2 samples
            
            # Generate sequences
            generation_results = self.model.generate_sequence(
                initial_frames=input_frames,
                horizon=10,
                temperature=0.8
            )
            
            # Calculate generation quality metrics
            generated_frames = generation_results['generated_frames']
            predicted_actions = generation_results['predicted_actions']
            
            # Frame generation consistency (smoothness)
            frame_diffs = torch.diff(generated_frames, dim=1)
            frame_smoothness = torch.mean(torch.norm(frame_diffs, dim=-1))
            generation_metrics['frame_smoothness'].append(frame_smoothness.item())
            
            # Action prediction diversity
            action_diversity = torch.mean(torch.std(predicted_actions, dim=1))
            generation_metrics['action_diversity'].append(action_diversity.item())
        
        # Average metrics
        return {key: np.mean(values) for key, values in generation_metrics.items()}
    
    def _evaluate_single_video(self, test_loader: DataLoader, video_id: str) -> Dict[str, float]:
        """Evaluate model on a single video."""
        
        video_predictions = []
        video_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_frames = batch['input_frames'].to(self.device)
                target_actions = batch['target_actions'].to(self.device)
                
                outputs = self.model(frame_embeddings=input_frames)
                action_probs = outputs['action_pred']
                
                video_predictions.append(action_probs.cpu().numpy())
                video_targets.append(target_actions.cpu().numpy())
        
        if not video_predictions:
            return {'mAP': 0.0}
        
        # Calculate video-specific metrics
        predictions = np.concatenate(video_predictions, axis=0)
        targets = np.concatenate(video_targets, axis=0)
        
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        
        # mAP for this video
        ap_scores = []
        for i in range(target_flat.shape[1]):
            if np.sum(target_flat[:, i]) > 0:
                try:
                    ap = average_precision_score(target_flat[:, i], pred_flat[:, i])
                    ap_scores.append(ap)
                except:
                    ap_scores.append(0.0)
        
        video_mAP = np.mean(ap_scores) if ap_scores else 0.0
        
        return {
            'mAP': video_mAP,
            'num_frames': len(pred_flat),
            'avg_action_density': np.mean(np.sum(target_flat, axis=1))
        }
    
    def save_training_plots(self):
        """Save training history plots."""
        
        if not self.metrics_history:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        if 'train_total_loss' in self.metrics_history:
            axes[0, 0].plot(self.metrics_history['train_total_loss'], label='Train', color='blue')
            axes[0, 0].plot(self.metrics_history.get('val_total_loss', []), label='Val', color='red')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Frame generation loss
        if 'train_frame_loss' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['train_frame_loss'], label='Train', color='blue')
            axes[0, 1].plot(self.metrics_history.get('val_frame_loss', []), label='Val', color='red')
            axes[0, 1].set_title('Frame Generation Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Action prediction loss
        if 'train_action_loss' in self.metrics_history:
            axes[0, 2].plot(self.metrics_history['train_action_loss'], label='Train', color='blue')
            axes[0, 2].plot(self.metrics_history.get('val_action_loss', []), label='Val', color='red')
            axes[0, 2].set_title('Action Prediction Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Action mAP
        if 'val_action_mAP' in self.metrics_history:
            axes[1, 0].plot(self.metrics_history['val_action_mAP'], color='green')
            axes[1, 0].set_title('Action Prediction mAP')
            axes[1, 0].grid(True)
        
        # Exact match accuracy
        if 'val_action_exact_match' in self.metrics_history:
            axes[1, 1].plot(self.metrics_history['val_action_exact_match'], color='orange')
            axes[1, 1].set_title('Exact Match Accuracy')
            axes[1, 1].grid(True)
        
        # Top-1 accuracy
        if 'val_top_1_accuracy' in self.metrics_history:
            axes[1, 2].plot(self.metrics_history['val_top_1_accuracy'], color='purple')
            axes[1, 2].set_title('Top-1 Accuracy')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'autoregressive_il_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        metrics_path = os.path.join(self.log_dir, 'autoregressive_il_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
        
        self.logger.info(f"📊 Training plots saved to: {plot_path}")
        self.logger.info(f"📊 Metrics saved to: {metrics_path}")


# Example usage and testing
if __name__ == "__main__":
    print("🎓 AUTOREGRESSIVE IL TRAINER")
    print("=" * 60)
    
    # This would be called from the main experiment script
    print("✅ Trainer ready for Method 1 (Autoregressive IL)")
    print("🎯 Focus: Causal frame generation → action prediction")
    print("📋 Key features:")
    print("   • No action conditioning during training")
    print("   • Comprehensive action prediction metrics")
    print("   • Generation quality evaluation")
    print("   • Video-level analysis")
    print("   • Optimized for IL performance")
