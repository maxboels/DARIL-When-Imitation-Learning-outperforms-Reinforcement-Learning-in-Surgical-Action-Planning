#!/usr/bin/env python3
"""
ENHANCED: Autoregressive IL Trainer with Threshold-Based IVT Model Saving
- Saves best models based on IVT current mAP (recognition task) only if mAP ≥ 0.30
- Saves best models based on IVT next mAP (next action prediction task) only if mAP ≥ 0.30
- Saves best combined IVT performance only if combined score ≥ 0.30
- Prevents disk bloat by not saving poor-performing models
- Provides fallback to final model if no thresholds are met
- Comprehensive tracking and logging with threshold progress
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

from evaluation.evaluation_metrics import calculate_comprehensive_action_metrics, surgical_metrics
from evaluation.planning_evaluation_ivt import AutoregressivePlanningEvaluator, add_planning_evaluation_to_trainer

from utils.optimizer_scheduler import OptimizerScheduler

try:
    import ivtmetrics
    IVT_AVAILABLE = True
except ImportError:
    IVT_AVAILABLE = False
    print("⚠️  ivtmetrics not available. Install with: pip install ivtmetrics")


class AutoregressiveILTrainer:
    """
    ENHANCED: Trainer for Autoregressive Imitation Learning with IVT-based model saving.
    
    Features:
    - Saves best models based on IVT current mAP (recognition task)
    - Saves best models based on IVT next mAP (next action prediction task)
    - Saves best combined IVT performance model
    - Comprehensive IVT metrics tracking and logging
    """
    
    def __init__(self, model, config: Dict[str, Any], logger, device: str = 'cuda'):
        """Initialize with enhanced IVT-based model saving."""
        
        # Your existing initialization code...
        self.model = model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Enhanced optimizer will be set up in _setup_optimizer
        self.optimizer_scheduler = None
        
        # Training configuration
        self.train_config = config['training']
        self.epochs = self.train_config['epochs']
        self.lr = self.train_config['learning_rate']
        self.weight_decay = self.train_config['weight_decay']
        self.clip_grad = self.train_config.get('gradient_clip_val', 1.0)
        self.log_every_n_steps = self.train_config.get('log_every_n_steps', 100)
        
        # Defaults for planning horizons and frames per second
        self.fps = 1 # Default FPS for CholecT50 dataset
        self.context_length = self.config.get('data', {}).get('context_length', 20)
        planning_horizons = self.config.get('planning_evaluation', {}).get('planning_horizons', [1, 2, 3, 5, 10, 20])
        self.planning_horizons = {f"{h}s": h * self.fps for h in planning_horizons} 
        self.fps = 1

        # Setup logging directories
        self.log_dir = logger.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Tensorboard
        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard', 'autoregressive_il')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        # Visualization
        self.visualization_dir = os.path.join(self.log_dir, 'visualization', 'autoregressive_il')
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Track metrics
        self.metrics_history = defaultdict(list)
        
        # ENHANCED: Track best IVT metrics separately
        self.best_ivt_current_map = 0.0  # Best current action recognition mAP
        self.best_ivt_next_map = 0.0     # Best next action prediction mAP
        self.best_combined_ivt_score = 0.0  # Best combined score
        
        # ENHANCED: Minimum thresholds for saving models (avoid saving too many heavy files)
        self.min_current_map_threshold = config.get('training', {}).get('min_current_map_threshold', 0.30)
        self.min_next_map_threshold = config.get('training', {}).get('min_next_map_threshold', 0.30)
        self.min_combined_threshold = config.get('training', {}).get('min_combined_threshold', 0.30)
        
        # ENHANCED: Track best model paths for different metrics
        self.best_current_model_path = None
        self.best_next_model_path = None
        self.best_combined_model_path = None
        
        # Keep legacy for backward compatibility
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Per-video tracking
        self.video_performance_history = defaultdict(list)
        self.last_validation_details = {}
        
        # Set logger for shared metrics
        surgical_metrics.logger = logger
        
        self.logger.info("🎓 ENHANCED Autoregressive IL Trainer initialized with IVT-based saving")
        self.logger.info(f"   Device: {device}")
        self.logger.info(f"   Epochs: {self.epochs}")
        self.logger.info(f"   Learning rate: {self.lr}")
        self.logger.info(f"   ✅ IVT-based model saving enabled with minimum thresholds")
        self.logger.info(f"   📊 Will track: Current mAP, Next mAP, Combined score")
        self.logger.info(f"   🎯 Minimum thresholds: Current={self.min_current_map_threshold:.2f}, Next={self.min_next_map_threshold:.2f}, Combined={self.min_combined_threshold:.2f}")

    def _setup_optimizer(self):
        """Enhanced optimizer setup with intelligent parameter grouping."""
        
        # Create enhanced optimizer-scheduler
        self.optimizer_scheduler = OptimizerScheduler(
            model=self.model,
            config=self.train_config,
            total_epochs=self.epochs,
            steps_per_epoch=len(self.train_loader) if hasattr(self, 'train_loader') else 100,
            logger=self.logger
        )
        
        # Replace optimizer and scheduler
        self.optimizer = self.optimizer_scheduler.optimizer
        self.scheduler = self.optimizer_scheduler.scheduler
        
        self.logger.info("✅ Enhanced optimizer and scheduler integrated")

    def _enhanced_validation_step(self, val_metrics, epoch):
        """Enhanced validation with scheduler stepping."""
        
        val_loss = val_metrics['total_loss']
        
        # Step scheduler for epoch-based schedulers
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.optimizer_scheduler.step(validation_loss=val_loss, epoch=epoch)
        
        # Log current learning rates
        current_lrs = self.optimizer_scheduler.get_current_lrs()
        lr_summary = ", ".join([f"{name}: {lr:.2e}" for name, lr in current_lrs.items()])
        self.logger.info(f"📊 Current LRs: {lr_summary}")
        
        # Get optimization recommendations periodically
        if epoch % 5 == 0:
            recommendations = self.optimizer_scheduler.get_optimization_recommendations()
            for rec in recommendations:
                self.logger.info(f"💡 Optimization tip: {rec}")

    
    def train(self, train_loader: DataLoader, test_loaders: Dict[str, DataLoader]) -> str:
        """Main training function with enhanced IVT-based model saving."""
        
        self.train_loader = train_loader  # Store for optimizer setup
        
        self.logger.info("🎓 Starting ENHANCED Autoregressive IL Training with threshold-based IVT saving...")
        self.logger.info(f"   Minimum thresholds: Current={self.min_current_map_threshold:.2f}, Next={self.min_next_map_threshold:.2f}, Combined={self.min_combined_threshold:.2f}")
        
        for epoch in range(self.epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch_per_video_with_planning(test_loaders, epoch)
            
            # ENHANCED: Validation step with scheduler
            self._enhanced_validation_step(val_metrics, epoch)
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, epoch)
            
            # ENHANCED: Save best models based on IVT metrics
            self._save_best_models_based_on_ivt(val_metrics, epoch)
            
            # Enhanced epoch summary with IVT metrics
            current_lrs = self.optimizer_scheduler.get_current_lrs()
            main_lr = current_lrs.get('action_prediction', 0)
            
            # Get IVT metrics for logging
            current_ivt = val_metrics.get('ivt_current_mAP', 0)
            next_ivt = val_metrics.get('ivt_next_mAP', 0)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Current IVT: {current_ivt:.4f} | "
                f"Next IVT: {next_ivt:.4f} | "
                f"Action LR: {main_lr:.2e}"
            )
        
        # ENHANCED: Training completion with comprehensive analysis
        self._enhanced_training_complete()
        
        # ENHANCED: Handle case where no models exceed thresholds
        best_return_path = None
        
        # Priority order: combined > next > current > fallback
        if self.best_combined_model_path:
            best_return_path = self.best_combined_model_path
            self.logger.info(f"✅ Returning best combined model: {best_return_path}")
        elif self.best_next_model_path:
            best_return_path = self.best_next_model_path
            self.logger.info(f"✅ Returning best next prediction model: {best_return_path}")
        elif self.best_current_model_path:
            best_return_path = self.best_current_model_path
            self.logger.info(f"✅ Returning best current recognition model: {best_return_path}")
        else:
            # FALLBACK: Save final model if no thresholds were met
            fallback_path = os.path.join(
                self.checkpoint_dir, f"autoregressive_il_final_epoch_{self.epochs}.pt"
            )
            self.model.save_model(fallback_path)
            best_return_path = fallback_path
            self.logger.warning(f"⚠️  No models exceeded thresholds, saving final model: {fallback_path}")
            self.logger.info(f"   Final performance: Current={self.best_ivt_current_map:.4f}, Next={self.best_ivt_next_map:.4f}")
            
            # Update legacy path for compatibility
            self.best_model_path = fallback_path
        
        return best_return_path

    def _prepare_training_data(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare training data with proper temporal alignment for dual-path architecture.
        
        Your approach: sequences include both current and next tokens
        - Recognition: input_frames[t] → current_actions[t] (BiLSTM path)
        - Generation: input_frames[t] → next_frames[t+1], next_actions[t+1] (GPT2 path)
        """
        
        # Recognition path: current state prediction
        input_frames = batch['target_next_frames'][:, :-1].to(self.device)      # [B, T, D]
        current_actions = batch['target_next_actions'][:, :-1].to(self.device)  # [B, T, A]
        current_phases = batch['target_next_phases'][:, :-1].to(self.device) if 'target_next_phases' in batch else None
        
        # Generation path: next state prediction  
        next_frames = batch['target_next_frames'][:, 1:].to(self.device)        # [B, T, D]
        next_actions = batch['target_next_actions'][:, 1:].to(self.device)      # [B, T, A]
        next_phases = batch['target_next_phases'][:, 1:].to(self.device) if 'target_next_phases' in batch else None
        
        return {
            'input_frames': input_frames,
            'current_actions': current_actions,
            'current_phases': current_phases,
            'target_next_frames': next_frames,
            'target_next_actions': next_actions,
            'target_next_phases': next_phases
        }

    def _enhanced_training_complete(self):
        """Mark enhanced training as complete and return best model info"""
        self.logger.info("🎓 Enhanced Autoregressive IL Training Complete!")
        
        # Log final best model summary
        if hasattr(self, 'best_model_paths') and self.best_model_paths:
            self.logger.info("📁 FINAL BEST MODELS SUMMARY:")
            for model_type, path in self.best_model_paths.items():
                if path and os.path.exists(path):
                    self.logger.info(f"   🎯 {model_type}: {path}")
            
            # Return the best combined model path, or fallback to any available model
            return (self.best_model_paths.get('best_combined') or 
                    self.best_model_paths.get('best_next_prediction') or 
                    self.best_model_paths.get('best_current_recognition') or
                    list(self.best_model_paths.values())[0])
        else:
            self.logger.warning("⚠️ No best model paths found!")
            return None

    def get_best_model_paths(self):
        """Return dictionary of best model paths"""
        if hasattr(self, 'best_model_paths'):
            return self.best_model_paths
        else:
            return {}

    def get_best_metrics(self):
        """Return dictionary of best metrics achieved during training"""
        if hasattr(self, 'best_metrics'):
            return self.best_metrics
        else:
            return {
                'best_ivt_current_mAP': 0.0,
                'best_ivt_next_mAP': 0.0,
                'best_combined_score': 0.0
            }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Enhanced training epoch with comprehensive monitoring."""
        
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                
                # Your existing data preparation
                data = self._prepare_training_data(batch)
                
                # Forward pass
                outputs = self.model(
                    frame_embeddings=data['input_frames'],
                    target_next_frames=data['target_next_frames'],
                    target_current_actions=data['current_actions'],
                    target_actions=data['target_next_actions'],
                    target_phases=data['target_next_phases'],
                    epoch=epoch
                )
                
                loss = outputs['total_loss']
                
                # Backward pass with enhanced monitoring
                self.optimizer.zero_grad()
                loss.backward()
                
                # ENHANCED: Track gradients before clipping
                self.optimizer_scheduler.track_gradients()
                
                # Gradient clipping
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
                self.optimizer.step()
                
                # ENHANCED: Step scheduler for step-based schedulers
                if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.optimizer_scheduler.step()
                
                # Accumulate metrics
                for key, value in outputs.items():
                    if key.endswith('loss') and isinstance(value, torch.Tensor):
                        epoch_metrics[key] += value.item()
                
                # ENHANCED: Log learning rates to tensorboard
                global_step = epoch * num_batches + batch_idx
                if batch_idx % self.log_every_n_steps == 0:
                    self.optimizer_scheduler.log_to_tensorboard(self.tb_writer, global_step)
                    
                    # Log current learning rates in progress bar
                    current_lrs = self.optimizer_scheduler.get_current_lrs()
                    action_lr = current_lrs.get('action_prediction', 0)
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'frame': f"{outputs.get('frame_loss', 0):.4f}",
                        'action': f"{outputs.get('action_loss', 0):.4f}",
                        'action_lr': f"{action_lr:.2e}"  # Show action head LR
                    })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            self.metrics_history[f"train_{key}"].append(epoch_metrics[key])
        
        return dict(epoch_metrics)    

    def _save_best_models_based_on_ivt(self, val_metrics: Dict[str, float], epoch: int):
        """
        ENHANCED: Save best models based on different IVT metrics with minimum thresholds.
        
        Args:
            val_metrics: Validation metrics dictionary
            epoch: Current epoch number
        """
        
        # Extract IVT metrics
        current_ivt_map = val_metrics.get('ivt_current_mAP', 0.0)
        next_ivt_map = val_metrics.get('ivt_next_mAP', 0.0)
        
        # Get weights from config or use defaults
        il_config = self.config.get('experiment', {}).get('autoregressive_il', {})
        ivt_config = il_config.get('ivt_saving', {})
        current_weight = ivt_config.get('current_weight', 0.4)
        next_weight = ivt_config.get('next_weight', 0.6)
        
        # Calculate combined score
        combined_score = current_weight * current_ivt_map + next_weight * next_ivt_map
        
        models_saved = []
        models_skipped = []
        
        # 1. Save best current action recognition model
        if current_ivt_map > self.best_ivt_current_map:
            if current_ivt_map >= self.min_current_map_threshold:
                self.best_ivt_current_map = current_ivt_map
                self.best_current_model_path = os.path.join(
                    self.checkpoint_dir, f"autoregressive_il_best_current_epoch_{epoch+1}.pt"
                )
                self.model.save_model(self.best_current_model_path)
                models_saved.append(f"Current Recognition (mAP: {current_ivt_map:.4f})")
            else:
                models_skipped.append(f"Current Recognition (mAP: {current_ivt_map:.4f} < {self.min_current_map_threshold:.2f})")
            
        # 2. Save best next action prediction model  
        if next_ivt_map > self.best_ivt_next_map:
            if next_ivt_map >= self.min_next_map_threshold:
                self.best_ivt_next_map = next_ivt_map
                self.best_next_model_path = os.path.join(
                    self.checkpoint_dir, f"autoregressive_il_best_next_epoch_{epoch+1}.pt"
                )
                self.model.save_model(self.best_next_model_path)
                models_saved.append(f"Next Prediction (mAP: {next_ivt_map:.4f})")
            else:
                models_skipped.append(f"Next Prediction (mAP: {next_ivt_map:.4f} < {self.min_next_map_threshold:.2f})")
            
        # 3. Save best combined performance model
        if combined_score > self.best_combined_ivt_score:
            if combined_score >= self.min_combined_threshold:
                self.best_combined_ivt_score = combined_score
                self.best_combined_model_path = os.path.join(
                    self.checkpoint_dir, f"autoregressive_il_best_combined_epoch_{epoch+1}.pt"
                )
                self.model.save_model(self.best_combined_model_path)
                models_saved.append(f"Combined (score: {combined_score:.4f})")
                
                # Update legacy best model path for backward compatibility
                self.best_model_path = self.best_combined_model_path
            else:
                models_skipped.append(f"Combined (score: {combined_score:.4f} < {self.min_combined_threshold:.2f})")
        
        # Legacy: Save based on loss (for backward compatibility) - no threshold for loss
        if val_metrics.get('total_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = val_metrics['total_loss']
            legacy_path = os.path.join(
                self.checkpoint_dir, f"autoregressive_il_best_loss_epoch_{epoch+1}.pt"
            )
            self.model.save_model(legacy_path)
            models_saved.append(f"Best Loss ({self.best_val_loss:.4f})")
        
        # Log what was saved
        if models_saved:
            self.logger.info(f"✅ New best models saved:")
            for model_desc in models_saved:
                self.logger.info(f"   📁 {model_desc}")
        
        # Log what was skipped due to threshold
        if models_skipped:
            self.logger.info(f"⏭️  Models skipped (below threshold):")
            for model_desc in models_skipped:
                self.logger.info(f"   📊 {model_desc}")
        
        # Always log current best scores
        self.logger.info(f"📊 Current best IVT scores:")
        self.logger.info(f"   🎯 Current Recognition: {self.best_ivt_current_map:.4f} (threshold: {self.min_current_map_threshold:.2f})")
        self.logger.info(f"   🎯 Next Prediction: {self.best_ivt_next_map:.4f} (threshold: {self.min_next_map_threshold:.2f})")
        self.logger.info(f"   🎯 Combined Score: {self.best_combined_ivt_score:.4f} (threshold: {self.min_combined_threshold:.2f})")
        
        # Generate analysis
        analysis = self.optimizer_scheduler.analyze_learning_progress()
        self.logger.info("📊 Learning Progress Analysis:")
        for key, value in analysis.items():
            if isinstance(value, dict):
                self.logger.info(f"   {key}:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"     {subkey}: {subvalue}")
            else:
                self.logger.info(f"   {key}: {value}")
        
        # Final recommendations
        recommendations = self.optimizer_scheduler.get_optimization_recommendations()
        self.logger.info("💡 Final Optimization Recommendations:")
        for rec in recommendations:
            self.logger.info(f"   • {rec}")
            
        # ENHANCED: Summary of saved models
        self.logger.info("📁 SAVED MODELS SUMMARY:")
        if self.best_current_model_path:
            self.logger.info(f"   🎯 Best Current Recognition: {self.best_current_model_path}")
            self.logger.info(f"      IVT Current mAP: {self.best_ivt_current_map:.4f}")
        
        if self.best_next_model_path:
            self.logger.info(f"   🎯 Best Next Prediction: {self.best_next_model_path}")
            self.logger.info(f"      IVT Next mAP: {self.best_ivt_next_map:.4f}")
        
        if self.best_combined_model_path:
            self.logger.info(f"   🎯 Best Combined Performance: {self.best_combined_model_path}")
            self.logger.info(f"      Combined Score: {self.best_combined_ivt_score:.4f}")
        
        self.logger.info("✅ Enhanced training completed with IVT-based model saving!")

    def _validate_epoch_per_video(self, test_loaders: Dict[str, DataLoader], epoch: int) -> Dict[str, float]:
        """
        🔧 FIXED: Per-video validation with correct variable naming.
        
        Process:
        1. For each video: compute metrics individually
        2. Aggregate across videos with proper statistics
        3. Track per-video performance over time
        """
        
        self.model.eval()

        ivt_rec_curr = ivtmetrics.Recognition(num_class=100)        
        ivt_rec_next = ivtmetrics.Recognition(num_class=100)

        # Store metrics per video
        video_loss_metrics = {}
        video_action_metrics = {}
        
        with torch.no_grad():
            for video_id, test_loader in test_loaders.items():
                
                # Per-video collections
                # Current frame recognition
                video_actions_recognition = []
                video_actions_targets_curr = []
                # Next frame prediction
                video_predictions = []
                video_targets = []
                video_losses = defaultdict(float)
                num_batches = len(test_loader)
                
                for batch in tqdm(test_loader, desc=f"Validating {video_id}", leave=True):
                    
                    # Prepare data for this video
                    data = self._prepare_training_data(batch)

                    # Forward pass (no action conditioning!)
                    outputs = self.model(
                        frame_embeddings=data['input_frames'],
                        target_next_frames=data['target_next_frames'],
                        target_current_actions=data['current_actions'],
                        target_actions=data['target_next_actions'],
                        target_phases=data['target_next_phases'] if epoch >= 0 else None,
                    )
                    
                    # Accumulate loss metrics for this video
                    for key, value in outputs.items():
                        if key.endswith('loss') and isinstance(value, torch.Tensor):
                            video_losses[key] += value.item()
                    
                    # CURRENT ACTION RECOGNITION
                    if 'action_rec_probs' in outputs:
                        action_rec_probs = outputs['action_rec_probs']
                        current_actions = data['current_actions']  # [batch, seq_len, num_actions]
                        
                        # Collect action recognition predictions
                        # video_actions_recognition.append(action_rec_probs.cpu().numpy())
                        # video_actions_targets_curr.append(current_actions.cpu().numpy())

                        # Handle sequence dimension for IVT metrics
                        if action_rec_probs.ndim == 3:  # [batch, seq_len, num_actions]
                            action_rec_probs_flat = action_rec_probs[:, -1, :].cpu().numpy()  # Current timestep
                            target_current_actions_flat = current_actions[:, -1, :].cpu().numpy()
                        else:
                            action_rec_probs_flat = action_rec_probs.cpu().numpy()
                            target_current_actions_flat = current_actions.cpu().numpy()

                        # Convert targets to int (ivtmetrics expects binary labels)
                        target_current_actions_int = (target_current_actions_flat > 0.5).astype(int)
                        ivt_rec_curr.update(target_current_actions_int, action_rec_probs_flat)

                        video_actions_recognition.append((action_rec_probs_flat > 0.5).astype(int))
                        video_actions_targets_curr.append(target_current_actions_flat)
                    
                    # NEXT ACTION PREDICTION
                    action_probs = outputs['action_pred']  # [batch, seq_len, num_actions]
                    next_actions = data['target_next_actions']  # [batch, seq_len, num_actions]

                    # Handle sequence dimension
                    if action_probs.ndim == 3:  # [batch, seq_len, num_actions]
                        action_probs_flat = action_probs[:, -1, :].cpu().numpy()  # Last timestep
                        next_actions_flat = next_actions[:, -1, :].cpu().numpy()  # FIXED: next_actions
                    else:
                        action_probs_flat = action_probs.cpu().numpy()
                        next_actions_flat = next_actions.cpu().numpy()  # FIXED: next_actions
                    
                    video_predictions.append(action_probs_flat)
                    video_targets.append(next_actions_flat)
                    
                    # Convert targets to int (ivtmetrics expects binary labels)
                    next_actions_int = (next_actions_flat > 0.5).astype(int)  # FIXED: next_actions
                    
                    # Update IVT recognizer (standard protocol)
                    ivt_rec_next.update(next_actions_int, action_probs_flat)
            
                # End of current video processing
                ivt_rec_curr.video_end()
                ivt_rec_next.video_end()

                # 🎯 PER-VIDEO: Compute action metrics for this specific video
                if video_predictions:
                    # Concatenate all batches for this video
                    video_pred_array = np.concatenate(video_predictions, axis=0)
                    video_target_array = np.concatenate(video_targets, axis=0)
                    
                    # Handle sequence dimension (take last timestep for autoregressive models)
                    if video_pred_array.ndim == 3:  # [batch, seq_len, num_actions]
                        pred_flat = video_pred_array[:, -1, :]  # [batch, num_actions]
                        target_flat = video_target_array[:, -1, :]    # [batch, num_actions]
                    else:  # [batch, num_actions]
                        pred_flat = video_pred_array
                        target_flat = video_target_array
                    
                    # Use shared metrics module for this video
                    video_action_result = calculate_comprehensive_action_metrics(
                        predictions=pred_flat,
                        ground_truth=target_flat,
                        method_name=f"AutoregressiveIL_video_{video_id}_epoch_{epoch}"
                    )
                    
                    # Store this video's action metrics
                    video_action_metrics[video_id] = video_action_result
                    
                    # Track per-video performance over time
                    self.video_performance_history[video_id].append({
                        'epoch': epoch,
                        'mAP': video_action_result['mAP'],
                    })
                
                # Average loss metrics for this video
                for key in video_losses:
                    video_losses[key] /= num_batches
                
                # Store this video's loss metrics
                video_loss_metrics[video_id] = dict(video_losses)

                # if save_predictions and epoch<0:
                if self.config.get('visualization', {}).get('qualitative_evaluation', False):
                    # Save qualitative predictions for this video
                    self._save_video_predictions(
                        video_id=video_id,
                        predictions=video_actions_recognition,
                        targets=video_actions_targets_curr,
                    )
        
        # END OF ALL VIDEOS PROCESSING

        # IVT metrics computation
        final_metrics = {}
        ivt_metrics = {}

        # ACTION RECOGNITION IVT metrics
        ivt_result = ivt_rec_curr.compute_video_AP("ivt")
        ivt_metrics['ivt_current_mAP'] = ivt_result["mAP"]
        ivt_metrics['ivt_current_AP_per_class'] = ivt_result["AP"]
        self.logger.info(f"IVT current mAP: {ivt_metrics['ivt_current_mAP']:.4f}")

        for component in ['i', 'v', 't', 'iv', 'it']:
            comp_result = ivt_rec_curr.compute_video_AP(component)
            ivt_metrics[f'ivt_rec_{component}_mAP'] = comp_result["mAP"]
            self.logger.info(f"IVT {component} mAP: {ivt_metrics[f'ivt_rec_{component}_mAP']:.4f}")
        
        # NEXT ACTION PREDICTION IVT metrics
        ivt_result = ivt_rec_next.compute_video_AP("ivt")
        ivt_metrics['ivt_next_mAP'] = ivt_result["mAP"]
        ivt_metrics['ivt_next_AP_per_class'] = ivt_result["AP"]
        self.logger.info(f"IVT next mAP: {ivt_metrics['ivt_next_mAP']:.4f}")

        for component in ['i', 'v', 't', 'iv', 'it']:
            comp_result = ivt_rec_next.compute_video_AP(component)
            ivt_metrics[f'ivt_{component}_mAP'] = comp_result["mAP"]
            self.logger.info(f"IVT {component} mAP: {ivt_metrics[f'ivt_{component}_mAP']:.4f}")
                
        final_metrics.update(ivt_metrics)

        # 🎯 AGGREGATE: Average metrics across all videos
        aggregated_loss_metrics = {}
        if video_loss_metrics:
            # Get all loss metric keys
            loss_keys = set()
            for video_losses in video_loss_metrics.values():
                loss_keys.update(video_losses.keys())
            
            # Average each loss metric across videos
            for key in loss_keys:
                values = [video_losses.get(key, 0.0) for video_losses in video_loss_metrics.values()]
                aggregated_loss_metrics[key] = np.mean(values)
        
        # Aggregate action metrics across videos
        aggregated_action_metrics = {}
        action_metric_variance = {}
        if video_action_metrics:
            # Get all action metric keys
            action_keys = set()
            for video_metrics in video_action_metrics.values():
                action_keys.update(video_metrics.keys())
            
            # Average each action metric across videos and compute variance
            for key in action_keys:
                values = [video_metrics.get(key, 0.0) for video_metrics in video_action_metrics.values()]
                if isinstance(values[0], (int, float, np.number)):  # Only average numeric values
                    aggregated_action_metrics[key] = np.mean(values)
                    if len(values) > 1:
                        action_metric_variance[f"{key}_std"] = np.std(values)
        
        # 🔧 COMBINE: Merge loss and action metrics
        final_metrics.update(aggregated_loss_metrics)
        
        # 🔧 UPDATED: Map action metrics to expected names for training monitoring
        # Using the new naming convention: clean names = surgical actions only
        if aggregated_action_metrics:
            action_metric_mapping = {
                'action_mAP': aggregated_action_metrics.get('mAP', 0.0),  # Main mAP (surgical actions only)
                'action_mAP_standard': aggregated_action_metrics.get('mAP_standard', 0.0),  # Standard on surgical actions
                'action_mAP_freq_weighted': aggregated_action_metrics.get('mAP_freq_weighted', 0.0),  # Freq weighted on surgical actions
                'action_hamming_accuracy': aggregated_action_metrics.get('hamming_accuracy', 0.0),  # Hamming on surgical actions
                'action_precision': aggregated_action_metrics.get('precision', 0.0),  # Precision on surgical actions
                'action_recall': aggregated_action_metrics.get('recall', 0.0),  # Recall on surgical actions
                'action_f1': aggregated_action_metrics.get('f1', 0.0),  # F1 on surgical actions
                'action_sparsity': aggregated_action_metrics.get('action_sparsity', 0.0),  # Sparsity of surgical actions
                'num_actions_present': aggregated_action_metrics.get('num_actions_present', 0.0),  # Present surgical actions
                                
                # 🔧 NEW: Add alternative metrics for comparison (with null verbs)
                'action_mAP_with_null_verb': aggregated_action_metrics.get('mAP_present_only_with_null_verb', 0.0),
                'action_sparsity_with_null_verb': aggregated_action_metrics.get('action_sparsity_with_null_verb', 0.0),
                'num_actions_total': aggregated_action_metrics.get('num_actions_total', 0.0),  # Total surgical actions
                'num_actions_total_with_null_verb': aggregated_action_metrics.get('num_actions_total_with_null_verb', 0.0)  # All 100 actions
            }
            final_metrics.update(action_metric_mapping)
            
            # Add variance metrics with correct names
            if action_metric_variance:
                # Map variance metrics to training expected names
                variance_mapping = {
                    'action_mAP_std': action_metric_variance.get('mAP_std', 0.0),
                    'action_sparsity_std': action_metric_variance.get('action_sparsity_std', 0.0)
                }
                final_metrics.update(variance_mapping)
                
        # MODIFY YOUR LOGGING: Add IVT metrics
        num_videos = len(video_action_metrics)
        map_std = final_metrics.get('action_mAP_std', 0.0)
        next_map = final_metrics.get('action_mAP', 0.0)
        ivt_current_map = final_metrics.get('ivt_current_mAP', 0.0)
        ivt_next_map = final_metrics.get('ivt_next_mAP', 0.0)
        
        self.logger.info(f"📊 Validation Summary (averaged across {num_videos} videos):")
        # self.logger.info(f"   🎯 Current System mAP:     {next_map:.4f} ± {map_std:.4f}")

        self.logger.info(f"   📊 IVT current mAP:                {ivt_current_map:.4f}")
        self.logger.info(f"   📊 IVT next mAP:                   {ivt_next_map:.4f}")

        curr_ivt_map = final_metrics.get('ivt_current_mAP', 0.0)
        self.logger.info(f"   📊 IVT current mAP:         {curr_ivt_map:.4f}")

        for comp in ['i', 'v', 't', 'iv', 'it']:
            comp_map = final_metrics.get(f'ivt_rec_{comp}_mAP', 0.0)
            self.logger.info(f"   📊 IVT current {comp.upper()} mAP: {comp_map:.4f}")
        
        for comp in ['i', 'v', 't', 'iv', 'it']:
            comp_map = final_metrics.get(f'ivt_{comp}_mAP', 0.0)
            self.logger.info(f"   📊 IVT next {comp.upper()} mAP:     {comp_map:.4f}")
        
        # Store detailed per-video results for analysis
        self.last_validation_details = {
            'epoch': epoch,
            'video_loss_metrics': video_loss_metrics,
            'video_action_metrics': video_action_metrics,
            'aggregated_metrics': final_metrics,
            'num_videos': num_videos
        }
        
        # Update metrics history
        for key, value in final_metrics.items():
            self.metrics_history[f"val_{key}"].append(value)
        
        # Save per-class APs to JSON
        if 'ivt_current_AP_per_class' in final_metrics and 'ivt_next_AP_per_class' in final_metrics:
            self._save_per_class_AP_to_json(final_metrics)
        
        return final_metrics

    def _save_video_predictions(self, video_id: str, predictions: List[np.ndarray], targets: List[np.ndarray]):
        """
        Save qualitative predictions for a specific video.
        
        Args:
            video_id: Unique identifier for the video
            predictions: List of action recognition predictions
            targets: List of ground truth actions
        """
        
        # Create directory if it doesn't exist
        predictions_dir = os.path.join(self.log_dir, "outputs", "predictions")
        ground_truth_dir = os.path.join(self.log_dir, "outputs", "ground_truth")
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        
        # Save predictions and targets as numpy arrays
        pred_path = os.path.join(predictions_dir, f'{video_id}_recognition_pred.npy')
        target_path = os.path.join(ground_truth_dir, f'{video_id}_recognition_gt.npy')
        
        np.save(pred_path, np.concatenate(predictions, axis=0))  # Concatenate across batches
        np.save(target_path, np.concatenate(targets, axis=0))
        
        self.logger.info(f"📁 Saved video predictions for {video_id} to:")
        self.logger.info(f"   Predictions: {pred_path}")
        self.logger.info(f"   Ground Truth: {target_path}")

    def _save_per_class_AP_to_json(self, final_metrics: Dict[str, np.ndarray]):

        # use self.log_dir
        output_path = os.path.join(self.log_dir, "per_class_APs.json")
        self.logger.info(f"📊 Saving per-class APs to {output_path}...")
        
        # Mapping class indices to human-readable names
        class_names_path = "data/labels.json"
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        action_classes = class_names.get('action', {}) # format: {"0": "class_name_0", "1": "class_name_1", ...}

        # Prepare per-class AP data
        per_class_data = {
            'ivt_current_AP_per_class': {},
            'ivt_next_AP_per_class': {}
        }
        # Map array values to class names
        for i, ap in enumerate(final_metrics['ivt_current_AP_per_class']):
            class_name = action_classes[str(i)]
            per_class_data['ivt_current_AP_per_class'][class_name] = float(ap) if not np.isnan(ap) else None
        for i, ap in enumerate(final_metrics['ivt_next_AP_per_class']):
            class_name = action_classes[str(i)]
            per_class_data['ivt_next_AP_per_class'][class_name] = float(ap) if not np.isnan(ap) else None
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(per_class_data, f, indent=4)
        self.logger.info(f"📊 Per-class AP saved to {output_path}")
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to tensorboard and history with enhanced IVT tracking."""
        
        # Log to tensorboard
        for key, value in train_metrics.items():
            self.tb_writer.add_scalar(f"train/{key}_epoch", value, epoch)
        
        for key, value in val_metrics.items():
            self.tb_writer.add_scalar(f"val/{key}_epoch", value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.tb_writer.add_scalar("train/learning_rate", current_lr, epoch)
        
        # ENHANCED: Log IVT metrics specifically
        if 'ivt_current_mAP' in val_metrics:
            self.tb_writer.add_scalar("val/ivt_current_mAP", val_metrics['ivt_current_mAP'], epoch)
        if 'ivt_next_mAP' in val_metrics:
            self.tb_writer.add_scalar("val/ivt_next_mAP", val_metrics['ivt_next_mAP'], epoch)
        
        # Log best scores
        self.tb_writer.add_scalar("best/ivt_current_mAP", self.best_ivt_current_map, epoch)
        self.tb_writer.add_scalar("best/ivt_next_mAP", self.best_ivt_next_map, epoch)
        self.tb_writer.add_scalar("best/combined_score", self.best_combined_ivt_score, epoch)
        
        # 🔧 NEW: Log per-video variance to tensorboard
        if hasattr(self, 'last_validation_details'):
            video_metrics = self.last_validation_details.get('video_action_metrics', {})
            if len(video_metrics) > 1:
                map_values = [metrics['mAP'] for metrics in video_metrics.values()]
                map_std = np.std(map_values)
                self.tb_writer.add_scalar("val/action_mAP_std", map_std, epoch)
                
                # Log per-video mAP values
                for video_id, metrics in video_metrics.items():
                    self.tb_writer.add_scalar(f"val_per_video/mAP_{video_id}", metrics['mAP'], epoch)

    def evaluate_model(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        🔧 FIXED: Comprehensive evaluation using per-video aggregation.
        Planning evaluation is done ONCE in _validate_epoch_per_video_with_planning.
        """
        
        self.logger.info("📊 Evaluating Autoregressive IL Model with Planning Assessment...")
        self.model.eval()
        
        # Use the per-video validation approach (includes planning evaluation)
        evaluate_results = self._validate_epoch_per_video_with_planning(
            test_loaders=test_loaders, 
            epoch=-1,  # Special epoch for final evaluation
            include_planning=True  # Include planning evaluation
        )
        
        # Get detailed per-video results
        detailed_results = getattr(self, 'last_validation_details', {})

        return {
            'overall_metrics': evaluate_results,
            'detailed_results': detailed_results,
            'last_planning_results': self._last_planning_results if hasattr(self, '_last_planning_results') else None,
            'best_model_paths': self.get_best_model_paths(),
        }

    def _extract_planning_metric(self, planning_results: Dict, horizon: str, metric: str) -> float:
        """FIXED: Helper to extract planning metrics safely with proper error handling."""
        if planning_results is None:
            return 0.0
        
        try:
            aggregated = planning_results.get('aggregated_results', {})
            horizon_data = aggregated.get('horizon_aggregated', {})
            horizon_metrics = horizon_data.get(horizon, {})
            return horizon_metrics.get(metric, 0.0)
        except (KeyError, TypeError, AttributeError):
            return 0.0

    def _calculate_planning_degradation(self, planning_results: Dict) -> float:
        """FIXED: Calculate how much performance degrades with longer planning horizons."""
        if planning_results is None:
            return 0.0
        
        aggregated = planning_results.get('aggregated_results', {})
        horizon_data = aggregated.get('horizon_aggregated', {})
        
        # FIXED: Use consistent key names
        perf_1s = horizon_data.get('1s', {}).get('mean_ivt_mAP', 0)
        perf_5s = horizon_data.get('5s', {}).get('mean_ivt_mAP', 0)
        perf_10s = horizon_data.get('10s', {}).get('mean_ivt_mAP', 0)
        
        if perf_1s > 0:
            degradation = (perf_1s - perf_5s) / perf_1s
            return max(0, degradation)  # Return 0 if performance improves
        
        return 0.0


    def _log_comprehensive_results(self, overall_metrics: Dict, planning_results: Dict):
        """FIXED: Enhanced logging with planning results."""
        
        next_map = overall_metrics.get('action_mAP', 0.0)
        ivt_map = overall_metrics.get('ivt_mAP', 0.0)
        map_std = overall_metrics.get('action_mAP_std', 0.0)
        
        self.logger.info(f"✅ Comprehensive evaluation completed")
        self.logger.info(f"📊 SINGLE-STEP PERFORMANCE:")
        self.logger.info(f"   Current System mAP:     {next_map:.4f} ± {map_std:.4f}")
        self.logger.info(f"   IVT Standard mAP:       {ivt_map:.4f}")
        
        # FIXED: Planning results with proper null checking
        if planning_results and 'aggregated_results' in planning_results:
            self.logger.info(f"🎯 PLANNING PERFORMANCE:")
            aggregated = planning_results['aggregated_results'].get('horizon_aggregated', {})
            
            # FIXED: Use consistent key names
            for horizon_name in ['1s', '2s', '3s', '5s', '10s']:
                if horizon_name in aggregated:
                    horizon_data = aggregated[horizon_name]
                    seconds = horizon_data.get('planning_horizon_seconds', 0)
                    planning_map = horizon_data.get('mean_ivt_mAP', 0)
                    consistency = horizon_data.get('mean_action_consistency', 0)
                    
                    self.logger.info(f"   {horizon_name} ({seconds:.0f}s): mAP={planning_map:.4f}, Consistency={consistency:.3f}")
            
            # Planning degradation
            degradation = self._calculate_planning_degradation(planning_results)
            self.logger.info(f"   Planning degradation (1s→5s): {degradation:.3f}")
        else:
            self.logger.warning("🎯 Planning evaluation not available")
        
        self.logger.info(f"📈 Model demonstrates both recognition and planning capabilities")

    def _validate_epoch_per_video_with_planning(self, test_loaders, epoch, include_planning=False):
        """
        FIXED: Enhanced validation that includes planning evaluation during training.
        Stores planning results to avoid duplicate evaluation.
        """
        
        # Standard validation
        standard_metrics = self._validate_epoch_per_video(test_loaders, epoch)

        # Add planning evaluation every few epochs
        if include_planning and (epoch % 5 == 0 or epoch == -1):  # Every 5 epochs or final
            self.logger.info(f"🎯 Including planning evaluation for epoch {epoch}...")
            
            # Quick planning evaluation (shorter horizons only)
            planning_evaluator = AutoregressivePlanningEvaluator(
                model=self.model,
                device=self.device,
                logger=self.logger,
                fps=self.fps,
                planning_horizons=self.planning_horizons,
            )
                        
            planning_results = planning_evaluator.evaluate_planning_on_dataset(test_loaders, self.context_length)
            self._last_planning_results = planning_results
            
            from visualization.map_horizon_plotter import plot_map_vs_horizon, plot_sparsity_analysis
            if self.config.get('visualization', {}).get('enhanced_planning_analysis', True):
                self.logger.info("📊 Generating enhanced planning analysis plots...")
                plot_map_vs_horizon(planning_results, 
                                    save_path=os.path.join(self.visualization_dir, 'planning_analysis.png'),
                                    style='paper',
                                    include_additional_metrics=True)

            if self.config.get('visualization', {}).get('simple_planning_analysis', True):
                self.logger.info("📊 Generating simple planning analysis plots...")
                plot_map_vs_horizon(planning_results,
                                    save_path=os.path.join(self.visualization_dir, 'planning_analysis_simple.png'),
                                    style='paper', 
                                    include_additional_metrics=False)

            if self.config.get('visualization', {}).get('sparsity_analysis', True):
                self.logger.info("📊 Generating sparsity analysis plots...")
                plot_sparsity_analysis(planning_results,
                                    save_path=os.path.join(self.visualization_dir, 'planning_sparsity_analysis.png'))

            # Add planning metrics to standard metrics
            planning_metrics = {}
            for name, seconds in planning_evaluator.planning_horizons.items():
                planning_metrics[f'planning_{name}_mAP'] = self._extract_planning_metric(planning_results, name, 'mean_ivt_mAP')
            standard_metrics.update(planning_metrics)
            standard_metrics.update({
                'planning_available': True
            })                            
        else:
            # No planning evaluation requested or not the right epoch
            self._last_planning_results = None
        
        return standard_metrics
    
    def _analyze_video_performance(self, video_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        🔧 NEW: Analyze performance patterns across videos.
        """
        
        if not video_metrics:
            return {'error': 'No video metrics available'}
        
        # Extract metrics for analysis
        map_values = [metrics['mAP'] for metrics in video_metrics.values()]
        # sparsity_values = [metrics['action_sparsity'] for metrics in video_metrics.values()]
        frame_counts = [metrics['num_predictions'] for metrics in video_metrics.values()]
        
        # Performance categories
        high_performers = [vid for vid, metrics in video_metrics.items() if metrics['mAP'] > np.mean(map_values) + np.std(map_values)]
        low_performers = [vid for vid, metrics in video_metrics.items() if metrics['mAP'] < np.mean(map_values) - np.std(map_values)]
        
        analysis = {
            'summary_stats': {
                'num_videos': len(video_metrics),
                'mAP_stats': {
                    'mean': np.mean(map_values),
                    'std': np.std(map_values),
                    'min': np.min(map_values),
                    'max': np.max(map_values),
                    'median': np.median(map_values),
                    'q25': np.percentile(map_values, 25),
                    'q75': np.percentile(map_values, 75)
                },
                'sparsity_stats': {
                    # 'mean': np.mean(sparsity_values),
                    # 'std': np.std(sparsity_values),
                    # 'correlation_with_mAP': np.corrcoef(map_values, sparsity_values)[0, 1] if len(map_values) > 1 else 0
                }
            },
            'performance_categories': {
                'high_performers': high_performers,
                'low_performers': low_performers,
                'consistent_performers': [vid for vid in video_metrics.keys() 
                                        if vid not in high_performers and vid not in low_performers]
            },
            'detailed_rankings': sorted(
                [(vid, metrics['mAP']) for vid, metrics in video_metrics.items()],
                key=lambda x: x[1], reverse=True
            )
        }
        
        return analysis
    
    def _generate_video_performance_analysis(self):
        """
        🔧 NEW: Generate and save comprehensive video performance analysis.
        """
        
        if not self.video_performance_history:
            return
        
        # Create performance over time plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: mAP over epochs for each video
        for video_id, history in self.video_performance_history.items():
            epochs = [entry['epoch'] for entry in history]
            maps = [entry['mAP'] for entry in history]
            axes[0, 0].plot(epochs, maps, label=video_id, marker='o', markersize=3)
        
        axes[0, 0].set_title('mAP Evolution per Video')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('mAP')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True)
        
        # Plot 2: Final mAP distribution
        if self.last_validation_details:
            video_metrics = self.last_validation_details.get('video_action_metrics', {})
            if video_metrics:
                final_maps = [metrics['mAP'] for metrics in video_metrics.values()]
                axes[0, 1].hist(final_maps, bins=min(10, len(final_maps)), alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(np.mean(final_maps), color='red', linestyle='--', label=f'Mean: {np.mean(final_maps):.3f}')
                axes[0, 1].set_title('Final mAP Distribution')
                axes[0, 1].set_xlabel('mAP')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
        
        # Plot 3: Action sparsity vs performance
        if self.last_validation_details:
            video_metrics = self.last_validation_details.get('video_action_metrics', {})
            if video_metrics:
                sparsities = [metrics['action_sparsity'] for metrics in video_metrics.values()]
                maps = [metrics['mAP'] for metrics in video_metrics.values()]
                axes[1, 0].scatter(sparsities, maps, alpha=0.7)
                axes[1, 0].set_title('Action Sparsity vs mAP')
                axes[1, 0].set_xlabel('Action Sparsity')
                axes[1, 0].set_ylabel('mAP')
                
                # Add correlation
                if len(maps) > 1:
                    corr = np.corrcoef(sparsities, maps)[0, 1]
                    axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                                   transform=axes[1, 0].transAxes, verticalalignment='top')
        
        # Plot 4: Training curves (overall)
        if 'val_action_mAP' in self.metrics_history:
            epochs = range(len(self.metrics_history['val_action_mAP']))
            axes[1, 1].plot(epochs, self.metrics_history['val_action_mAP'], 'b-', label='Validation mAP')
            if 'val_mAP_std' in self.metrics_history:
                maps = np.array(self.metrics_history['val_action_mAP'])
                stds = np.array(self.metrics_history['val_mAP_std'])
                axes[1, 1].fill_between(epochs, maps - stds, maps + stds, alpha=0.3)
            axes[1, 1].set_title('Overall Training Progress')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        analysis_plot_path = os.path.join(self.log_dir, 'per_video_performance_analysis.png')
        plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed performance data
        performance_data = {
            'video_performance_history': dict(self.video_performance_history),
            'final_video_metrics': self.last_validation_details.get('video_action_metrics', {}),
            'training_summary': {
                'epochs': len(self.metrics_history.get('val_action_mAP', [])),
                'final_mAP': self.metrics_history['val_action_mAP'][-1] if self.metrics_history.get('val_action_mAP') else 0,
                'best_mAP': max(self.metrics_history['val_action_mAP']) if self.metrics_history.get('val_action_mAP') else 0
            }
        }
        
        performance_data_path = os.path.join(self.log_dir, 'per_video_performance_data.json')
        with open(performance_data_path, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Per-video analysis saved to: {analysis_plot_path}")
        self.logger.info(f"📊 Performance data saved to: {performance_data_path}")
    
    # def _evaluate_generation_quality(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, float]:
    #     """Evaluate autoregressive generation quality."""
        
    #     generation_metrics = defaultdict(list)
    #     self.logger.info("📊 Evaluating generation quality...")
        
    #     # Test generation on a few samples
    #     for video_id, test_loader in list(test_loaders.items())[:3]:  # Test on 3 videos
    #         batch = next(iter(test_loader))
    #         target_next_frames = batch['target_next_frames'][:10].to(self.device)  # First 10 samples
            
    #         # Generate sequences
    #         generation_results = self.model.generate_sequence(
    #             initial_frames=target_next_frames,
    #             horizon=10,
    #             temperature=0.8
    #         )
            
    #         # Calculate generation quality metrics
    #         generated_frames = generation_results['generated_frames']
    #         predicted_actions = generation_results['predicted_actions']
            
    #         # Frame generation consistency (smoothness)
    #         frame_diffs = torch.diff(generated_frames, dim=1)
    #         frame_smoothness = torch.mean(torch.norm(frame_diffs, dim=-1))
    #         generation_metrics['frame_smoothness'].append(frame_smoothness.item())
            
    #         # Action prediction diversity
    #         action_diversity = torch.mean(torch.std(predicted_actions, dim=1))
    #         generation_metrics['action_diversity'].append(action_diversity.item())
        
    #     # Average metrics
    #     return {key: np.mean(values) for key, values in generation_metrics.items()}
    
    def get_video_performance_summary(self) -> Dict[str, Any]:
        """
        🔧 NEW: Get detailed per-video performance summary.
        """
        
        if not hasattr(self, 'last_validation_details') or not self.last_validation_details:
            return {'error': 'No validation details available'}
        
        video_metrics = self.last_validation_details.get('video_action_metrics', {})
        
        if not video_metrics:
            return {'error': 'No video metrics available'}
        
        # Generate analysis
        analysis = self._analyze_video_performance(video_metrics)
        
        # Add training history context
        analysis['training_context'] = {
            'total_epochs': len(self.metrics_history.get('val_action_mAP', [])),
            'performance_trend': 'improving' if len(self.metrics_history.get('val_action_mAP', [])) > 1 and 
                               self.metrics_history['val_action_mAP'][-1] > self.metrics_history['val_action_mAP'][0] else 'stable'
        }
        
        return analysis
    
    def save_training_plots(self):
        """Save enhanced training history plots with per-video insights."""
        
        if not self.metrics_history:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
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
        
        # Action mAP with variance
        if 'val_action_mAP' in self.metrics_history:
            epochs = range(len(self.metrics_history['val_action_mAP']))
            maps = np.array(self.metrics_history['val_action_mAP'])
            axes[1, 0].plot(epochs, maps, color='green', linewidth=2, label='Mean mAP')
            
            # Add variance bands if available
            if 'val_mAP_std' in self.metrics_history:
                stds = np.array(self.metrics_history['val_mAP_std'])
                axes[1, 0].fill_between(epochs, maps - stds, maps + stds, alpha=0.3, color='green', label='±1 std')
            
            axes[1, 0].set_title('Action Prediction mAP (Per-Video Aggregated)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # ADD THIS: IVT vs Current comparison plot
        if 'val_ivt_mAP' in self.metrics_history and 'val_action_mAP' in self.metrics_history:
            epochs = range(len(self.metrics_history['val_action_mAP']))
            
            axes[1, 2].plot(epochs, self.metrics_history['val_action_mAP'], 
                        'b-', linewidth=2, label='Current System')
            axes[1, 2].plot(epochs, self.metrics_history['val_ivt_mAP'], 
                        'r-', linewidth=2, label='IVT Standard')
            
            axes[1, 2].set_title('Current vs IVT Standard mAP')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('mAP')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        # YOUR EXISTING SAVE CODE
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_curves_with_ivt.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Training plots with IVT comparison saved to: {plot_path}")

        # Save metrics to JSON
        metrics_path = os.path.join(self.log_dir, 'autoregressive_il_metrics_per_video.json')
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
        
        self.logger.info(f"📊 Training plots (PER-VIDEO) saved to: {plot_path}")
        self.logger.info(f"📊 Metrics saved to: {metrics_path}")


# Example usage and testing
if __name__ == "__main__":
    print("🎓 ENHANCED AUTOREGRESSIVE IL TRAINER - IVT-BASED MODEL SAVING")
    print("=" * 70)
    
    print("✅ Enhanced trainer ready for Method 1 (Autoregressive IL)")
    print("🎯 Focus: IVT-based model saving for both tasks")
    print("📊 ENHANCED: Separate best models for:")
    print("   • Current action recognition (IVT current mAP)")
    print("   • Next action prediction (IVT next mAP)")  
    print("   • Combined performance (weighted score)")
    print("📋 Key features:")
    print("   • Tracks best performance for each task separately")
    print("   • Saves multiple model checkpoints")
    print("   • Comprehensive IVT metrics logging")
    print("   • Backward compatible with existing code")
    print("   🚀 PERFORMANCE: Optimized model saving for both tasks!")