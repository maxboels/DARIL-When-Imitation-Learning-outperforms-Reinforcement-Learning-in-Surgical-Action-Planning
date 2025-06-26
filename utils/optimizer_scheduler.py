import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Any
import json
import os

class OptimizerScheduler:
    """
    Enhanced optimizer and scheduler setup for small dataset training (40 videos)
    Features:
    - Intelligent parameter grouping
    - Multiple scheduling strategies
    - Comprehensive LR tracking and visualization
    - Adaptive warmup
    - Per-component learning rates
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 total_epochs: int,
                 steps_per_epoch: int,
                 logger=None):
        
        self.model = model
        self.config = config
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.logger = logger
        
        # Learning rate tracking
        self.lr_history = defaultdict(list)
        self.gradient_norms = defaultdict(list)
        self.parameter_norms = defaultdict(list)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_enhanced_optimizer()
        self.scheduler = self._setup_enhanced_scheduler()
        
        # Track initial state
        self._log_initial_state()
    
    def _setup_enhanced_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with intelligent parameter grouping for dual-path architecture."""
        
        base_lr = float(self.config.get('learning_rate', 1e-4))
        weight_decay = float(self.config.get('weight_decay', 0.01))
        
        # Analyze model architecture for intelligent grouping
        param_groups = []
        
        # 1. BiLSTM Recognition Path (needs careful tuning)
        bilstm_params = []
        for name, param in self.model.named_parameters():
            if 'bilstm' in name.lower() or 'bi_lstm' in name.lower():
                bilstm_params.append(param)
        
        if bilstm_params:
            param_groups.append({
                'params': bilstm_params,
                'lr': base_lr * 0.5,  # Conservative for LSTM
                'weight_decay': weight_decay * 0.5,
                'name': 'bilstm_recognition'
            })
            if self.logger:
                self.logger.info(f"📊 BiLSTM params: {sum(p.numel() for p in bilstm_params):,}")
        
        # 2. GPT2 Backbone (pre-trained-like, needs low LR)
        gpt2_params = []
        for name, param in self.model.named_parameters():
            if 'gpt2' in name.lower() and 'projection' not in name.lower():
                gpt2_params.append(param)
        
        if gpt2_params:
            param_groups.append({
                'params': gpt2_params,
                'lr': base_lr * 0.1,  # Very conservative for transformer
                'weight_decay': weight_decay,
                'name': 'gpt2_backbone'
            })
            if self.logger:
                self.logger.info(f"📊 GPT2 params: {sum(p.numel() for p in gpt2_params):,}")
        
        # 3. Frame Projection and Processing (important for IL)
        projection_params = []
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['projection', 'frame_head', 'next_frame']):
                projection_params.append(param)
        
        if projection_params:
            param_groups.append({
                'params': projection_params,
                'lr': base_lr * 1.5,  # Higher for frame generation
                'weight_decay': weight_decay * 0.3,
                'name': 'frame_processing'
            })
            if self.logger:
                self.logger.info(f"📊 Frame processing params: {sum(p.numel() for p in projection_params):,}")
        
        # 4. Action Prediction Heads (most important for evaluation)
        action_params = []
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['action_head', 'action_prediction', 'next_action']):
                action_params.append(param)
        
        if action_params:
            param_groups.append({
                'params': action_params,
                'lr': base_lr * 2.0,  # Highest LR for action heads
                'weight_decay': weight_decay * 0.1,  # Light regularization
                'name': 'action_prediction'
            })
            if self.logger:
                self.logger.info(f"📊 Action prediction params: {sum(p.numel() for p in action_params):,}")
        
        # 5. Phase and Other Auxiliary Heads
        auxiliary_params = []
        assigned_params = set()
        for group in param_groups:
            assigned_params.update(id(p) for p in group['params'])
        
        for name, param in self.model.named_parameters():
            if id(param) not in assigned_params:
                auxiliary_params.append(param)
        
        if auxiliary_params:
            param_groups.append({
                'params': auxiliary_params,
                'lr': base_lr * 0.8,  # Moderate LR for auxiliary tasks
                'weight_decay': weight_decay,
                'name': 'auxiliary_heads'
            })
            if self.logger:
                self.logger.info(f"📊 Auxiliary params: {sum(p.numel() for p in auxiliary_params):,}")
        
        # Choose optimizer type
        optimizer_type = self.config.get('optimizer_type', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True  # More stable for small datasets
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        if self.logger:
            total_params = sum(sum(p.numel() for p in group['params']) for group in param_groups)
            self.logger.info(f"🎯 Enhanced optimizer created with {len(param_groups)} parameter groups")
            self.logger.info(f"📊 Total parameters: {total_params:,}")
            self.logger.info(f"🔧 Optimizer type: {optimizer_type.upper()}")
        
        return optimizer
    
    def _setup_enhanced_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup sophisticated learning rate scheduler optimized for small datasets."""
        
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine_with_warmup')
        
        # Calculate warmup steps
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        warmup_steps = warmup_epochs * self.steps_per_epoch
        total_steps = self.total_epochs * self.steps_per_epoch
        
        if scheduler_type == 'cosine_with_warmup':
            # Custom cosine with warmup
            scheduler = self._create_cosine_with_warmup_scheduler(warmup_steps, total_steps)
            
        elif scheduler_type == 'linear_with_warmup':
            # Linear decay after warmup
            scheduler = self._create_linear_with_warmup_scheduler(warmup_steps, total_steps)
            
        elif scheduler_type == 'step':
            # Step scheduler
            step_size = scheduler_config.get('step_size', max(1, self.total_epochs // 3))
            gamma = scheduler_config.get('gamma', 0.5)
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            
        elif scheduler_type == 'exponential':
            # Exponential decay
            gamma = scheduler_config.get('gamma', 0.95)
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
            
        elif scheduler_type == 'reduce_on_plateau':
            # Adaptive scheduling based on validation loss
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
            
        elif scheduler_type == 'one_cycle':
            # One cycle policy (good for small datasets)
            max_lr = scheduler_config.get('max_lr', self.config.get('learning_rate', 1e-4) * 3)
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=self.total_epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=scheduler_config.get('pct_start', 0.1),
                anneal_strategy='cos'
            )
            
        elif scheduler_type == 'cosine':
            # Simple cosine annealing
            eta_min = scheduler_config.get('eta_min', 1e-6)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_epochs,
                eta_min=eta_min
            )
            
        else:
            # No scheduler
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
            
        if self.logger:
            self.logger.info(f"📈 Scheduler: {scheduler_type}")
            if hasattr(scheduler, 'warmup_steps'):
                self.logger.info(f"🔥 Warmup steps: {scheduler.warmup_steps}")
            
        return scheduler
    
    def _create_cosine_with_warmup_scheduler(self, warmup_steps: int, total_steps: int):
        """Create custom cosine scheduler with warmup."""
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        scheduler.warmup_steps = warmup_steps
        scheduler.total_steps = total_steps
        return scheduler
    
    def _create_linear_with_warmup_scheduler(self, warmup_steps: int, total_steps: int):
        """Create linear decay scheduler with warmup."""
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay
                return max(0.0, 1.0 - (current_step - warmup_steps) / (total_steps - warmup_steps))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        scheduler.warmup_steps = warmup_steps
        scheduler.total_steps = total_steps
        return scheduler
    
    def step(self, validation_loss: Optional[float] = None, epoch: Optional[int] = None):
        """Step the scheduler with optional validation loss for adaptive schedulers."""
        
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if validation_loss is not None:
                self.scheduler.step(validation_loss)
        else:
            self.scheduler.step()
        
        # Track learning rates for all parameter groups
        self._track_learning_rates(epoch)
    
    def _track_learning_rates(self, epoch: Optional[int] = None):
        """Track learning rates for all parameter groups."""
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            current_lr = param_group['lr']
            self.lr_history[group_name].append(current_lr)
    
    def track_gradients(self):
        """Track gradient norms for monitoring."""
        
        for name, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{name}')
            
            # Calculate gradient norm for this group
            grad_norm = 0.0
            param_norm = 0.0
            num_params = 0
            
            for param in param_group['params']:
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                param_norm += param.data.norm(2).item() ** 2
                num_params += param.numel()
            
            grad_norm = grad_norm ** 0.5
            param_norm = param_norm ** 0.5
            
            self.gradient_norms[group_name].append(grad_norm)
            self.parameter_norms[group_name].append(param_norm)
    
    def get_current_lrs(self) -> Dict[str, float]:
        """Get current learning rates for all parameter groups."""
        
        current_lrs = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            current_lrs[group_name] = param_group['lr']
        
        return current_lrs
    
    def log_to_tensorboard(self, tb_writer, global_step: int):
        """Log learning rates and gradient information to tensorboard."""
        
        current_lrs = self.get_current_lrs()
        
        # Log learning rates
        for group_name, lr in current_lrs.items():
            tb_writer.add_scalar(f"learning_rate/{group_name}", lr, global_step)
        
        # Log gradient norms if available
        if self.gradient_norms:
            for group_name in self.gradient_norms:
                if self.gradient_norms[group_name]:
                    grad_norm = self.gradient_norms[group_name][-1]
                    tb_writer.add_scalar(f"gradients/grad_norm_{group_name}", grad_norm, global_step)
                
                if self.parameter_norms[group_name]:
                    param_norm = self.parameter_norms[group_name][-1]
                    tb_writer.add_scalar(f"gradients/param_norm_{group_name}", param_norm, global_step)
    
    def save_lr_plots(self, save_dir: str):
        """Generate and save comprehensive learning rate analysis plots."""
        
        if not self.lr_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Learning rate curves for all groups
        for group_name, lr_values in self.lr_history.items():
            epochs = range(len(lr_values))
            axes[0, 0].plot(epochs, lr_values, label=group_name, linewidth=2, marker='o', markersize=2)
        
        axes[0, 0].set_title('Learning Rate Schedule by Parameter Group')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Learning rate ratios
        if len(self.lr_history) > 1:
            group_names = list(self.lr_history.keys())
            base_group = group_names[0]
            base_lrs = np.array(self.lr_history[base_group])
            
            for group_name in group_names[1:]:
                if len(self.lr_history[group_name]) == len(base_lrs):
                    ratios = np.array(self.lr_history[group_name]) / (base_lrs + 1e-8)
                    epochs = range(len(ratios))
                    axes[0, 1].plot(epochs, ratios, label=f'{group_name}/{base_group}', linewidth=2)
            
            axes[0, 1].set_title('Learning Rate Ratios Between Groups')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('LR Ratio')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient norms
        if self.gradient_norms:
            for group_name, grad_values in self.gradient_norms.items():
                if grad_values:
                    epochs = range(len(grad_values))
                    axes[1, 0].plot(epochs, grad_values, label=group_name, linewidth=2)
            
            axes[1, 0].set_title('Gradient Norms by Parameter Group')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Plot 4: Parameter norms
        if self.parameter_norms:
            for group_name, param_values in self.parameter_norms.items():
                if param_values:
                    epochs = range(len(param_values))
                    axes[1, 1].plot(epochs, param_values, label=group_name, linewidth=2)
            
            axes[1, 1].set_title('Parameter Norms by Group')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Parameter Norm')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'learning_rate_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.info(f"📊 Learning rate analysis saved to: {plot_path}")
    
    def save_lr_history(self, save_dir: str):
        """Save learning rate history to JSON for later analysis."""
        
        history_data = {
            'lr_history': {k: v for k, v in self.lr_history.items()},
            'gradient_norms': {k: v for k, v in self.gradient_norms.items()},
            'parameter_norms': {k: v for k, v in self.parameter_norms.items()},
            'config': self.config,
            'total_epochs': self.total_epochs,
            'steps_per_epoch': self.steps_per_epoch
        }
        
        history_path = os.path.join(save_dir, 'lr_scheduler_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        if self.logger:
            self.logger.info(f"📊 LR history saved to: {history_path}")
    
    def _log_initial_state(self):
        """Log initial optimizer and scheduler state."""
        
        if self.logger:
            self.logger.info("🎯 Enhanced Optimizer & Scheduler Setup:")
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                initial_lr = param_group['lr']
                weight_decay = param_group['weight_decay']
                num_params = sum(p.numel() for p in param_group['params'])
                
                self.logger.info(f"   {group_name}:")
                self.logger.info(f"     LR: {initial_lr:.6f}")
                self.logger.info(f"     Weight Decay: {weight_decay:.6f}")
                self.logger.info(f"     Parameters: {num_params:,}")
    
    def analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress based on LR and gradient history."""
        
        analysis = {
            'parameter_groups': len(self.optimizer.param_groups),
            'total_steps_tracked': len(next(iter(self.lr_history.values()))) if self.lr_history else 0
        }
        
        # Analyze learning rate trends
        if self.lr_history:
            lr_analysis = {}
            for group_name, lr_values in self.lr_history.items():
                if len(lr_values) > 1:
                    lr_analysis[group_name] = {
                        'initial_lr': lr_values[0],
                        'final_lr': lr_values[-1],
                        'min_lr': min(lr_values),
                        'max_lr': max(lr_values),
                        'lr_reduction_factor': lr_values[0] / (lr_values[-1] + 1e-8)
                    }
            analysis['lr_analysis'] = lr_analysis
        
        # Analyze gradient norms
        if self.gradient_norms:
            grad_analysis = {}
            for group_name, grad_values in self.gradient_norms.items():
                if grad_values:
                    grad_analysis[group_name] = {
                        'mean_grad_norm': np.mean(grad_values),
                        'std_grad_norm': np.std(grad_values),
                        'max_grad_norm': max(grad_values),
                        'final_grad_norm': grad_values[-1]
                    }
            analysis['gradient_analysis'] = grad_analysis
        
        return analysis
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations based on training history."""
        
        recommendations = []
        
        if not self.gradient_norms:
            recommendations.append("Enable gradient tracking for better optimization insights")
            return recommendations
        
        # Analyze gradient patterns
        for group_name, grad_values in self.gradient_norms.items():
            if not grad_values:
                continue
                
            mean_grad = np.mean(grad_values[-10:])  # Last 10 steps
            
            if mean_grad > 10.0:
                recommendations.append(f"High gradients in {group_name} - consider gradient clipping or lower LR")
            elif mean_grad < 1e-5:
                recommendations.append(f"Very small gradients in {group_name} - consider higher LR or check vanishing gradients")
        
        # Analyze learning rate effectiveness
        if self.lr_history:
            for group_name, lr_values in self.lr_history.items():
                if len(lr_values) > 10:
                    recent_lr_change = abs(lr_values[-1] - lr_values[-10]) / (lr_values[-10] + 1e-8)
                    if recent_lr_change < 0.01:
                        recommendations.append(f"{group_name} LR has plateaued - consider adaptive scheduling")
        
        return recommendations if recommendations else ["Training appears stable"]


# Integration with the trainer
def integrate_enhanced_optimizer(trainer_instance):
    """Integrate enhanced optimizer into existing trainer."""
    
    # Replace the existing _setup_optimizer method
    def _setup_enhanced_optimizer_integration(self):
        """Enhanced optimizer setup for the trainer."""
        
        # Get training configuration
        train_config = self.train_config
        steps_per_epoch = len(self.train_loader) if hasattr(self, 'train_loader') else 100
        
        # Create enhanced optimizer-scheduler
        self.optimizer_scheduler = EnhancedOptimizerScheduler(
            model=self.model,
            config=train_config,
            total_epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            logger=self.logger
        )
        
        # Replace optimizer and scheduler
        self.optimizer = self.optimizer_scheduler.optimizer
        self.scheduler = self.optimizer_scheduler.scheduler
        
        self.logger.info("✅ Enhanced optimizer and scheduler integrated")
    
    # Enhanced training step with gradient tracking
    def _enhanced_training_step(self, batch, batch_idx, epoch):
        """Enhanced training step with comprehensive monitoring."""
        
        # Original forward pass
        outputs = self.model(**batch)
        loss = outputs['total_loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Track gradients before clipping
        self.optimizer_scheduler.track_gradients()
        
        # Gradient clipping
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        
        self.optimizer.step()
        
        # Step scheduler (for step-based schedulers)
        if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.optimizer_scheduler.step()
        
        # Log to tensorboard
        global_step = epoch * len(self.train_loader) + batch_idx
        self.optimizer_scheduler.log_to_tensorboard(self.tb_writer, global_step)
        
        return outputs
    
    # Enhanced epoch end with scheduler step
    def _enhanced_epoch_end(self, val_loss, epoch):
        """Enhanced epoch end with scheduler stepping."""
        
        # Step scheduler (for epoch-based schedulers)
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.optimizer_scheduler.step(validation_loss=val_loss, epoch=epoch)
        
        # Log current learning rates
        current_lrs = self.optimizer_scheduler.get_current_lrs()
        self.logger.info(f"📊 Current LRs: {current_lrs}")
        
        # Get optimization recommendations periodically
        if epoch % 10 == 0:
            recommendations = self.optimizer_scheduler.get_optimization_recommendations()
            for rec in recommendations:
                self.logger.info(f"💡 Optimization tip: {rec}")
    
    # Enhanced training completion
    def _enhanced_training_complete(self, save_dir):
        """Enhanced training completion with comprehensive analysis."""
        
        # Save learning rate plots and history
        self.optimizer_scheduler.save_lr_plots(save_dir)
        self.optimizer_scheduler.save_lr_history(save_dir)
        
        # Generate analysis
        analysis = self.optimizer_scheduler.analyze_learning_progress()
        self.logger.info("📊 Learning Progress Analysis:")
        for key, value in analysis.items():
            self.logger.info(f"   {key}: {value}")
        
        # Final recommendations
        recommendations = self.optimizer_scheduler.get_optimization_recommendations()
        self.logger.info("💡 Final Optimization Recommendations:")
        for rec in recommendations:
            self.logger.info(f"   • {rec}")
    
    # Bind methods to trainer instance
    trainer_instance._setup_optimizer = _setup_enhanced_optimizer_integration.__get__(trainer_instance)
    trainer_instance._enhanced_training_step = _enhanced_training_step.__get__(trainer_instance)
    trainer_instance._enhanced_epoch_end = _enhanced_epoch_end.__get__(trainer_instance)
    trainer_instance._enhanced_training_complete = _enhanced_training_complete.__get__(trainer_instance)
    
    return trainer_instance


# Example usage and configuration
if __name__ == "__main__":
    print("🎯 Enhanced Optimizer & Scheduler with LR Tracking")
    print("=" * 60)
    print("Features:")
    print("✅ Intelligent parameter grouping")
    print("✅ Multiple scheduling strategies")
    print("✅ Comprehensive LR tracking")
    print("✅ Gradient monitoring")
    print("✅ Adaptive recommendations")
    print("✅ Visualization and analysis")
    print("✅ Integration with existing trainer")