import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
from typing import Dict, List, Tuple, Any, Optional
import os
import json

class DualModelEvaluator:
    """
    Comprehensive evaluator for the DualWorldModel that can assess both:
    1. Autoregressive action prediction performance
    2. RL state prediction and reward modeling performance
    """
    
    def __init__(self, 
                 model, 
                 config: Dict[str, Any], 
                 device: str = 'cuda',
                 logger=None):
        """
        Initialize the evaluator.
        
        Args:
            model: DualWorldModel instance
            config: Configuration dictionary
            device: Device to evaluate on
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or self._create_dummy_logger()
        
        # Evaluation configurations
        self.eval_config = config.get('evaluation', {})
        self.supervised_config = self.eval_config.get('supervised', {})
        self.rl_config = self.eval_config.get('rl', {})
        
    def _create_dummy_logger(self):
        """Create a dummy logger if none provided."""
        class DummyLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        return DummyLogger()
    
    def evaluate_supervised_mode(self, 
                                test_loaders: Dict[str, Any],
                                save_results: bool = True,
                                save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Evaluate the model's autoregressive action prediction capabilities.
        
        Args:
            test_loaders: Dictionary of test data loaders
            save_results: Whether to save results to disk
            save_dir: Directory to save results
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting supervised mode evaluation (autoregressive action prediction)")
        
        self.model.eval()
        results = {
            'action_prediction': {},
            'state_prediction': {},
            'autoregressive_performance': {},
            'per_video_metrics': {}
        }
        
        # Extract evaluation parameters
        action_config = self.supervised_config.get('action_prediction', {})
        horizons = action_config.get('horizons', [1, 3, 5, 10, 15])
        top_ks = action_config.get('top_ks', [1, 3, 5, 10])
        temperature = action_config.get('temperature', 1.0)
        
        video_count = 0
        total_samples = 0
        total_videos = len(test_loaders)
        self.logger.info(f"Evaluating {total_videos} videos with horizons {horizons} and top-k {top_ks}")
        
        with torch.no_grad():
            for video_id, test_loader in test_loaders.items():
                video_count += 1
                self.logger.info(f"Evaluating video: {video_id} ({video_count}/{total_videos})")
                
                video_metrics = {
                    'single_step': defaultdict(list),
                    'autoregressive': {h: defaultdict(list) for h in horizons}
                }
                
                for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {video_id}")):
                    # Move batch to device
                    current_states = batch['current_states'].to(self.device)
                    next_states = batch['next_states'].to(self.device)
                    next_actions = batch['next_actions'].to(self.device)
                    future_states = batch.get('future_states', None)
                    future_actions = batch.get('future_actions', None)
                    
                    if future_states is not None:
                        future_states = future_states.to(self.device)
                    if future_actions is not None:
                        future_actions = future_actions.to(self.device)
                    
                    # 1. Single-step prediction evaluation
                    single_step_metrics = self._evaluate_single_step_prediction(
                        current_states, next_states, next_actions
                    )
                    
                    for metric, value in single_step_metrics.items():
                        video_metrics['single_step'][metric].append(value)
                    
                    # 2. Autoregressive prediction evaluation
                    if future_states is not None and future_actions is not None:
                        autoregressive_metrics = self._evaluate_autoregressive_prediction(
                            current_states, future_states, future_actions, horizons, top_ks, temperature
                        )
                        
                        for horizon in horizons:
                            if horizon in autoregressive_metrics:
                                for metric, value in autoregressive_metrics[horizon].items():
                                    video_metrics['autoregressive'][horizon][metric].append(value)
                    
                    total_samples += current_states.size(0)
                
                # Average metrics for this video
                video_results = self._aggregate_video_metrics(video_metrics)
                results['per_video_metrics'][video_id] = video_results
        
        # Aggregate overall results
        results['action_prediction'] = self._aggregate_action_prediction_metrics(results['per_video_metrics'])
        results['state_prediction'] = self._aggregate_state_prediction_metrics(results['per_video_metrics'])
        results['autoregressive_performance'] = self._aggregate_autoregressive_metrics(results['per_video_metrics'])
        
        # Add summary statistics
        results['summary'] = {
            'total_samples': total_samples,
            'num_videos': len(test_loaders),
            'evaluation_mode': 'supervised'
        }
        
        self.logger.info(f"Supervised evaluation completed. Processed {total_samples} samples from {len(test_loaders)} videos.")
        
        # Save results if requested
        if save_results:
            os.makedirs(save_dir, exist_ok=True)
            self._save_supervised_results(results, save_dir)
        
        return results
    
    def _evaluate_single_step_prediction(self, 
                                       current_states: torch.Tensor,
                                       next_states: torch.Tensor,
                                       next_actions: torch.Tensor) -> Dict[str, float]:
        """Evaluate single-step prediction performance."""
        outputs = self.model(
            current_states=current_states,
            next_states=next_states,
            next_actions=next_actions,
            mode='supervised'
        )
        
        metrics = {}
        
        # State prediction error
        if 'state_pred' in outputs:
            state_mse = F.mse_loss(outputs['state_pred'], next_states).item()
            metrics['state_mse'] = state_mse
        
        # Action prediction accuracy
        if 'action_pred' in outputs:
            action_logits = outputs['action_pred']
            action_probs = torch.sigmoid(action_logits)
            action_pred_binary = (action_probs > 0.5).float()
            
            # Exact match accuracy
            exact_match = torch.all(action_pred_binary == next_actions, dim=-1).float().mean().item()
            metrics['action_exact_match'] = exact_match
            
            # Hamming accuracy (per-class)
            hamming_acc = (action_pred_binary == next_actions).float().mean().item()
            metrics['action_hamming_accuracy'] = hamming_acc
            
            # mAP calculation
            try:
                ap_scores = []
                for i in range(next_actions.size(-1)):
                    if torch.sum(next_actions[:, :, i]) > 0:  # Only if class is present
                        y_true = next_actions[:, :, i].cpu().numpy().flatten()
                        y_scores = action_probs[:, :, i].cpu().numpy().flatten()
                        ap = average_precision_score(y_true, y_scores)
                        ap_scores.append(ap)
                
                if ap_scores:
                    metrics['action_map'] = np.mean(ap_scores)
                else:
                    metrics['action_map'] = 0.0
            except:
                metrics['action_map'] = 0.0
        
        return metrics
    
    def _evaluate_autoregressive_prediction(self,
                                          current_states: torch.Tensor,
                                          future_states: torch.Tensor,
                                          future_actions: torch.Tensor,
                                          horizons: List[int],
                                          top_ks: List[int],
                                          temperature: float) -> Dict[int, Dict[str, float]]:
        """Evaluate autoregressive prediction performance at different horizons."""
        batch_size = current_states.size(0)
        results = {}
        
        for horizon in horizons:
            if horizon > future_states.size(1):
                continue
            
            horizon_metrics = {}
            
            # Generate autoregressive predictions
            generation_output = self.model.autoregressive_action_prediction(
                initial_states=current_states,
                horizon=horizon,
                temperature=temperature
            )
            
            # Extract predictions
            pred_states = generation_output['predicted_states']  # [batch, horizon, embed_dim]
            pred_actions = generation_output['predicted_actions']  # [batch, horizon, num_actions]
            
            # Evaluate state prediction
            if pred_states.size(1) >= horizon and future_states.size(1) >= horizon:
                state_errors = []
                for t in range(horizon):
                    error_t = F.mse_loss(pred_states[:, t], future_states[:, t]).item()
                    state_errors.append(error_t)
                
                horizon_metrics['state_mse_mean'] = np.mean(state_errors)
                horizon_metrics['state_mse_final'] = state_errors[-1]
                horizon_metrics['state_error_growth'] = state_errors[-1] / (state_errors[0] + 1e-8)
            
            # Evaluate action prediction at different top-k values
            if pred_actions.size(1) >= horizon and future_actions.size(1) >= horizon:
                for k in top_ks:
                    k = min(k, pred_actions.size(-1))
                    
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for t in range(horizon):
                        # Get predictions and targets for timestep t
                        pred_t = pred_actions[:, t, :]  # [batch, num_actions]
                        target_t = future_actions[:, t, :]  # [batch, num_actions]
                        
                        # Convert targets to class indices (assuming one-hot or multi-hot)
                        target_indices = torch.nonzero(target_t, as_tuple=False)
                        
                        if target_indices.numel() > 0:
                            # Get top-k predictions
                            _, topk_pred = torch.topk(pred_t, k, dim=-1)
                            
                            # Check if any target class is in top-k
                            for batch_idx in range(batch_size):
                                batch_targets = target_indices[target_indices[:, 0] == batch_idx, 1]
                                if len(batch_targets) > 0:
                                    batch_topk = topk_pred[batch_idx]
                                    hit = any(target in batch_topk for target in batch_targets)
                                    correct_predictions += int(hit)
                                    total_predictions += 1
                    
                    if total_predictions > 0:
                        horizon_metrics[f'action_top_{k}_accuracy'] = correct_predictions / total_predictions
                    else:
                        horizon_metrics[f'action_top_{k}_accuracy'] = 0.0
            
            results[horizon] = horizon_metrics
        
        return results
    
    def evaluate_rl_mode(self,
                        test_loaders: Dict[str, Any],
                        save_results: bool = True,
                        save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Evaluate the model's RL state prediction and reward modeling capabilities.
        
        Args:
            test_loaders: Dictionary of test data loaders
            save_results: Whether to save results to disk
            save_dir: Directory to save results
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting RL mode evaluation (state and reward prediction)")
        
        self.model.eval()
        results = {
            'state_prediction': {},
            'reward_prediction': {},
            'rl_environment_simulation': {},
            'per_video_metrics': {}
        }
        
        total_samples = 0
        
        with torch.no_grad():
            for video_id, test_loader in test_loaders.items():
                self.logger.info(f"Evaluating RL mode on video: {video_id}")
                
                video_metrics = defaultdict(list)
                
                for batch in tqdm(test_loader, desc=f"RL eval {video_id}"):
                    # Move batch to device
                    current_states = batch['current_states'].to(self.device)
                    next_states = batch['next_states'].to(self.device)
                    next_actions = batch['next_actions'].to(self.device)
                    next_rewards = batch.get('next_rewards', {})
                    next_rewards = {k: v.to(self.device) for k, v in next_rewards.items()}
                    
                    # RL prediction
                    rl_outputs = self.model.rl_state_prediction(
                        current_states=current_states,
                        planned_actions=next_actions,
                        return_rewards=True
                    )
                    
                    # Evaluate state prediction
                    pred_states = rl_outputs['next_states']
                    state_mse = F.mse_loss(pred_states, next_states).item()
                    video_metrics['state_mse'].append(state_mse)
                    
                    # Evaluate reward prediction
                    if 'rewards' in rl_outputs and next_rewards:
                        reward_errors = {}
                        for reward_type, pred_reward in rl_outputs['rewards'].items():
                            if reward_type in next_rewards:
                                target_reward = next_rewards[reward_type]
                                reward_mse = F.mse_loss(pred_reward.squeeze(-1), target_reward.squeeze(-1)).item()
                                reward_errors[f'reward_{reward_type}_mse'] = reward_mse
                                video_metrics[f'reward_{reward_type}_mse'].append(reward_mse)
                    
                    total_samples += current_states.size(0)
                
                # Average metrics for this video
                video_results = {}
                for metric, values in video_metrics.items():
                    video_results[metric] = np.mean(values) if values else 0.0
                
                results['per_video_metrics'][video_id] = video_results
        
        # Aggregate overall results
        results['state_prediction'] = self._aggregate_rl_state_metrics(results['per_video_metrics'])
        results['reward_prediction'] = self._aggregate_rl_reward_metrics(results['per_video_metrics'])
        
        # Add summary statistics
        results['summary'] = {
            'total_samples': total_samples,
            'num_videos': len(test_loaders),
            'evaluation_mode': 'rl'
        }
        
        self.logger.info(f"RL evaluation completed. Processed {total_samples} samples from {len(test_loaders)} videos.")
        
        # Save results if requested
        if save_results:
            os.makedirs(save_dir, exist_ok=True)
            self._save_rl_results(results, save_dir)
        
        return results
    
    def _aggregate_video_metrics(self, video_metrics: Dict) -> Dict:
        """Aggregate metrics for a single video."""
        aggregated = {}
        
        # Single-step metrics
        for metric, values in video_metrics['single_step'].items():
            aggregated[f'single_step_{metric}'] = np.mean(values) if values else 0.0
        
        # Autoregressive metrics
        for horizon, metrics in video_metrics['autoregressive'].items():
            for metric, values in metrics.items():
                aggregated[f'autoregressive_h{horizon}_{metric}'] = np.mean(values) if values else 0.0
        
        return aggregated
    
    def _aggregate_action_prediction_metrics(self, per_video_metrics: Dict) -> Dict:
        """Aggregate action prediction metrics across all videos."""
        metrics = defaultdict(list)
        
        for video_results in per_video_metrics.values():
            for metric, value in video_results.items():
                if 'action' in metric:
                    metrics[metric].append(value)
        
        aggregated = {}
        for metric, values in metrics.items():
            aggregated[metric] = {
                'mean': np.mean(values) if values else 0.0,
                'std': np.std(values) if values else 0.0,
                'min': np.min(values) if values else 0.0,
                'max': np.max(values) if values else 0.0
            }
        
        return aggregated
    
    def _aggregate_state_prediction_metrics(self, per_video_metrics: Dict) -> Dict:
        """Aggregate state prediction metrics across all videos."""
        metrics = defaultdict(list)
        
        for video_results in per_video_metrics.values():
            for metric, value in video_results.items():
                if 'state' in metric:
                    metrics[metric].append(value)
        
        aggregated = {}
        for metric, values in metrics.items():
            aggregated[metric] = {
                'mean': np.mean(values) if values else 0.0,
                'std': np.std(values) if values else 0.0,
                'min': np.min(values) if values else 0.0,
                'max': np.max(values) if values else 0.0
            }
        
        return aggregated
    
    def _aggregate_autoregressive_metrics(self, per_video_metrics: Dict) -> Dict:
        """Aggregate autoregressive prediction metrics."""
        horizon_metrics = defaultdict(lambda: defaultdict(list))
        
        for video_results in per_video_metrics.values():
            for metric, value in video_results.items():
                if 'autoregressive' in metric:
                    # Extract horizon information
                    parts = metric.split('_')
                    if len(parts) >= 3 and parts[1].startswith('h'):
                        horizon = parts[1]
                        metric_name = '_'.join(parts[2:])
                        horizon_metrics[horizon][metric_name].append(value)
        
        aggregated = {}
        for horizon, metrics in horizon_metrics.items():
            aggregated[horizon] = {}
            for metric, values in metrics.items():
                aggregated[horizon][metric] = {
                    'mean': np.mean(values) if values else 0.0,
                    'std': np.std(values) if values else 0.0
                }
        
        return aggregated
    
    def _aggregate_rl_state_metrics(self, per_video_metrics: Dict) -> Dict:
        """Aggregate RL state prediction metrics."""
        state_mse_values = []
        
        for video_results in per_video_metrics.values():
            if 'state_mse' in video_results:
                state_mse_values.append(video_results['state_mse'])
        
        return {
            'state_mse': {
                'mean': np.mean(state_mse_values) if state_mse_values else 0.0,
                'std': np.std(state_mse_values) if state_mse_values else 0.0,
                'min': np.min(state_mse_values) if state_mse_values else 0.0,
                'max': np.max(state_mse_values) if state_mse_values else 0.0
            }
        }
    
    def _aggregate_rl_reward_metrics(self, per_video_metrics: Dict) -> Dict:
        """Aggregate RL reward prediction metrics."""
        reward_metrics = defaultdict(list)
        
        for video_results in per_video_metrics.values():
            for metric, value in video_results.items():
                if 'reward_' in metric and '_mse' in metric:
                    reward_metrics[metric].append(value)
        
        aggregated = {}
        for metric, values in reward_metrics.items():
            aggregated[metric] = {
                'mean': np.mean(values) if values else 0.0,
                'std': np.std(values) if values else 0.0
            }
        
        return aggregated
    
    def _save_supervised_results(self, results: Dict, save_dir: str):
        """Save supervised evaluation results."""
        # Save raw results
        with open(os.path.join(save_dir, 'supervised_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create visualizations
        self._create_supervised_visualizations(results, save_dir)
        
        self.logger.info(f"Supervised evaluation results saved to {save_dir}")
    
    def _save_rl_results(self, results: Dict, save_dir: str):
        """Save RL evaluation results."""
        # Save raw results
        with open(os.path.join(save_dir, 'rl_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create visualizations
        self._create_rl_visualizations(results, save_dir)
        
        self.logger.info(f"RL evaluation results saved to {save_dir}")
    
    def _create_supervised_visualizations(self, results: Dict, save_dir: str):
        """Create visualizations for supervised evaluation results."""
        # Action prediction performance across horizons
        if 'autoregressive_performance' in results:
            self._plot_autoregressive_performance(results['autoregressive_performance'], save_dir)
        
        # Action prediction accuracy by top-k
        if 'action_prediction' in results:
            self._plot_action_prediction_metrics(results['action_prediction'], save_dir)
    
    def _create_rl_visualizations(self, results: Dict, save_dir: str):
        """Create visualizations for RL evaluation results."""
        # State prediction error
        if 'state_prediction' in results:
            self._plot_state_prediction_metrics(results['state_prediction'], save_dir)
        
        # Reward prediction error
        if 'reward_prediction' in results:
            self._plot_reward_prediction_metrics(results['reward_prediction'], save_dir)
    
    def _plot_autoregressive_performance(self, autoregressive_results: Dict, save_dir: str):
        """Plot autoregressive performance across horizons."""
        horizons = []
        state_errors = []
        action_accuracies = []
        
        for horizon_key, metrics in autoregressive_results.items():
            if horizon_key.startswith('h'):
                horizon = int(horizon_key[1:])
                horizons.append(horizon)
                
                # State error
                if 'state_mse_mean' in metrics:
                    state_errors.append(metrics['state_mse_mean']['mean'])
                
                # Action accuracy (top-1)
                if 'action_top_1_accuracy' in metrics:
                    action_accuracies.append(metrics['action_top_1_accuracy']['mean'])
        
        if horizons:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # State prediction error
            if state_errors:
                ax1.plot(horizons, state_errors, 'o-', linewidth=2, markersize=8)
                ax1.set_xlabel('Prediction Horizon')
                ax1.set_ylabel('State MSE')
                ax1.set_title('State Prediction Error vs Horizon')
                ax1.grid(True, alpha=0.3)
            
            # Action prediction accuracy
            if action_accuracies:
                ax2.plot(horizons, action_accuracies, 'o-', linewidth=2, markersize=8, color='orange')
                ax2.set_xlabel('Prediction Horizon')
                ax2.set_ylabel('Top-1 Action Accuracy')
                ax2.set_title('Action Prediction Accuracy vs Horizon')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'autoregressive_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_action_prediction_metrics(self, action_results: Dict, save_dir: str):
        """Plot action prediction metrics."""
        # Extract metrics for plotting
        metrics_to_plot = ['single_step_action_exact_match', 'single_step_action_hamming_accuracy', 'single_step_action_map']
        
        values = []
        labels = []
        
        for metric in metrics_to_plot:
            if metric in action_results:
                values.append(action_results[metric]['mean'])
                labels.append(metric.replace('single_step_action_', '').replace('_', ' ').title())
        
        if values:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values, alpha=0.7)
            plt.ylabel('Score')
            plt.title('Action Prediction Performance')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'action_prediction_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_state_prediction_metrics(self, state_results: Dict, save_dir: str):
        """Plot state prediction metrics."""
        if 'state_mse' in state_results:
            metrics = state_results['state_mse']
            
            plt.figure(figsize=(8, 6))
            plt.bar(['Mean', 'Min', 'Max'], [metrics['mean'], metrics['min'], metrics['max']], alpha=0.7)
            plt.ylabel('MSE')
            plt.title('State Prediction Error Statistics')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'state_prediction_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_reward_prediction_metrics(self, reward_results: Dict, save_dir: str):
        """Plot reward prediction metrics."""
        reward_types = []
        mean_errors = []
        
        for metric, values in reward_results.items():
            if metric.endswith('_mse'):
                reward_type = metric.replace('reward_', '').replace('_mse', '').replace('_', ' ').title()
                reward_types.append(reward_type)
                mean_errors.append(values['mean'])
        
        if reward_types:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(reward_types, mean_errors, alpha=0.7)
            plt.ylabel('MSE')
            plt.title('Reward Prediction Errors by Type')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, error in zip(bars, mean_errors):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_errors)*0.01,
                        f'{error:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'reward_prediction_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def evaluate_both_modes(self,
                           test_loaders: Dict[str, Any],
                           save_results: bool = True,
                           save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Evaluate both supervised and RL modes and create a comparison.
        
        Args:
            test_loaders: Dictionary of test data loaders
            save_results: Whether to save results to disk
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing results for both modes
        """
        self.logger.info("Starting comprehensive evaluation of both modes")
        
        # Evaluate supervised mode
        supervised_results = self.evaluate_supervised_mode(test_loaders, save_results=False)
        
        # Evaluate RL mode
        rl_results = self.evaluate_rl_mode(test_loaders, save_results=False)
        
        # Combine results
        combined_results = {
            'supervised': supervised_results,
            'rl': rl_results,
            'comparison': self._create_mode_comparison(supervised_results, rl_results)
        }
        
        # Save combined results
        if save_results:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save individual results
            self._save_supervised_results(supervised_results, save_dir)
            self._save_rl_results(rl_results, save_dir)
            
            # Save combined results
            with open(os.path.join(save_dir, 'combined_results.json'), 'w') as f:
                json.dump(combined_results, f, indent=2, default=str)
            
            # Create comparison visualizations
            self._create_comparison_visualizations(combined_results, save_dir)
        
        self.logger.info("Comprehensive evaluation completed")
        return combined_results
    
    def _create_mode_comparison(self, supervised_results: Dict, rl_results: Dict) -> Dict:
        """Create comparison between supervised and RL modes."""
        comparison = {}
        
        # Compare state prediction performance
        supervised_state_mse = supervised_results.get('state_prediction', {}).get('single_step_state_mse', {}).get('mean', float('inf'))
        rl_state_mse = rl_results.get('state_prediction', {}).get('state_mse', {}).get('mean', float('inf'))
        
        comparison['state_prediction'] = {
            'supervised_mse': supervised_state_mse,
            'rl_mse': rl_state_mse,
            'better_mode': 'supervised' if supervised_state_mse < rl_state_mse else 'rl',
            'improvement': abs(supervised_state_mse - rl_state_mse) / max(supervised_state_mse, rl_state_mse, 1e-8)
        }
        
        # Add action prediction performance (only available in supervised mode)
        if 'action_prediction' in supervised_results:
            comparison['action_prediction'] = {
                'available_in': 'supervised_only',
                'performance': supervised_results['action_prediction']
            }
        
        # Add reward prediction performance (only available in RL mode)
        if 'reward_prediction' in rl_results:
            comparison['reward_prediction'] = {
                'available_in': 'rl_only',
                'performance': rl_results['reward_prediction']
            }
        
        return comparison
    
    def _create_comparison_visualizations(self, combined_results: Dict, save_dir: str):
        """Create comparison visualizations between modes."""
        comparison = combined_results.get('comparison', {})
        
        if 'state_prediction' in comparison:
            # State prediction comparison
            modes = ['Supervised', 'RL']
            mse_values = [
                comparison['state_prediction']['supervised_mse'],
                comparison['state_prediction']['rl_mse']
            ]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(modes, mse_values, alpha=0.7, color=['blue', 'orange'])
            plt.ylabel('State Prediction MSE')
            plt.title('State Prediction Performance Comparison')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, mse_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                        f'{value:.4f}', ha='center', va='bottom')
            
            # Highlight better mode
            better_mode_idx = 0 if comparison['state_prediction']['better_mode'] == 'supervised' else 1
            bars[better_mode_idx].set_color('green')
            bars[better_mode_idx].set_alpha(0.8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'mode_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Comparison visualizations saved to {save_dir}")