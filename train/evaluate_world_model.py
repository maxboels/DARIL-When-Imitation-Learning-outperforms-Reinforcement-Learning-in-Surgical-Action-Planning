
"""
Enhanced Inference Evaluation for World Model
This module provides an enhanced evaluation function for the World Model, focusing on:
1. Class-based metrics for action prediction (mAP, precision, recall, F1)
2. Auto-regressive prediction for longer horizons
3. Visualization of results
4. Summary report generation
"""
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import get_video_loaders
from models import WorldModel
from utils import visualize_sample_rollout, weighted_average, generate_summary_report
from datasets import get_video_loaders


def enhanced_inference_evaluation(cfg, logger, model, test_video_loaders, device='cuda'):
    """
    Enhanced inference evaluation with class-based metrics and auto-regressive prediction.
    
    This function builds on the existing run_generation_inference but adds:
    1. Detailed action prediction metrics (mAP, precision, recall, F1)
    2. Longer horizon auto-regressive evaluation where model uses its own predictions
    3. Better visualization of results
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        model: WorldModel instance
        test_video_loaders: Dictionary of DataLoaders for test videos
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation results
    """
    import torch
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    from sklearn.decomposition import PCA
    
    model.eval()
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"enhanced_inference_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Running enhanced inference evaluation, saving to {results_dir}")
    
    # Extract evaluation config
    eval_config = cfg['evaluation']['world_model']
    horizons = eval_config.get('eval_horizons', [1, 3, 5, 10, 15])
    
    # Initialize metrics
    metrics = {
        'action': {
            'overall': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mAP': 0.0
            },
            'per_class': [],
            'per_video': {},
            'horizon': {h: {} for h in horizons}
        },
        'state': {
            'overall': {
                'mse': 0.0
            },
            'per_video': {},
            'horizon': {h: {} for h in horizons}
        },
        'rollout': {
            'overall': {
                'mean_error': 0.0,
                'growth_rate': 0.0
            },
            'per_video': {}
        }
    }
    
    # Process each video
    num_videos = len(test_video_loaders)
    for video_id, data_loader in test_video_loaders.items():
        logger.info(f"Evaluating video {video_id} with enhanced metrics")
        
        # Initialize video-specific metrics
        video_action_preds = []
        video_action_gts = []
        video_state_mse = []
        video_horizon_metrics = {h: {'action_preds': [], 'action_gts': [], 'state_mse': []} for h in horizons}
        
        for batch in data_loader:
            # Move data to device
            current_states = batch['current_states'].to(device)
            next_states = batch['next_states'].to(device)
            next_actions = batch['next_actions'].to(device)
            future_states = batch['future_states'].to(device)
            future_actions = batch['future_actions'].to(device)
            
            batch_size = current_states.size(0)
            
            with torch.no_grad():
                # 1. First step prediction (standard)
                outputs = model(current_states=current_states)
                
                # Extract action predictions
                if '_a_hat' in outputs:
                    action_probs = torch.sigmoid(outputs['_a_hat'])
                    action_preds = (action_probs > 0.5).float()
                    
                    # Store first step predictions and ground truth
                    video_action_preds.append(action_preds.cpu().numpy())
                    video_action_gts.append(next_actions[:, 0].cpu().numpy())
                
                # Extract state predictions
                if '_z_hat' in outputs:
                    next_state_preds = outputs['_z_hat']
                    next_state_gts = next_states[:, -1]  # Last state in context
                    
                    # Calculate MSE
                    mse = torch.mean((next_state_preds - next_state_gts) ** 2, dim=1)
                    video_state_mse.append(mse.cpu().numpy())
                
                # 2. Auto-regressive multi-horizon prediction
                # Use model's own predictions as inputs for future steps
                horizon_outputs = evaluate_auto_regressive_horizons(
                    model, current_states, future_states, future_actions, 
                    horizons, device
                )
                
                # Store horizon metrics
                for h in horizons:
                    if h in horizon_outputs:
                        video_horizon_metrics[h]['action_preds'].append(
                            horizon_outputs[h]['action_preds'].cpu().numpy())
                        video_horizon_metrics[h]['action_gts'].append(
                            future_actions[:, min(h-1, future_actions.size(1)-1)].cpu().numpy())
                        video_horizon_metrics[h]['state_mse'].append(
                            horizon_outputs[h]['state_mse'].cpu().numpy())
                
                # 3. Generate rollout trajectory for visualization (selected samples)
                if batch_size > 0:
                    for sample_idx in range(min(batch_size, 2)):  # Only process max 2 samples per batch
                        sample_states = current_states[sample_idx:sample_idx+1]
                        sample_future = future_states[sample_idx:sample_idx+1]
                        
                        # Generate rollout
                        rollout = model.generate_conditional_future_states(
                            input_embeddings=sample_states,
                            horizon=max(horizons),
                            temperature=0.7,
                            use_past=True
                        )
                        
                        # Visualize comparison between prediction and ground truth
                        visualize_sample_rollout(
                            rollout, sample_future,
                            os.path.join(results_dir, f"{video_id}_sample_{sample_idx}.png"),
                            logger, title=f"Video {video_id} - Sample {sample_idx}"
                        )
        
        # Process video metrics
        if video_action_preds and video_action_gts:
            # Concatenate all predictions and ground truth for this video
            video_action_preds = np.vstack(video_action_preds)
            video_action_gts = np.vstack(video_action_gts)
            
            # Calculate action metrics
            video_action_metrics = calculate_action_metrics(video_action_preds, video_action_gts)
            metrics['action']['per_video'][video_id] = video_action_metrics
            
            logger.info(f"Video {video_id} - Action Acc: {video_action_metrics['accuracy']:.4f}, "
                       f"mAP: {video_action_metrics['mAP']:.4f}")
        
        if video_state_mse:
            # Calculate average MSE for this video
            video_state_mse = np.concatenate(video_state_mse)
            video_avg_mse = np.mean(video_state_mse)
            
            metrics['state']['per_video'][video_id] = {
                'mse': float(video_avg_mse),
                'num_samples': len(video_state_mse)
            }
            
            logger.info(f"Video {video_id} - State MSE: {video_avg_mse:.4f}")
        
        # Process horizon metrics
        for h in horizons:
            if (video_horizon_metrics[h]['action_preds'] and 
                video_horizon_metrics[h]['action_gts'] and 
                video_horizon_metrics[h]['state_mse']):
                
                # Concatenate horizon metrics
                h_action_preds = np.vstack(video_horizon_metrics[h]['action_preds'])
                h_action_gts = np.vstack(video_horizon_metrics[h]['action_gts'])
                h_state_mse = np.concatenate(video_horizon_metrics[h]['state_mse'])
                
                # Calculate action metrics for this horizon
                h_action_metrics = calculate_action_metrics(h_action_preds, h_action_gts)
                metrics['action']['horizon'][h][video_id] = h_action_metrics
                
                # Calculate state MSE for this horizon
                h_avg_mse = np.mean(h_state_mse)
                metrics['state']['horizon'][h][video_id] = {
                    'mse': float(h_avg_mse),
                    'num_samples': len(h_state_mse)
                }
                
                logger.info(f"Video {video_id} - Horizon {h} - Action Acc: {h_action_metrics['accuracy']:.4f}, "
                           f"mAP: {h_action_metrics['mAP']:.4f}, State MSE: {h_avg_mse:.4f}")
    
    # Calculate overall metrics across videos
    # Action metrics
    if metrics['action']['per_video']:
        calculate_overall_metrics(metrics['action'])
        logger.info(f"Overall Action - Acc: {metrics['action']['overall']['accuracy']:.4f}, "
                   f"mAP: {metrics['action']['overall']['mAP']:.4f}")
    
    # State metrics
    if metrics['state']['per_video']:
        metrics['state']['overall']['mse'] = weighted_average(
            [metrics['state']['per_video'][v]['mse'] for v in metrics['state']['per_video']],
            [metrics['state']['per_video'][v]['num_samples'] for v in metrics['state']['per_video']]
        )
        logger.info(f"Overall State MSE: {metrics['state']['overall']['mse']:.4f}")
    
    # Horizon metrics
    for h in horizons:
        if metrics['action']['horizon'][h]:
            action_h_overall = {}
            calculate_overall_metrics({'per_video': metrics['action']['horizon'][h], 'overall': action_h_overall})
            metrics['action']['horizon'][h]['overall'] = action_h_overall
            
            state_h_overall = {'mse': weighted_average(
                [metrics['state']['horizon'][h][v]['mse'] for v in metrics['state']['horizon'][h]],
                [metrics['state']['horizon'][h][v]['num_samples'] for v in metrics['state']['horizon'][h]]
            )}
            metrics['state']['horizon'][h]['overall'] = state_h_overall
            
            logger.info(f"Horizon {h} - Overall Action Acc: {action_h_overall['accuracy']:.4f}, "
                       f"mAP: {action_h_overall['mAP']:.4f}, State MSE: {state_h_overall['mse']:.4f}")
    
    # Generate visualizations
    visualize_results(metrics, os.path.join(results_dir, "visualizations"), logger)
    
    # Generate summary report
    generate_summary_report(metrics, os.path.join(results_dir, "summary.md"), logger)
    
    return metrics

def evaluate_auto_regressive_horizons(model, current_states, future_states, future_actions, horizons, device):
    """
    Evaluate model predictions at different horizons using auto-regressive prediction.
    
    At each step, the model uses its own previous predictions as input.
    
    Args:
        model: WorldModel instance
        current_states: Current state tensor [batch_size, context_length, embedding_dim]
        future_states: Future state tensor [batch_size, future_length, embedding_dim]
        future_actions: Future action tensor [batch_size, future_length, action_dim]
        horizons: List of horizons to evaluate at
        device: Device to evaluate on
        
    Returns:
        Dictionary of horizon-specific metrics
    """
    import torch
    import numpy as np
    
    max_horizon = max(horizons)
    batch_size = current_states.size(0)
    
    # Initialize output dictionary
    horizon_outputs = {h: {} for h in horizons}
    
    # Initial state is the last frame in the context window
    state = current_states[:, -1:].clone()  # [batch_size, 1, embedding_dim]
    
    # Predictions and ground truth for each horizon
    all_action_preds = []
    all_state_preds = []
    
    # Auto-regressive rollout
    for step in range(max_horizon):
        # Forward pass
        with torch.no_grad():
            outputs = model(current_states=state)
        
        # Extract predictions
        action_probs = torch.sigmoid(outputs['_a_hat'])
        action_preds = (action_probs > 0.5).float()
        state_preds = outputs['_z_hat']
        
        # Store predictions
        all_action_preds.append(action_preds)
        all_state_preds.append(state_preds)
        
        # Use predictions as next input (auto-regressive)
        state = state_preds.unsqueeze(1)  # Add time dimension
        
        # Store metrics for horizons we care about
        horizon = step + 1
        if horizon in horizons:
            # Store action predictions
            horizon_outputs[horizon]['action_preds'] = action_preds
            
            # Calculate state prediction MSE
            if horizon < future_states.size(1):
                state_mse = torch.mean((state_preds - future_states[:, horizon-1]) ** 2, dim=1)
                horizon_outputs[horizon]['state_mse'] = state_mse
    
    return horizon_outputs

def calculate_action_metrics(predictions, ground_truth):
    """
    Calculate comprehensive metrics for action prediction.
    
    Args:
        predictions: Binary predictions [num_samples, num_classes]
        ground_truth: Binary ground truth [num_samples, num_classes]
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    import numpy as np
    
    # Calculate accuracy (exact match across all classes)
    exact_match = np.all(predictions == ground_truth, axis=1)
    accuracy = np.mean(exact_match)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth.flatten(), predictions.flatten(), average='binary'
    )
    
    # Calculate mAP for each action class
    ap_scores = []
    per_class_metrics = []
    
    for i in range(ground_truth.shape[1]):
        # Only calculate AP if class is present
        class_metrics = {}
        
        if len(np.unique(ground_truth[:, i])) > 1:
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                ground_truth[:, i], predictions[:, i], average='binary'
            )
            
            class_ap = average_precision_score(ground_truth[:, i], predictions[:, i])
            ap_scores.append(class_ap)
            
            class_metrics = {
                'precision': float(class_precision),
                'recall': float(class_recall),
                'f1': float(class_f1),
                'ap': float(class_ap),
                'support': int(np.sum(ground_truth[:, i]))
            }
        else:
            # If class is not present or only one class value exists
            class_metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ap': 0.0,
                'support': int(np.sum(ground_truth[:, i]))
            }
        
        per_class_metrics.append(class_metrics)
    
    # Overall mAP
    mAP = np.mean(ap_scores) if ap_scores else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mAP': float(mAP),
        'per_class': per_class_metrics,
        'num_samples': int(predictions.shape[0])
    }

def visualize_sample_rollout(rollout, ground_truth, save_path, logger, title=None):
    """
    Visualize a comparison between predicted and ground truth trajectories.
    
    Args:
        rollout: Dictionary with model rollout outputs
        ground_truth: Ground truth future states [batch_size, horizon, embedding_dim]
        save_path: Path to save visualization
        logger: Logger instance
        title: Optional title for the plot
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    
    # Extract predicted trajectory
    if 'full_embeddings' in rollout:
        pred_trajectory = rollout['full_embeddings'].cpu().numpy().squeeze(0)
        
        # Extract ground truth trajectory
        gt_trajectory = ground_truth.cpu().numpy().squeeze(0)
        
        # Only compare up to the minimum length
        min_length = min(pred_trajectory.shape[0], gt_trajectory.shape[0])
        pred_trajectory = pred_trajectory[:min_length]
        gt_trajectory = gt_trajectory[:min_length]
        
        # Apply PCA for visualization
        combined = np.vstack([pred_trajectory, gt_trajectory])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        
        # Split back into predicted and ground truth
        pred_pca = pca_result[:len(pred_trajectory)]
        gt_pca = pca_result[len(pred_trajectory):]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predicted trajectory
        plt.scatter(pred_pca[:, 0], pred_pca[:, 1], 
                   c=range(len(pred_pca)), cmap='viridis', 
                   s=100, marker='o', label='Predicted')
        
        # Connect with arrows
        for i in range(len(pred_pca) - 1):
            plt.arrow(pred_pca[i, 0], pred_pca[i, 1], 
                     pred_pca[i+1, 0] - pred_pca[i, 0], 
                     pred_pca[i+1, 1] - pred_pca[i, 1],
                     head_width=0.05, head_length=0.08, fc='blue', ec='blue', alpha=0.6)
        
        # Plot ground truth trajectory
        plt.scatter(gt_pca[:, 0], gt_pca[:, 1], 
                   c=range(len(gt_pca)), cmap='viridis', 
                   s=100, marker='x', label='Ground Truth')
        
        # Connect with arrows
        for i in range(len(gt_pca) - 1):
            plt.arrow(gt_pca[i, 0], gt_pca[i, 1], 
                     gt_pca[i+1, 0] - gt_pca[i, 0], 
                     gt_pca[i+1, 1] - gt_pca[i, 1],
                     head_width=0.05, head_length=0.08, fc='green', ec='green', alpha=0.6)
        
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved rollout visualization to {save_path}")

def visualize_results(metrics, save_dir, logger):
    """
    Create visualizations for the evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save visualizations
        logger: Logger instance
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Action metrics summary
    if 'action' in metrics and 'overall' in metrics['action']:
        plt.figure(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'mAP']
        metric_values = [
            metrics['action']['overall']['accuracy'],
            metrics['action']['overall']['precision'],
            metrics['action']['overall']['recall'],
            metrics['action']['overall']['f1'],
            metrics['action']['overall']['mAP']
        ]
        
        plt.bar(metric_names, metric_values)
        plt.ylim(0, 1)
        plt.title('Overall Action Prediction Metrics')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'action_metrics.png'))
        plt.close()
    
    # 2. Action metrics across horizons
    if 'action' in metrics and 'horizon' in metrics['action']:
        horizons = sorted(metrics['action']['horizon'].keys())
        
        metrics_to_plot = ['accuracy', 'mAP']
        for metric_name in metrics_to_plot:
            metric_values = [metrics['action']['horizon'][h]['overall'][metric_name] 
                            for h in horizons if 'overall' in metrics['action']['horizon'][h]]
            
            if metric_values:
                plt.figure(figsize=(10, 6))
                plt.plot(horizons[:len(metric_values)], metric_values, 'o-', linewidth=2, markersize=8)
                plt.xlabel('Prediction Horizon')
                plt.ylabel(metric_name.capitalize())
                plt.title(f'Action {metric_name.capitalize()} vs. Prediction Horizon')
                plt.grid(alpha=0.3)
                plt.ylim(0, 1)
                plt.xticks(horizons[:len(metric_values)])
                plt.savefig(os.path.join(save_dir, f'action_{metric_name}_vs_horizon.png'))
                plt.close()
    
    # 3. State MSE across horizons
    if 'state' in metrics and 'horizon' in metrics['state']:
        horizons = sorted(metrics['state']['horizon'].keys())
        
        mse_values = [metrics['state']['horizon'][h]['overall']['mse'] 
                     for h in horizons if 'overall' in metrics['state']['horizon'][h]]
        
        if mse_values:
            plt.figure(figsize=(10, 6))
            plt.plot(horizons[:len(mse_values)], mse_values, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Prediction Horizon')
            plt.ylabel('MSE')
            plt.title('State Prediction Error vs. Horizon')
            plt.grid(alpha=0.3)
            plt.xticks(horizons[:len(mse_values)])
            plt.savefig(os.path.join(save_dir, 'state_mse_vs_horizon.png'))
            plt.close()
    
    logger.info(f"Saved visualizations to {save_dir}")

def generate_summary_report(metrics, save_path, logger):
    """
    Generate a summary report of evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the report
        logger: Logger instance
    """
    import os
    
    # Create markdown report
    report = "# World Model Inference Evaluation Summary\n\n"
    
    # Add overall action metrics
    if 'action' in metrics and 'overall' in metrics['action']:
        report += "## Action Prediction Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Accuracy | {metrics['action']['overall']['accuracy']:.4f} |\n"
        report += f"| Precision | {metrics['action']['overall']['precision']:.4f} |\n"
        report += f"| Recall | {metrics['action']['overall']['recall']:.4f} |\n"
        report += f"| F1 Score | {metrics['action']['overall']['f1']:.4f} |\n"
        report += f"| Mean Average Precision (mAP) | {metrics['action']['overall']['mAP']:.4f} |\n\n"
    
    # Add state prediction metrics
    if 'state' in metrics and 'overall' in metrics['state']:
        report += "## State Prediction Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Mean Squared Error (MSE) | {metrics['state']['overall']['mse']:.4f} |\n\n"
    
    # Add horizon metrics
    if 'action' in metrics and 'horizon' in metrics['action']:
        horizons = sorted(metrics['action']['horizon'].keys())
        valid_horizons = [h for h in horizons if 'overall' in metrics['action']['horizon'][h]]
        
        if valid_horizons:
            report += "## Multi-Horizon Prediction Metrics\n\n"
            report += "| Horizon | Action Accuracy | Action mAP | State MSE |\n"
            report += "|---------|----------------|------------|----------|\n"
            
            for h in valid_horizons:
                action_acc = metrics['action']['horizon'][h]['overall']['accuracy']
                action_map = metrics['action']['horizon'][h]['overall']['mAP']
                state_mse = metrics['state']['horizon'][h]['overall']['mse']
                
                report += f"| {h} | {action_acc:.4f} | {action_map:.4f} | {state_mse:.4f} |\n"
            
            report += "\n"
    
    # Add analysis
    report += "## Analysis\n\n"
    
    # Action prediction analysis
    if 'action' in metrics and 'overall' in metrics['action']:
        mAP = metrics['action']['overall']['mAP']
        if mAP > 0.8:
            report += "The model demonstrates excellent action prediction performance with high mAP, "
            report += "suggesting strong accuracy in identifying the correct surgical actions.\n\n"
        elif mAP > 0.6:
            report += "The model shows good action prediction performance, but there's room for improvement "
            report += "in certain action classes.\n\n"
        else:
            report += "The model's action prediction performance could be improved. Consider further training "
            report += "or exploring different model architectures.\n\n"
    
    # Horizon analysis
    if 'action' in metrics and 'horizon' in metrics['action']:
        horizons = sorted(metrics['action']['horizon'].keys())
        valid_horizons = [h for h in horizons if 'overall' in metrics['action']['horizon'][h]]
        
        if valid_horizons and len(valid_horizons) > 1:
            first_h = valid_horizons[0]
            last_h = valid_horizons[-1]
            
            first_map = metrics['action']['horizon'][first_h]['overall']['mAP']
            last_map = metrics['action']['horizon'][last_h]['overall']['mAP']
            
            map_drop = first_map - last_map
            
            if map_drop < 0.1:
                report += "The model maintains consistent performance over longer prediction horizons, "
                report += "indicating strong temporal modeling capabilities.\n\n"
            elif map_drop < 0.3:
                report += "The model shows moderate degradation in performance over longer prediction horizons, "
                report += "which is expected for auto-regressive predictions.\n\n"
            else:
                report += "The model's performance degrades significantly over longer prediction horizons. "
                report += "This suggests accumulated errors in auto-regressive prediction. Consider techniques "
                report += "like scheduled sampling during training to improve long-horizon prediction.\n\n"
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved summary report to {save_path}")

def calculate_overall_metrics(metrics_dict):
    """
    Calculate overall metrics from per-video metrics, weighted by sample count.
    
    Args:
        metrics_dict: Dictionary with 'per_video' and 'overall' keys
        
    Updates the 'overall' dictionary in place.
    """
    if not metrics_dict['per_video']:
        return
    
    # Get all video IDs
    video_ids = list(metrics_dict['per_video'].keys())
    
    # Calculate total number of samples
    total_samples = sum(metrics_dict['per_video'][v]['num_samples'] for v in video_ids)
    
    if total_samples == 0:
        return
    
    # Initialize overall metrics
    metrics_dict['overall'] = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'mAP': 0.0
    }
    
    # Calculate weighted averages
    for metric in metrics_dict['overall'].keys():
        metrics_dict['overall'][metric] = weighted_average(
            [metrics_dict['per_video'][v][metric] for v in video_ids],
            [metrics_dict['per_video'][v]['num_samples'] for v in video_ids]
        )

def weighted_average(values, weights):
    """
    Calculate weighted average.
    
    Args:
        values: List of values
        weights: List of weights (same length as values)
        
    Returns:
        Weighted average as a float
    """
    import numpy as np
    
    if not values or not weights or sum(weights) == 0:
        return 0.0
    
    return float(np.average(values, weights=weights))
