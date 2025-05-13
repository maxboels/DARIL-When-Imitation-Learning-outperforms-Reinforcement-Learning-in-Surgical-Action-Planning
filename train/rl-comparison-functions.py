# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file   rl-comparison-functions.py
"""
import os

def compare_rl_and_autoregressive(cfg, world_model, policy_model, test_video_loaders, device, logger):
    """
    Compare RL-based policy with auto-regressive action prediction.
    
    This function directly compares the performance of your trained RL policy against
    the auto-regressive action prediction from the world model.
    
    Args:
        cfg: Configuration dictionary
        world_model: The trained world model
        policy_model: The trained RL policy model
        test_video_loaders: Dictionary of DataLoaders for test videos
        device: Device to evaluate on
        logger: Logger instance
        
    Returns:
        Dictionary of comparison results
    """
    import torch
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"comparison_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Comparing RL policy with auto-regressive prediction, saving to {results_dir}")
    
    # Extract evaluation config
    eval_config = cfg['evaluation']['world_model']
    
    # Initialize metrics
    metrics = {
        'autoregressive': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mAP': 0.0,
            'per_class': [],
            'per_video': {}
        },
        'rl_policy': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mAP': 0.0,
            'per_class': [],
            'per_video': {}
        },
        'difference': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mAP': 0.0
        },
        'per_video_comparison': {}
    }
    
    # Count for statistics
    num_videos = 0
    total_samples = 0
    
    # Set models to evaluation mode
    world_model.eval()
    policy_model.eval()
    
    # Process each video
    for video_id, data_loader in test_video_loaders.items():
        logger.info(f"Comparing models on video {video_id}")
        
        # Initialize video metrics
        video_metrics = {
            'autoregressive': {
                'predictions': [],
                'accuracy': 0.0
            },
            'rl_policy': {
                'predictions': [],
                'accuracy': 0.0
            },
            'ground_truth': [],
            'num_samples': 0
        }
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                current_states = batch['current_states'].to(device)
                next_actions = batch['next_actions'].to(device)
                
                # 1. Get auto-regressive predictions from world model
                outputs_ar = world_model(current_states=current_states)
                
                # Extract action predictions
                if '_a_hat' in outputs_ar:
                    action_probs_ar = torch.sigmoid(outputs_ar['_a_hat'])
                    action_preds_ar = (action_probs_ar > 0.5).float()
                    video_metrics['autoregressive']['predictions'].append(action_preds_ar.cpu().numpy())
                else:
                    continue  # Skip if no action predictions
                
                # 2. Get RL policy predictions
                # Extract state representation (last state in context window)
                states = current_states[:, -1]  # [batch_size, embedding_dim]
                
                # Get actions from policy model
                action_probs_rl = policy_model(states)
                action_preds_rl = (action_probs_rl > 0.5).float()
                video_metrics['rl_policy']['predictions'].append(action_preds_rl.cpu().numpy())
                
                # 3. Store ground truth
                video_metrics['ground_truth'].append(next_actions[:, 0].cpu().numpy())  # First action in sequence
                video_metrics['num_samples'] += current_states.size(0)
        
        # Process video results
        if video_metrics['autoregressive']['predictions'] and video_metrics['rl_policy']['predictions']:
            # Concatenate all predictions and ground truth for this video
            auto_preds = np.vstack(video_metrics['autoregressive']['predictions'])
            rl_preds = np.vstack(video_metrics['rl_policy']['predictions'])
            ground_truth = np.vstack(video_metrics['ground_truth'])
            
            # Calculate action metrics for both models
            auto_metrics = calculate_action_metrics(auto_preds, ground_truth)
            rl_metrics = calculate_action_metrics(rl_preds, ground_truth)
            
            # Store metrics for this video
            metrics['autoregressive']['per_video'][video_id] = auto_metrics
            metrics['rl_policy']['per_video'][video_id] = rl_metrics
            
            # Calculate improvement
            metrics['per_video_comparison'][video_id] = {
                'accuracy_diff': float(rl_metrics['accuracy'] - auto_metrics['accuracy']),
                'precision_diff': float(rl_metrics['precision'] - auto_metrics['precision']),
                'recall_diff': float(rl_metrics['recall'] - auto_metrics['recall']),
                'f1_diff': float(rl_metrics['f1'] - auto_metrics['f1']),
                'mAP_diff': float(rl_metrics['mAP'] - auto_metrics['mAP']),
                'win': 'rl' if rl_metrics['accuracy'] > auto_metrics['accuracy'] else 'auto' if auto_metrics['accuracy'] > rl_metrics['accuracy'] else 'tie'
            }
            
            # Update statistics
            num_videos += 1
            total_samples += ground_truth.shape[0]
            
            # Log results
            logger.info(f"Video {video_id} - Autoregressive: Acc={auto_metrics['accuracy']:.4f}, F1={auto_metrics['f1']:.4f}, mAP={auto_metrics['mAP']:.4f}")
            logger.info(f"Video {video_id} - RL Policy: Acc={rl_metrics['accuracy']:.4f}, F1={rl_metrics['f1']:.4f}, mAP={rl_metrics['mAP']:.4f}")
            logger.info(f"Video {video_id} - Improvement: Acc={rl_metrics['accuracy']-auto_metrics['accuracy']:.4f}, F1={rl_metrics['f1']-auto_metrics['f1']:.4f}")
    
    # Calculate overall metrics (weighted by number of samples)
    if metrics['autoregressive']['per_video'] and metrics['rl_policy']['per_video']:
        calculate_overall_metrics(metrics['autoregressive'])
        calculate_overall_metrics(metrics['rl_policy'])
        
        # Calculate difference
        for metric_key in ['accuracy', 'precision', 'recall', 'f1', 'mAP']:
            metrics['difference'][metric_key] = metrics['rl_policy']['overall'][metric_key] - metrics['autoregressive']['overall'][metric_key]
        
        # Log overall results
        logger.info("\nOverall Comparison Results:")
        logger.info(f"Autoregressive: Acc={metrics['autoregressive']['overall']['accuracy']:.4f}, "
                   f"F1={metrics['autoregressive']['overall']['f1']:.4f}, mAP={metrics['autoregressive']['overall']['mAP']:.4f}")
        logger.info(f"RL Policy: Acc={metrics['rl_policy']['overall']['accuracy']:.4f}, "
                   f"F1={metrics['rl_policy']['overall']['f1']:.4f}, mAP={metrics['rl_policy']['overall']['mAP']:.4f}")
        logger.info(f"Improvement: Acc={metrics['difference']['accuracy']:.4f}, "
                   f"F1={metrics['difference']['f1']:.4f}, mAP={metrics['difference']['mAP']:.4f}")
        
        # Calculate win/tie/loss statistics
        wins = sum(1 for v in metrics['per_video_comparison'] if metrics['per_video_comparison'][v]['win'] == 'rl')
        ties = sum(1 for v in metrics['per_video_comparison'] if metrics['per_video_comparison'][v]['win'] == 'tie')
        losses = sum(1 for v in metrics['per_video_comparison'] if metrics['per_video_comparison'][v]['win'] == 'auto')
        
        metrics['win_stats'] = {
            'rl_wins': wins,
            'ties': ties,
            'auto_wins': losses,
            'win_rate': float(wins / num_videos) if num_videos > 0 else 0.0
        }
        
        logger.info(f"Win/Tie/Loss: {wins}/{ties}/{losses} "
                   f"(Win Rate: {metrics['win_stats']['win_rate']:.2%})")
    
    # Visualize comparison
    visualize_comparison(metrics, results_dir, logger)
    
    # Generate summary report
    generate_comparison_report(metrics, os.path.join(results_dir, 'comparison_summary.md'), logger)
    
    # Save results
    save_results_to_json(metrics, os.path.join(results_dir, 'comparison_results.json'))
    
    return metrics

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

def visualize_comparison(metrics, save_dir, logger):
    """
    Visualize comparison between autoregressive and RL policy.
    
    Args:
        metrics: Dictionary of comparison metrics
        save_dir: Directory to save visualizations
        logger: Logger instance
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    viz_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Overall metrics comparison
    plt.figure(figsize=(12, 6))
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'mAP']
    auto_values = [metrics['autoregressive']['overall'][m.lower()] for m in metric_names]
    rl_values = [metrics['rl_policy']['overall'][m.lower()] for m in metric_names]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    plt.bar(x - width/2, auto_values, width, label='Auto-regressive')
    plt.bar(x + width/2, rl_values, width, label='RL Policy')
    
    plt.ylabel('Score')
    plt.title('Comparison of Overall Metrics')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(viz_dir, 'overall_comparison.png'))
    plt.close()
    
    # 2. Improvement by metric
    plt.figure(figsize=(10, 6))
    
    diff_values = [metrics['difference'][m.lower()] for m in metric_names]
    
    colors = ['green' if v > 0 else 'red' for v in diff_values]
    
    plt.bar(metric_names, diff_values, color=colors)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.ylabel('Improvement (RL - Auto)')
    plt.title('RL Policy Improvement Over Auto-regressive')
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(viz_dir, 'improvement.png'))
    plt.close()
    
    # 3. Per-video accuracy comparison
    if 'per_video_comparison' in metrics and metrics['per_video_comparison']:
        plt.figure(figsize=(14, 8))
        
        videos = list(metrics['per_video_comparison'].keys())
        auto_acc = [metrics['autoregressive']['per_video'][v]['accuracy'] for v in videos]
        rl_acc = [metrics['rl_policy']['per_video'][v]['accuracy'] for v in videos]
        
        # Sort by improvement
        improvements = [rl_acc[i] - auto_acc[i] for i in range(len(videos))]
        sorted_indices = np.argsort(improvements)
        
        sorted_videos = [videos[i] for i in sorted_indices]
        sorted_auto = [auto_acc[i] for i in sorted_indices]
        sorted_rl = [rl_acc[i] for i in sorted_indices]
        
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(sorted_videos))
        width = 0.35
        
        plt.bar(x - width/2, sorted_auto, width, label='Auto-regressive')
        plt.bar(x + width/2, sorted_rl, width, label='RL Policy')
        
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison by Video (Sorted by Improvement)')
        plt.xticks(x, sorted_videos, rotation=90)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(viz_dir, 'per_video_accuracy.png'))
        plt.close()
        
        # 4. Win/Tie/Loss pie chart
        if 'win_stats' in metrics:
            plt.figure(figsize=(8, 8))
            
            labels = ['RL Wins', 'Ties', 'Auto-regressive Wins']
            sizes = [metrics['win_stats']['rl_wins'], 
                    metrics['win_stats']['ties'],
                    metrics['win_stats']['auto_wins']]
            colors = ['green', 'gray', 'red']
            explode = (0.1, 0, 0)  # explode the 1st slice (RL Wins)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Win/Tie/Loss Distribution')
            
            plt.savefig(os.path.join(viz_dir, 'win_distribution.png'))
            plt.close()
    
    logger.info(f"Comparison visualizations saved to {viz_dir}")

def generate_comparison_report(metrics, save_path, logger):
    """
    Generate a comparison report in Markdown format.
    
    Args:
        metrics: Dictionary of comparison metrics
        save_path: Path to save the report
        logger: Logger instance
    """
    import os
    
    # Generate markdown
    markdown = f"""# RL Policy vs. Auto-regressive Action Prediction Comparison

## Overall Results

| Metric | Auto-regressive | RL Policy | Improvement |
|--------|----------------|-----------|-------------|
| Accuracy | {metrics['autoregressive']['overall']['accuracy']:.4f} | {metrics['rl_policy']['overall']['accuracy']:.4f} | {metrics['difference']['accuracy']:+.4f} |
| Precision | {metrics['autoregressive']['overall']['precision']:.4f} | {metrics['rl_policy']['overall']['precision']:.4f} | {metrics['difference']['precision']:+.4f} |
| Recall | {metrics['autoregressive']['overall']['recall']:.4f} | {metrics['rl_policy']['overall']['recall']:.4f} | {metrics['difference']['recall']:+.4f} |
| F1 Score | {metrics['autoregressive']['overall']['f1']:.4f} | {metrics['rl_policy']['overall']['f1']:.4f} | {metrics['difference']['f1']:+.4f} |
| Mean AP | {metrics['autoregressive']['overall']['mAP']:.4f} | {metrics['rl_policy']['overall']['mAP']:.4f} | {metrics['difference']['mAP']:+.4f} |

## Win/Tie/Loss Statistics

"""
    
    if 'win_stats' in metrics:
        wins = metrics['win_stats']['rl_wins']
        ties = metrics['win_stats']['ties']
        losses = metrics['win_stats']['auto_wins']
        total = wins + ties + losses
        win_rate = metrics['win_stats']['win_rate']
        
        markdown += f"- **RL Policy Wins:** {wins}/{total} ({win_rate:.2%})\n"
        markdown += f"- **Ties:** {ties}/{total} ({ties/total:.2%})\n"
        markdown += f"- **Auto-regressive Wins:** {losses}/{total} ({losses/total:.2%})\n\n"
    
    markdown += f"""## Summary Analysis

The RL-based policy approach {"shows improvement" if metrics['difference']['accuracy'] > 0 else "does not show improvement"} over the auto-regressive prediction method, with an overall accuracy {"increase" if metrics['difference']['accuracy'] > 0 else "decrease"} of {abs(metrics['difference']['accuracy']):.4f}.

"""
    
    # Add detailed analysis based on metrics
    if metrics['difference']['accuracy'] > 0.01:  # Significant improvement
        markdown += """
The RL-based approach significantly outperforms the auto-regressive prediction model. This suggests that:

1. The reinforcement learning approach successfully learns to optimize beyond simple imitation
2. The reward function effectively guides the model toward better action selection
3. The policy is able to plan multi-step sequences more effectively
"""
    elif metrics['difference']['accuracy'] > 0:  # Slight improvement
        markdown += """
The RL-based approach shows modest improvements over the auto-regressive prediction model. This suggests that:

1. The reinforcement learning approach is learning something beyond simple imitation
2. There may be room for further optimization of the RL approach
3. Consider tuning the reward function or policy architecture for better performance
"""
    else:  # No improvement or worse
        markdown += """
The RL-based approach does not show improvement over the auto-regressive prediction model. This suggests that:

1. The current reward function may not effectively guide the policy toward better actions
2. The RL training process might need more data or iterations
3. The world model might have limitations that affect RL policy learning
4. The auto-regressive approach might already be performing optimally for this task
"""
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(markdown)
    
    logger.info(f"Comparison report saved to {save_path}")

def calculate_overall_metrics(metrics_dict):
    """
    Calculate overall metrics from per-video metrics, weighted by sample count.
    
    Args:
        metrics_dict: Dictionary with 'per_video' and 'overall' keys
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

def save_results_to_json(results, save_path):
    """
    Save results to JSON with numpy conversion.
    
    Args:
        results: Dictionary of results
        save_path: Path to save the results
    """
    import json
    import numpy as np
    
    # Define a function to convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    # Convert numpy types to Python types
    results_json = convert_numpy(results)
    
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(results_json, f, indent=2)
