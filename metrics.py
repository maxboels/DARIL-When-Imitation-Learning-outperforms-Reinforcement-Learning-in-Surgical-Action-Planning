import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt


def calculate_map(predictions, targets, class_names=None, skip_empty_frames=True, threshold=0.5, scale_factor=0.1):
    """
    Calculate Mean Average Precision (mAP) for multi-label action prediction.
    
    Args:
        predictions: Tensor of prediction scores [batch_size, sequence_length, num_classes]
        targets: Tensor of binary target labels [batch_size, sequence_length, num_classes]
        class_names: Optional list of class names for detailed reporting
        skip_empty_frames: Skip frames with no active classes (default: False), ignoring false positives
        threshold: Confidence threshold for considering a prediction as positive (default: 0.5)
        scale_factor: Penalty factor for false positives in empty frames (default
            0.1, meaning 1 false positive reduces AP by 0.1)        
    Returns:
        Dictionary containing mAP scores (overall and per-class if classes provided)
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    batch_size, seq_length, num_classes = predictions.shape
    
    # Initialize containers for AP scores
    all_ap_scores = []
    class_ap_scores = [[] for _ in range(num_classes)]
    
    # Process each sequence in the batch
    for b in range(batch_size):
        # Process each frame in the sequence
        for t in range(seq_length):
            y_pred = predictions[b, t]
            y_true = targets[b, t]
            
            # Do NOT skip frames with no active classes
            # Calculate AP for this frame (across all classes)
            try:
                # For frames with no active classes, we expect all zeros in prediction
                # If predictions contain positives, AP will be 0 (all false positives)
                if np.sum(y_true) == 0 and skip_empty_frames:
                    continue
                elif np.sum(y_true) == 0:
                    # Count false positives, but with reduced penalty
                    false_positive_count = np.sum(y_pred > threshold)
                    if false_positive_count > 0:
                        # Apply a scaled penalty based on confidence and number
                        ap_score = max(0, 1 - (false_positive_count * scale_factor))
                    else:
                        ap_score = 1.0  # Perfect score for correctly predicting no actions
                else:
                    ap_score = average_precision_score(y_true, y_pred)
                # Store AP score
                all_ap_scores.append(ap_score)
            except Exception as e:
                print(f"Error calculating AP: {e}")
                        
            # Calculate per-class AP scores
            for c in range(num_classes):
                # Only calculate for classes that appear in the dataset
                if np.any(targets[:, :, c] > 0):
                    try:
                        class_ap = average_precision_score(
                            targets[:, :, c].reshape(-1), 
                            predictions[:, :, c].reshape(-1)
                        )
                        class_ap_scores[c].append(class_ap)
                    except Exception as e:
                        pass
    
    # Calculate overall mAP
    if all_ap_scores:
        overall_map = np.nanmean(all_ap_scores)
    else:
        overall_map = np.nan # 0.0 # replace with None if needed or np.nan?
    
    # Prepare results
    results = {
        'mAP': overall_map
    }
    
    # Add per-class results if class names provided
    if class_names:
        per_class_map = {}
        for c in range(min(num_classes, len(class_names))):
            if class_ap_scores[c]:
                per_class_map[class_names[c]] = np.nanmean(class_ap_scores[c])
        results['per_class_mAP'] = per_class_map
    
    return results


def evaluate_action_prediction_map(model, val_loader, cfg, device='cuda'):
    """
    Evaluate model's action prediction using Mean Average Precision (mAP).
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        cfg: Configuration dictionary
        device: Device for evaluation
        
    Returns:
        Dictionary with evaluation results including mAP
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get evaluation parameters
    eval_horizons = cfg['eval']['world_model']['eval_horizons']
    max_horizon = cfg['eval']['world_model']['max_horizon']
    use_memory = cfg['eval']['world_model']['use_memory']
    
    # Get class names if available
    class_names = cfg.get('data', {}).get('class_names', None)
    
    # Initialize results
    map_results = {}
    for horizon in eval_horizons:
        map_results[f"horizon_{horizon}_mAP"] = []
    
    # Also track top-k accuracy for comparison
    accuracy_results = {}
    top_ks = cfg['eval']['world_model']['top_ks']
    for horizon in eval_horizons:
        for k in top_ks:
            accuracy_results[f"horizon_{horizon}_top_{k}"] = []
    
    # Run evaluation
    with torch.no_grad():
        for batch_idx, (z_seq, _z_seq, _a_seq, f_a_seq) in enumerate(tqdm(val_loader, desc="Evaluating with mAP")):
            z_seq, _z_seq, _a_seq, f_a_seq = z_seq.to(device), _z_seq.to(device), _a_seq.to(device), f_a_seq.to(device)
            
            # Generate predictions
            outputs = model.generate(z_seq, horizon=max_horizon, use_memory=use_memory)
            
            # Skip if no action predictions
            if 'f_a_seq_hat' not in outputs:
                continue
                
            # For each evaluation horizon
            for h in eval_horizons:
                if h <= outputs['f_a_seq_hat'].shape[1]:  # Check horizon is within range
                    # Get predictions and targets
                    f_a_h_hat = outputs['f_a_seq_hat'][:, :h, :]
                    f_a_h_targets = f_a_seq[:, :h, :]
                    
                    # Calculate mAP
                    # Convert logits to probabilities for mAP calculation
                    f_a_h_probs = torch.sigmoid(f_a_h_hat)
                    map_score = calculate_map(f_a_h_probs, f_a_h_targets, class_names)
                    map_results[f"horizon_{h}_mAP"].append(map_score['mAP'])
                    
                    # Also calculate top-k accuracy for comparison
                    for k in top_ks:
                        k = min(k, f_a_h_hat.shape[2])
                        
                        # Get top-k indices
                        _, topk_indices = torch.topk(f_a_h_probs, k, dim=2)
                        
                        # Count correct predictions
                        batch_size, horizon_len = f_a_h_targets.shape[0], f_a_h_targets.shape[1]
                        correct_count = 0
                        total_count = 0
                        
                        # Process each batch and frame
                        for b in range(batch_size):
                            for t in range(horizon_len):
                                # Get active classes
                                true_action_indices = torch.where(f_a_h_targets[b, t] > 0.5)[0]
                                
                                # Skip frames with no active classes
                                # TODO: Add option to skip or count as false positives
                                if len(true_action_indices) == 0:
                                    continue
                                    
                                # Get predictions
                                pred_action_indices = topk_indices[b, t]
                                
                                # Check for matches
                                match_found = False
                                for true_idx in true_action_indices:
                                    if true_idx in pred_action_indices:
                                        match_found = True
                                        break
                                        
                                if match_found:
                                    correct_count += 1
                                total_count += 1
                        
                        # Calculate accuracy
                        accuracy = correct_count / max(1, total_count)
                        accuracy_results[f"horizon_{h}_top_{k}"].append(accuracy)
    
    # Calculate average metrics
    avg_results = {
        'mAP': {},
        'accuracy': {}
    }
    
    for key, values in map_results.items():
        if values:
            avg_results['mAP'][key] = sum(values) / len(values)
        else:
            avg_results['mAP'][key] = np.nan
    
    for key, values in accuracy_results.items():
        if values:
            avg_results['accuracy'][key] = sum(values) / len(values)
        else:
            avg_results['accuracy'][key] = np.nan
        
    return avg_results

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_map_vs_accuracy(results, save_dir, experiment_name):
    """
    Create plots comparing mAP and accuracy metrics for a single horizon.
    
    Args:
        results: Dictionary containing evaluation results with keys like 'horizon_X_top_Y' and 'horizon_X_mAP'
        save_dir: Directory to save plots
        experiment_name: Name for the experiment
        
    Returns:
        Path to saved figure
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract horizon value from the results keys
    horizon = None
    for key in results.keys():
        if key.startswith('horizon_'):
            parts = key.split('_')
            if len(parts) > 1:
                try:
                    horizon = int(parts[1])
                    break
                except ValueError:
                    continue
    
    if horizon is None:
        print("Warning: Could not determine horizon from results keys")
        horizon = "unknown"
    
    # Extract top-k values for this horizon
    top_ks = []
    accuracies = []
    map_value = None
    
    for key, value in results.items():
        if key == f"horizon_{horizon}_mAP":
            # Found mAP value
            if isinstance(value, list):
                map_value = sum(value) / len(value) if value else np.nan
            else:
                map_value = value
        elif key.startswith(f"horizon_{horizon}_top_"):
            try:
                k = int(key.split('_top_')[1])
                top_ks.append(k)
                
                # Get average if it's a list
                if isinstance(value, list):
                    accuracies.append(sum(value) / len(value) if value else np.nan)
                else:
                    accuracies.append(value)
            except (ValueError, IndexError):
                continue
    
    # Sort by top-k values
    if top_ks and accuracies:
        sorted_pairs = sorted(zip(top_ks, accuracies))
        top_ks, accuracies = zip(*sorted_pairs)
    
    # Create the comparison plot
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy for each top-k
    plt.bar(range(len(top_ks)), accuracies, width=0.4, label='Top-k Accuracy', color='skyblue')
    
    # Add mAP as a horizontal line if available
    if map_value is not None:
        plt.axhline(y=map_value, color='r', linestyle='-', linewidth=2, label=f'mAP: {map_value:.4f}')
    
    # Add labels and formatting
    plt.title(f'Action Prediction Metrics for Horizon {horizon}', fontsize=16)
    plt.xlabel('Top-k Value', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(range(len(top_ks)), [f"Top-{k}" for k in top_ks])
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(fontsize=12)
    plt.ylim(0, 1.1)  # Allow space for annotations
    
    # Add value annotations
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{experiment_name}_h{horizon}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_all_horizons_comparison(results, save_dir, experiment_name):
    """
    Create a multi-panel plot showing mAP vs accuracy across all horizons.
    
    Args:
        results: Dictionary with evaluation results
        save_dir: Directory to save plots
        experiment_name: Name for the experiment
        
    Returns:
        Path to saved figure
    """
    # Extract horizons from results
    horizons = set()
    for key in results.keys():
        if key.startswith("horizon_"):
            parts = key.split("_")
            if len(parts) > 1:
                try:
                    horizons.add(int(parts[1]))
                except ValueError:
                    continue
    
    horizons = sorted(list(horizons))
    
    # Extract top-k values
    top_ks = set()
    for key in results.keys():
        if "_top_" in key:
            parts = key.split("_top_")
            if len(parts) > 1:
                try:
                    top_ks.add(int(parts[1]))
                except ValueError:
                    continue
    
    top_ks = sorted(list(top_ks))
    
    # Create multi-panel plot
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 6), sharey=True)
    
    # Handle case of single horizon
    if len(horizons) == 1:
        axes = [axes]
    
    # Process each horizon
    for i, horizon in enumerate(horizons):
        ax = axes[i]
        
        # Get mAP for this horizon
        map_key = f"horizon_{horizon}_mAP"
        map_value = None
        if map_key in results:
            if isinstance(results[map_key], list):
                map_value = sum(results[map_key]) / len(results[map_key]) if results[map_key] else np.nan
            else:
                map_value = results[map_key]
        
        # Get accuracy values for each top-k
        acc_values = []
        for k in top_ks:
            key = f"horizon_{horizon}_top_{k}"
            if key in results:
                if isinstance(results[key], list):
                    acc_values.append(sum(results[key]) / len(results[key]) if results[key] else np.nan)
                else:
                    acc_values.append(results[key])
            else:
                acc_values.append(np.nan)
        
        # Plot accuracy bars
        bars = ax.bar(range(len(top_ks)), acc_values, width=0.6, color='skyblue')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        # Add mAP line if available
        if map_value is not None:
            ax.axhline(y=map_value, color='r', linestyle='-', linewidth=2, 
                      label=f'mAP: {map_value:.4f}')
            # Add mAP value text
            ax.text(len(top_ks)/2 - 0.5, map_value + 0.03, f'mAP: {map_value:.4f}', 
                   color='red', fontsize=10, ha='center')
        
        ax.set_title(f'Horizon = {horizon}', fontsize=14)
        ax.set_xticks(range(len(top_ks)))
        ax.set_xticklabels([f'Top-{k}' for k in top_ks])
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        if i == 0:
            ax.set_ylabel('Accuracy / mAP', fontsize=14)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.05)
    
    plt.suptitle('Action Prediction: mAP vs Top-k Accuracy by Horizon', fontsize=16)
    plt.tight_layout()
    
    # Save combined figure
    filename = f"{experiment_name}_all_horizons.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_metrics_by_horizon(results, save_dir, experiment_name):
    """
    Create a line plot showing how metrics change with increasing horizon.
    
    Args:
        results: Dictionary with evaluation results
        save_dir: Directory to save plots
        experiment_name: Name for the experiment
        
    Returns:
        Path to saved figure
    """
    # Extract horizons
    horizons = set()
    map_values = {}
    accuracy_values = {}
    
    for key in results.keys():
        if key.startswith("horizon_"):
            parts = key.split("_")
            if len(parts) > 1:
                try:
                    horizon = int(parts[1])
                    horizons.add(horizon)
                    
                    # Store mAP values
                    if key.endswith("_mAP"):
                        if isinstance(results[key], list):
                            map_values[horizon] = sum(results[key]) / len(results[key]) if results[key] else np.nan
                        else:
                            map_values[horizon] = results[key]
                    
                    # Store top-k accuracy values
                    elif "_top_" in key:
                        k = int(key.split("_top_")[1])
                        if k not in accuracy_values:
                            accuracy_values[k] = {}
                        
                        if isinstance(results[key], list):
                            accuracy_values[k][horizon] = sum(results[key]) / len(results[key]) if results[key] else np.nan
                        else:
                            accuracy_values[k][horizon] = results[key]
                            
                except ValueError:
                    continue
    
    horizons = sorted(list(horizons))
    top_ks = sorted(list(accuracy_values.keys()))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot mAP line
    if map_values:
        map_line_values = [map_values.get(h, 0.0) for h in horizons]
        plt.plot(horizons, map_line_values, 'o-', linewidth=3, color='red', 
                label='mAP', markersize=8)
    
    # Plot accuracy lines for each top-k
    for k in top_ks:
        acc_line_values = [accuracy_values[k].get(h, 0.0) for h in horizons]
        plt.plot(horizons, acc_line_values, 's--', linewidth=2, 
                label=f'Top-{k} Accuracy')
    
    plt.title('Action Prediction Performance Across Horizons', fontsize=16)
    plt.xlabel('Prediction Horizon', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    plt.xticks(horizons)
    plt.ylim(0, 1.05)
    
    # Add data point values
    if map_values:
        for h in horizons:
            if h in map_values:
                plt.text(h, map_values[h] + 0.02, f'{map_values[h]:.3f}', 
                        ha='center', va='bottom', fontsize=9, color='red')
    
    for k in top_ks:
        for h in horizons:
            if h in accuracy_values[k]:
                plt.text(h, accuracy_values[k][h] - 0.04, f'{accuracy_values[k][h]:.3f}', 
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{experiment_name}_metrics_by_horizon.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


# Integration function for the training loop
def generate_map_vs_accuracy_plots(results, save_dir, experiment_name):
    """
    Generate all types of mAP vs accuracy plots.
    
    Args:
        results: Results dictionary with horizon and top-k keys
        save_dir: Directory to save plots
        experiment_name: Base name for the experiment
        
    Returns:
        Dictionary with paths to generated plots
    """
    plot_paths = {}
    
    # Create individual horizon plots
    for horizon in set([int(k.split('_')[1]) for k in results.keys() if k.startswith('horizon_')]):
        # Filter results for this horizon
        horizon_results = {k: v for k, v in results.items() if k.startswith(f'horizon_{horizon}_')}
        plot_name = f"{experiment_name}_horizon_{horizon}"
        plot_paths[f'horizon_{horizon}'] = plot_map_vs_accuracy(horizon_results, save_dir, plot_name)
    
    # Create all-horizons comparison
    plot_paths['all_horizons'] = plot_all_horizons_comparison(results, save_dir, experiment_name)
    
    # Create metrics by horizon plot
    plot_paths['metrics_by_horizon'] = plot_metrics_by_horizon(results, save_dir, experiment_name)
    
    return plot_paths


def generate_precision_recall_curves(model, val_loader, cfg, device='cuda', num_classes=10):
    """
    Generate precision-recall curves for a subset of action classes.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        cfg: Configuration dictionary
        device: Device to use
        num_classes: Number of classes to visualize (will select most frequent)
        
    Returns:
        Path to saved plot
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from tqdm import tqdm
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get class names if available
    class_names = cfg.get('data', {}).get('class_names', None)
    if class_names is None:
        # Generate generic class names
        class_names = [f"Class {i}" for i in range(num_classes)]
    else:
        # Limit to specified number of classes
        class_names = class_names[:num_classes]
    
    # Initialize containers for prediction scores and true labels
    all_predictions = []
    all_targets = []
    
    # Run forward pass to collect predictions
    with torch.no_grad():
        for batch_idx, (z_seq, _z_seq, _a_seq, f_a_seq) in enumerate(tqdm(val_loader, desc="Collecting PR curve data")):
            z_seq, f_a_seq = z_seq.to(device), f_a_seq.to(device)
            
            # Generate predictions
            outputs = model.generate(z_seq, horizon=1, use_memory=False)
            
            if 'f_a_seq_hat' not in outputs:
                continue
                
            # Get first-step predictions only for simplicity
            predictions = torch.sigmoid(outputs['f_a_seq_hat'][:, 0, :num_classes])
            targets = f_a_seq[:, 0, :num_classes]
            
            # Collect data
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate data
    if all_predictions and all_targets:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
    else:
        print("No predictions collected!")
        return None
    
    # Create precision-recall curve plot
    plt.figure(figsize=(12, 10))
    
    # Calculate class frequency
    class_counts = np.sum(all_targets, axis=0)
    most_frequent_indices = np.argsort(class_counts)[-num_classes:]
    
    # Generate a precision-recall curve for each selected class
    for i, class_idx in enumerate(most_frequent_indices):
        precision, recall, _ = precision_recall_curve(
            all_targets[:, class_idx], 
            all_predictions[:, class_idx]
        )
        
        ap = average_precision_score(
            all_targets[:, class_idx], 
            all_predictions[:, class_idx],
            average='macro'
        )
        
        plt.plot(recall, precision, lw=2, 
                 label=f'{class_names[class_idx]} (AP: {ap:.2f})')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves by Action Class', fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save plot
    save_dir = os.path.join(cfg['training']['log_dir'], cfg.get('experiment', {}).get('name', 'unnamed_experiment'))
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'precision_recall_curves.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def evaluate_multi_label_predictions(predictions, targets, top_ks=[1, 3, 5, 10]):
    """
    Comprehensive evaluation of multi-label action predictions with various metrics.
    
    Args:
        predictions: Prediction probabilities [batch_size, seq_length, num_classes]
        targets: Binary ground truth [batch_size, seq_length, num_classes]
        top_ks: List of k values for top-k metrics
        
    Returns:
        Dictionary with various evaluation metrics
    """
    import torch
    import numpy as np
    
    batch_size, seq_length, num_classes = predictions.shape
    results = {}
    
    # Initialize results dict
    for k in top_ks:
        results[f"top_{k}_any_match"] = []      # Any correct (current method)
        results[f"top_{k}_recall"] = []         # % of true actions found
        results[f"top_{k}_precision"] = []      # % of predicted actions that are correct
        results[f"top_{k}_f1"] = []             # Harmonic mean of precision and recall
        results[f"top_{k}_exact_match"] = []    # All true actions found (strict)
    
    # NOTE: Should we first select the predictions above 0.5 before using top-k?
    # I think k should only consider the positive predictions.
    # It would be strange to consider values that are close to zero as top-k predictions.

    # Get top-k predictions for each k
    top_k_indices = {}
    for k in top_ks:
        _, indices = torch.topk(predictions, min(k, num_classes), dim=2)
        top_k_indices[k] = indices
    
    # Calculate metrics for each frame
    for b in range(batch_size):
        for t in range(seq_length):
            # Get ground truth actions for this frame
            true_indices = torch.where(targets[b, t] > 0.5)[0]
            
            # Skip frames with no active classes if needed
            # TODO: Add option to skip or count as false positives (like we did for mAP)
            if len(true_indices) == 0:
                continue
            
            # Calculate metrics for each k
            for k in top_ks:
                pred_indices = top_k_indices[k][b, t]
                
                # 1. Any-match (original metric): frame is correct if ANY true action is found
                any_match = any(idx in pred_indices for idx in true_indices)
                results[f"top_{k}_any_match"].append(float(any_match))
                
                # 2. Recall: proportion of true actions found
                true_positives = sum(1 for idx in true_indices if idx in pred_indices)
                recall = true_positives / len(true_indices)
                results[f"top_{k}_recall"].append(recall)
                
                # 3. Precision: proportion of predictions that are correct
                precision = true_positives / len(pred_indices)
                results[f"top_{k}_precision"].append(precision)
                
                # 4. F1 score: harmonic mean of precision and recall
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                results[f"top_{k}_f1"].append(f1)
                
                # 5. Exact-match: frame is correct if ALL true actions are found
                exact_match = all(idx in pred_indices for idx in true_indices)
                results[f"top_{k}_exact_match"].append(float(exact_match))
    
    # Calculate averages
    averages = {}
    for key, values in results.items():
        if values:
            averages[key] = sum(values) / len(values)
        else:
            averages[key] = np.nan
    
    return averages


                            # # Top-K Accuracy over horizon
                            # for k in top_ks:
                            #     # Ensure k isn't larger than the number of classes
                            #     k = min(k, f_a_h_probs.shape[2])
                                
                            #     # Get top-k predictions for each frame
                            #     _, topk_indices = torch.topk(f_a_h_probs, k, dim=2)  # [batch, h, k]
                                
                            #     # Calculate accuracy - correct multi-label handling
                            #     batch_size, horizon_len = f_a_h_targets.shape[0], f_a_h_targets.shape[1]
                            #     correct_count = 0
                            #     total_count = 0
                                
                            #     # For each batch and time step
                            #     for b in range(batch_size):
                            #         for t in range(horizon_len):
                            #             # Get active classes (indices where value is 1) for this frame
                            #             true_action_indices = torch.where(f_a_h_targets[b, t] > 0.5)[0]
                                        
                            #             # Skip frames with no active classes
                            #             # TODO: is in not supposed to be part of the performance evaluation?
                            #             if len(true_action_indices) == 0:
                            #                 continue
                                            
                            #             # Get predicted top-k indices for this frame
                            #             pred_action_indices = topk_indices[b, t]
                                        
                            #             # Check if any true action is in the predicted top-k
                            #             match_found = False
                            #             for true_idx in true_action_indices:
                            #                 if true_idx in pred_action_indices:
                            #                     match_found = True
                            #                     break
                                                
                            #             if match_found:
                            #                 correct_count += 1
                            #             total_count += 1
                                
                            #     # Calculate accuracy
                            #     if total_count > 0:
                            #         accuracy = correct_count / max(1, total_count)  # Avoid division by zero
                            #         results[f"horizon_{h}_top_{k}"].append(accuracy)
                                
                            #         # Log accuracy
                            #         writer.add_scalar(f'Accuracy/Horizon_{h}_Top_{k}', accuracy, global_step)

def log_comprehensive_metrics(results, writer, epoch, logger):
    """
    Log comprehensive metrics to TensorBoard and console.
    
    Args:
        results: Dictionary with evaluation results
        writer: TensorBoard writer
        epoch: Current epoch
        logger: Logger for console output
    """
    # Extract horizons and metrics
    horizons = set()
    metrics = set()
    
    for key in results.keys():
        if key.startswith('horizon_'):
            parts = key.split('_')
            if len(parts) >= 3:
                horizons.add(int(parts[1]))
                metric_name = '_'.join(parts[2:])
                metrics.add(metric_name)
    
    # Sort horizons and metrics
    horizons = sorted(list(horizons))
    
    # Group metrics by type for better organization
    metric_groups = {
        'Accuracy': ['top_1_any_match', 'top_3_any_match', 'top_5_any_match', 'top_10_any_match',
                    'top_1_exact_match', 'top_3_exact_match', 'top_5_exact_match', 'top_10_exact_match'],
        'Recall': ['top_1_recall', 'top_3_recall', 'top_5_recall', 'top_10_recall'],
        'Precision': ['top_1_precision', 'top_3_precision', 'top_5_precision', 'top_10_precision'],
        'F1-Score': ['top_1_f1', 'top_3_f1', 'top_5_f1', 'top_10_f1'],
        'MAP': ['mAP']
    }
    
    # Log to TensorBoard
    for horizon in horizons:
        for group_name, metric_list in metric_groups.items():
            for metric in metric_list:
                key = f'horizon_{horizon}_{metric}'
                if key in results:
                    writer.add_scalar(f'Metrics/{group_name}/H{horizon}_{metric}', results[key], epoch)
    
    # Log to console with nice formatting
    logger.info("\nComprehensive Evaluation Results:")
    
    # Print loss values
    if 'val_loss' in results:
        logger.info(f"Validation Loss: {results['val_loss']:.4f}")
    if 'val_z_loss' in results:
        logger.info(f"Frame Prediction Loss: {results['val_z_loss']:.4f}")
    if 'val_a_loss' in results:
        logger.info(f"Action Prediction Loss: {results['val_a_loss']:.4f}")
    
    # Print metrics by horizon
    for horizon in horizons:
        logger.info(f"\n--- Prediction Horizon: {horizon} ---")
        
        # Print MAP if available
        if f'horizon_{horizon}_mAP' in results:
            logger.info(f"mAP: {results[f'horizon_{horizon}_mAP']:.4f}")
        
        # Print metrics by group
        logger.info("\nAccuracy Metrics:")
        logger.info(f"{'Metric':<20} {'Value':<10}")
        logger.info("-" * 30)
        
        for group_name, metric_list in metric_groups.items():
            if group_name != 'MAP':  # MAP already printed above
                metrics_in_group = [m for m in metric_list if f'horizon_{horizon}_{m}' in results]
                if metrics_in_group:
                    logger.info(f"\n{group_name} Metrics:")
                    logger.info(f"{'Metric':<20} {'Value':<10}")
                    logger.info("-" * 30)
                    
                    for metric in metrics_in_group:
                        key = f'horizon_{horizon}_{metric}'
                        logger.info(f"{metric:<20} {results[key]:.4f}")


def create_metrics_comparison_plot(results, save_path):
    """
    Create a comprehensive plot showing different metrics side by side.
    
    Args:
        results: Dictionary with evaluation results
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Extract horizons
    horizons = set()
    for key in results.keys():
        if key.startswith('horizon_'):
            parts = key.split('_')
            if len(parts) >= 3:
                horizons.add(int(parts[1]))
    
    horizons = sorted(list(horizons))
    
    # Define metrics to compare
    metrics_to_compare = [
        ('top_5_any_match', 'Any-Match (Top-5)', 'blue'),
        ('top_5_recall', 'Recall (Top-5)', 'green'),
        ('top_5_precision', 'Precision (Top-5)', 'orange'),
        ('top_5_f1', 'F1 Score (Top-5)', 'red'),
        ('top_5_exact_match', 'Exact-Match (Top-5)', 'purple'),
        ('mAP', 'mAP', 'brown')
    ]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each metric
    for metric_key, metric_name, color in metrics_to_compare:
        values = []
        for h in horizons:
            key = f'horizon_{h}_{metric_key}'
            if key in results:
                values.append(results[key])
            else:
                values.append(0)
        
        plt.plot(horizons, values, 'o-', color=color, label=metric_name, linewidth=2)
    
    # Add labels and formatting
    plt.xlabel('Prediction Horizon', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Comparison of Evaluation Metrics Across Horizons', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(horizons)
    plt.ylim(0, 1.05)
    
    # Save the plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_metric_breakdown_by_topk(results, save_path):
    """
    Create a plot showing how different k values affect performance.
    
    Args:
        results: Dictionary with evaluation results
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Extract horizons and top-k values
    horizons = set()
    top_ks = set()
    
    for key in results.keys():
        if key.startswith('horizon_'):
            parts = key.split('_')
            if len(parts) >= 4 and parts[2] == 'top':
                horizons.add(int(parts[1]))
                top_ks.add(int(parts[3]))
    
    horizons = sorted(list(horizons))
    top_ks = sorted(list(top_ks))
    
    # Create plot with subplots for each horizon
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 6), sharey=True)
    
    if len(horizons) == 1:
        axes = [axes]  # Ensure axes is iterable
    
    # Metrics to show
    metrics = ['any_match', 'recall', 'f1']
    colors = ['blue', 'green', 'red']
    
    # Plot data for each horizon
    for i, horizon in enumerate(horizons):
        ax = axes[i]
        
        # Plot each metric
        for j, metric in enumerate(metrics):
            values = []
            for k in top_ks:
                key = f'horizon_{horizon}_top_{k}_{metric}'
                if key in results:
                    values.append(results[key])
                else:
                    values.append(0)
            
            # Create line plot
            ax.plot(top_ks, values, 'o-', color=colors[j], label=metric.replace('_', ' ').title())
        
        # Add labels and formatting
        ax.set_title(f'Horizon {horizon}', fontsize=14)
        ax.set_xlabel('Top-k', fontsize=12)
        if i == 0:
            ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(top_ks)
        ax.set_ylim(0, 1.05)
    
    # Add legend to the first subplot
    axes[0].legend(fontsize=10)
    
    plt.suptitle('Effect of Top-k on Different Metrics', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path