import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt


def calculate_map(predictions, targets, classes=None):
    """
    Calculate Mean Average Precision (mAP) for multi-label action prediction.
    
    Args:
        predictions: Tensor of prediction scores [batch_size, sequence_length, num_classes]
        targets: Tensor of binary target labels [batch_size, sequence_length, num_classes]
        classes: Optional list of class names for detailed reporting
        
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
            
            # Skip frames with no active classes if needed
            if np.sum(y_true) == 0:
                continue
            
            # Calculate AP for this frame (across all classes)
            try:
                ap_score = average_precision_score(y_true, y_pred)
                all_ap_scores.append(ap_score)
            except Exception as e:
                # Handle potential issues with all-zero predictions/targets
                pass
            
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
        overall_map = np.mean(all_ap_scores)
    else:
        overall_map = 0.0
    
    # Prepare results
    results = {
        'mAP': overall_map
    }
    
    # Add per-class results if class names provided
    if classes:
        per_class_map = {}
        for c in range(min(num_classes, len(classes))):
            if class_ap_scores[c]:
                per_class_map[classes[c]] = np.mean(class_ap_scores[c])
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
            avg_results['mAP'][key] = 0.0
    
    for key, values in accuracy_results.items():
        if values:
            avg_results['accuracy'][key] = sum(values) / len(values)
        else:
            avg_results['accuracy'][key] = 0.0
    
    # Print results
    print("\nAction Prediction Evaluation:")
    print("-" * 60)
    print(f"{'Horizon':<10} {'Metric':<15} {'Value':<10}")
    print("-" * 60)
    
    for horizon in sorted(eval_horizons):
        # Print mAP
        map_key = f"horizon_{horizon}_mAP"
        print(f"{horizon:<10} {'mAP':<15} {avg_results['mAP'].get(map_key, 0.0):.4f}")
        
        # Print top-k accuracy
        for k in sorted(top_ks):
            acc_key = f"horizon_{horizon}_top_{k}"
            print(f"{horizon:<10} {'Top-'+str(k)+' Accuracy':<15} {avg_results['accuracy'].get(acc_key, 0.0):.4f}")
        
        # Add separator between horizons
        print("-" * 60)
    
    return avg_results


def plot_map_vs_accuracy(results, save_dir, experiment_name):
    """
    Create plots comparing mAP and accuracy metrics.
    
    Args:
        results: Dictionary containing evaluation results
        save_dir: Directory to save plots
        experiment_name: Name for the experiment
        
    Returns:
        Dictionary with file paths
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract horizons
    horizons = set()
    for key in results['mAP'].keys():
        if key.startswith('horizon_'):
            horizon = int(key.split('_')[1])
            horizons.add(horizon)
    
    horizons = sorted(list(horizons))
    
    # Extract top-k values
    top_ks = set()
    for key in results['accuracy'].keys():
        if '_top_' in key:
            k = int(key.split('_top_')[1])
            top_ks.add(k)
    
    top_ks = sorted(list(top_ks))
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot mAP
    map_values = [results['mAP'][f'horizon_{h}_mAP'] for h in horizons]
    plt.plot(horizons, map_values, 'o-', linewidth=2, label='mAP')
    
    # Plot accuracy for each top-k
    for k in top_ks:
        acc_values = [results['accuracy'][f'horizon_{h}_top_{k}'] for h in horizons]
        plt.plot(horizons, acc_values, 's--', linewidth=2, label=f'Top-{k} Accuracy')
    
    plt.title('mAP vs Accuracy across Prediction Horizons', fontsize=16)
    plt.xlabel('Prediction Horizon', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(horizons)
    plt.ylim(0, 1.0)
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{experiment_name}_map_vs_accuracy.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'comparison_plot': filepath}


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


# Example integration with training function
def update_train_next_frame_model(cfg, model, train_loader, val_loader=None, device='cuda'):
    """
    Updated training function to also calculate and log mAP.
    
    This version adds Mean Average Precision evaluation alongside the existing accuracy metric.
    """
    # [Keep existing code...]
    
    # During validation in the epoch loop:
    with torch.no_grad():
        for batch_idx, (z_seq, _z_seq, _a_seq, f_a_seq) in enumerate(val_loader):
            # [Keep existing processing...]
            
            # Calculate mAP for each horizon
            for h in eval_horizons:
                if h <= outputs['f_a_seq_hat'].shape[1]:
                    f_a_h_hat = outputs['f_a_seq_hat'][:, :h, :]
                    f_a_h_targets = f_a_seq[:, :h, :]
                    
                    # Convert logits to probabilities
                    f_a_h_probs = torch.sigmoid(f_a_h_hat)
                    
                    # Calculate mAP
                    map_score = calculate_map(f_a_h_probs, f_a_h_targets)
                    results[f"horizon_{h}_mAP"] = map_score['mAP']
                    
                    # Log mAP
                    writer.add_scalar(f'Metrics/mAP_Horizon_{h}', map_score['mAP'], global_step)
                    
                    # Continue with existing accuracy calculation...
                    
    # At the end of validation, print mAP scores alongside accuracy
    print("\nAction Prediction Metrics:")
    print("-" * 60)
    print(f"{'Horizon':<10} {'Metric':<15} {'Value':<10}")
    print("-" * 60)
    
    for horizon in sorted(eval_horizons):
        # Print mAP
        map_key = f"horizon_{horizon}_mAP"
        if map_key in results:
            print(f"{horizon:<10} {'mAP':<15} {results[map_key]:.4f}")
        
        # Print top-k accuracy
        for k in sorted(top_ks):
            acc_key = f"horizon_{horizon}_top_{k}"
            if acc_key in results and results[acc_key]:
                avg_accuracy = sum(results[acc_key]) / len(results[acc_key])
                print(f"{horizon:<10} {'Top-'+str(k)+' Accuracy':<15} {avg_accuracy:.4f}")
    
    # Additional plots comparing mAP and accuracy
    plot_map_vs_accuracy(results, plots_save_dir, experiment_name)
    
    # [Continue with existing code...]