import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def evaluate_action_prediction(model, train_dataloader, test_dataloader, cfg, device=None):
    """
    Evaluate model's action prediction accuracy using top-k metrics for different prediction horizons.
    
    This function:
    1. Creates an action classifier if not present in the model
    2. Generates future predictions for different horizons
    3. Evaluates action prediction accuracy using top-k metrics
    4. Returns and visualizes detailed results
    
    Args:
        model: Trained CausalGPT2ForFrameEmbeddings model
        data: List of video data dictionaries
        cfg: Configuration dictionary
        device: Device to run evaluation on (defaults to CUDA if available)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get evaluation parameters from config
    top_ks = cfg.get('eval', {}).get('action_prediction', {}).get('top_ks', [1, 3, 5, 10])
    horizons = cfg.get('eval', {}).get('action_prediction', {}).get('horizons', [1, 5, 10, 15])
    # batch_size = cfg.get('training', {}).get('batch_size', 16)
    
    # # Create dataset and dataloader
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize results dictionary
    results = {}
    for horizon in horizons:
        for k in top_ks:
            results[f"horizon_{horizon}_top_{k}"] = []

    # Evaluate action prediction
    with torch.no_grad():
        for batch_idx, (z_seq, _z_seq, _a_seq) in enumerate(tqdm(dataloader, desc="Evaluating action prediction")):
            z_seq, _z_seq, _a_seq = z_seq.to(device), _z_seq.to(device), _a_seq.to(device)
            
            # Process each horizon
            for horizon in horizons:
                try:
                    # Generate predictions
                    outputs = model.generate(
                        input_embeddings=z_seq,
                        horizon=horizon,
                        use_past=True
                    )
                    
                    # Extract hidden states
                    hidden_states = outputs.get("last_hidden_states")
                    
                    if hidden_states is None:
                        print(f"Warning: No hidden states found in model output for horizon {horizon}")
                        continue
                    
                    # For each step in the prediction horizon
                    for step in range(min(horizon, hidden_states.shape[1])):
                        # Get target step (adjust if needed based on dataset structure)
                        target_step = step
                        
                        # Ensure we have target data for this step
                        if target_step < _a_seq.shape[1]:
                            # Predict actions from hidden states
                            action_logits = model.action_head(hidden_states[:, step, :])
                            
                            # Get target actions
                            target_actions = _a_seq[:, target_step, :]
                            
                            # Convert one-hot encoded targets to indices
                            target_indices = torch.argmax(target_actions, dim=1)
                            
                            # Calculate top-k accuracy
                            for k in top_ks:
                                # Ensure k isn't larger than the number of classes
                                k = min(k, action_logits.shape[1])
                                
                                # Get top-k predictions
                                _, topk_indices = torch.topk(action_logits, k, dim=1)
                                
                                # Check if target is in top-k
                                correct = torch.any(topk_indices == target_indices.unsqueeze(1), dim=1)
                                
                                # Calculate and store accuracy
                                accuracy = correct.float().mean().item()
                                results[f"horizon_{horizon}_top_{k}"].append(accuracy)
                except Exception as e:
                    print(f"Error evaluating horizon {horizon}, batch {batch_idx}: {e}")
                    continue
    
    # Calculate average metrics
    avg_results = {}
    for key, values in results.items():
        if values:
            avg_results[key] = sum(values) / len(values)
        else:
            avg_results[key] = 0.0
    
    # Print results table
    print("\nAction Prediction Accuracy:")
    print("-" * 60)
    print(f"{'Horizon':<10} {'Top-k':<10} {'Accuracy':<10} {'Num. Samples':<15}")
    print("-" * 60)
    for horizon in sorted(horizons):
        for k in sorted(top_ks):
            key = f"horizon_{horizon}_top_{k}"
            samples = len(results[key])
            print(f"{horizon:<10} {k:<10} {avg_results[key]:.4f} {samples:<15}")
    print("-" * 60)
    
    # Visualize results
    plot_evaluation_results(avg_results, horizons, top_ks)
    
    return avg_results

def plot_evaluation_results(results, horizons, top_ks):
    """
    Plot the evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        horizons: List of horizon values
        top_ks: List of top-k values
    """
    plt.figure(figsize=(12, 6))
    
    # Plot lines for each top-k
    for k in sorted(top_ks):
        y_values = []
        for h in sorted(horizons):
            key = f"horizon_{h}_top_{k}"
            y_values.append(results.get(key, 0.0))
        
        plt.plot(sorted(horizons), y_values, marker='o', label=f'Top-{k}')
    
    plt.title('Action Prediction Accuracy by Horizon and Top-K')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'action_prediction_accuracy_{timestamp}.png')
    plt.show()

def run_evaluation_suite(model, train_data, val_data, cfg):
    """
    Run a comprehensive evaluation suite on the model.
    
    Args:
        model: Trained model
        train_data: Training data
        val_data: Validation data
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    print("Running evaluation suite...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure evaluation parameters are set
    if 'action_prediction' not in cfg.get('eval', {}):
        cfg.setdefault('eval', {})['action_prediction'] = {
            'top_ks': [1, 3, 5, 10],
            'horizons': [1, 3, 5, 10, 15]
        }
    
    # Run evaluation on validation data
    print("\nEvaluating on validation data:")
    val_results = evaluate_action_prediction(model, val_data, cfg, device)
    
    # Optional: Run evaluation on training data for comparison
    if cfg.get('eval', {}).get('evaluate_on_train', False):
        print("\nEvaluating on training data:")
        train_results = evaluate_action_prediction(model, train_data, cfg, device)
        
        # Compare training vs validation results
        compare_train_val_results(train_results, val_results, 
                                 cfg['eval']['action_prediction']['horizons'],
                                 cfg['eval']['action_prediction']['top_ks'])
    
    return val_results

def compare_train_val_results(train_results, val_results, horizons, top_ks):
    """
    Compare and visualize training vs validation results.
    
    Args:
        train_results: Training evaluation results
        val_results: Validation evaluation results
        horizons: List of horizon values
        top_ks: List of top-k values
    """
    # Create figure with multiple subplots (one for each top-k)
    fig, axes = plt.subplots(1, len(top_ks), figsize=(16, 5), sharey=True)
    
    # Plot each top-k in a separate subplot
    for i, k in enumerate(sorted(top_ks)):
        ax = axes[i] if len(top_ks) > 1 else axes
        
        # Get data for this top-k
        train_y = [train_results.get(f"horizon_{h}_top_{k}", 0.0) for h in sorted(horizons)]
        val_y = [val_results.get(f"horizon_{h}_top_{k}", 0.0) for h in sorted(horizons)]
        
        # Plot
        ax.plot(sorted(horizons), train_y, 'b-o', label='Train')
        ax.plot(sorted(horizons), val_y, 'r-o', label='Validation')
        
        ax.set_title(f'Top-{k} Accuracy')
        ax.set_xlabel('Prediction Horizon')
        if i == 0:
            ax.set_ylabel('Accuracy')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'train_val_comparison_{timestamp}.png')
    plt.show()

# Example usage:
# 1. Load your trained model and data
# 2. Run the evaluation
"""
# Example config extension for evaluation
cfg['eval']['action_prediction'] = {
    'top_ks': [1, 3, 5, 10],
    'horizons': [1, 3, 5, 10, 15],
    'evaluate_on_train': True
}

# Run the evaluation
results = run_evaluation_suite(model, train_data, val_data, cfg)
"""