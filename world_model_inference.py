import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import cv2
from tqdm import tqdm
import json
import torch.nn.functional as F
from PIL import Image
import imageio
from datetime import datetime


def create_qualitative_demo(model, test_loader, cfg, device='cuda', output_dir=None, num_samples=3):
    """
    Create a qualitative demo of the model's action prediction capabilities.
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        cfg: Configuration dictionary
        device: Computation device
        output_dir: Directory to save demo outputs
        num_samples: Number of sample sequences to visualize
        
    Returns:
        Path to demo outputs
    """
    # Set default output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("demos", f"qualitative_demo_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load class labels
    with open(cfg['data']['paths']['class_labels_file_path'], 'r') as f:
        class_labels = json.load(f)
    action_labels = [class_name for class_id, class_name in class_labels['action'].items()]
    
    # Evaluation parameters
    max_horizon = cfg['eval']['world_model']['max_horizon']
    use_memory = cfg['eval']['world_model']['use_memory']
    
    # Put model in evaluation mode
    model.eval()
    
    # Process a few batches for visualization
    samples_processed = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, (z_seq, _z_seq, _a_seq, f_a_seq) in enumerate(test_loader):
            # Process only a few samples for the demo
            if samples_processed >= num_samples:
                break
                
            z_seq, _z_seq, _a_seq, f_a_seq = z_seq.to(device), _z_seq.to(device), _a_seq.to(device), f_a_seq.to(device)
            
            # Generate predictions for this sequence
            outputs = model.generate(z_seq, horizon=max_horizon, use_memory=use_memory)
            
            # Get action predictions
            if 'f_a_seq_hat' in outputs:
                # Get probabilities
                action_probs = torch.sigmoid(outputs['f_a_seq_hat'])
                
                # Store results for each sequence in the batch
                for i in range(min(z_seq.size(0), num_samples - samples_processed)):
                    sample_result = {
                        'input_embedding': z_seq[i].cpu().numpy(),
                        'target_actions': f_a_seq[i].cpu().numpy(),
                        'predicted_actions': action_probs[i].cpu().numpy(),
                        'sequence_idx': f"{batch_idx}_{i}"
                    }
                    results.append(sample_result)
                    samples_processed += 1
                    
                    # Generate visualizations for this sample
                    visualize_action_predictions(
                        sample_result,
                        action_labels,
                        os.path.join(output_dir, f"sequence_{batch_idx}_{i}"),
                        cfg
                    )
            
            if samples_processed >= num_samples:
                break
    
    # Generate a summary video or composite visualization
    create_summary_visualization(results, action_labels, output_dir, cfg)
    
    print(f"Qualitative demo generated at: {output_dir}")
    return output_dir


def visualize_action_predictions(sample, action_labels, output_path, cfg):
    """
    Create visualizations for a single sample sequence.
    
    Args:
        sample: Dictionary with sample data
        action_labels: List of action class names
        output_path: Path to save visualizations
        cfg: Configuration dictionary
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Extract data
    target_actions = sample['target_actions']  # [seq_len, num_classes]
    predicted_actions = sample['predicted_actions']  # [seq_len, num_classes]
    
    seq_length, num_classes = predicted_actions.shape
    
    # 1. Top actions visualization
    visualize_top_actions(
        predicted_actions, 
        target_actions, 
        action_labels, 
        os.path.join(output_path, "top_actions.png"),
        top_k=5
    )
    
    # 2. Temporal prediction heatmap
    visualize_prediction_heatmap(
        predicted_actions,
        target_actions,
        action_labels,
        os.path.join(output_path, "prediction_heatmap.png")
    )
    
    # 3. Action trajectory visualization
    visualize_action_trajectory(
        predicted_actions,
        target_actions,
        action_labels,
        os.path.join(output_path, "action_trajectory.png")
    )
    
    # 4. Confidence over time
    visualize_confidence_over_time(
        predicted_actions,
        target_actions,
        action_labels,
        os.path.join(output_path, "confidence_over_time.png")
    )
    
    # 5. Generate GIF animation of predictions evolving over time
    create_prediction_animation(
        predicted_actions,
        target_actions,
        action_labels,
        os.path.join(output_path, "prediction_animation.gif")
    )


def visualize_top_actions(predictions, targets, action_labels, output_path, top_k=5):
    """
    Visualize the top-k predicted actions for the first few timesteps.
    
    Args:
        predictions: Predicted action probabilities [seq_len, num_classes]
        targets: Target action labels [seq_len, num_classes]
        action_labels: List of action class names
        output_path: Path to save visualization
        top_k: Number of top actions to display
    """
    # Select first few timesteps to visualize
    num_timesteps = min(5, predictions.shape[0])
    
    fig, axes = plt.subplots(num_timesteps, 1, figsize=(12, 3*num_timesteps))
    if num_timesteps == 1:
        axes = [axes]
    
    for t in range(num_timesteps):
        # Get top-k predictions for this timestep
        probs = predictions[t]
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        top_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in top_indices]
        
        # Get ground truth actions
        true_indices = np.where(targets[t] > 0.5)[0]
        true_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in true_indices]
        
        # Create horizontal bar chart
        bars = axes[t].barh(range(len(top_indices)), top_probs, align='center')
        
        # Color bars based on whether they match ground truth
        for i, idx in enumerate(top_indices):
            if idx in true_indices:
                bars[i].set_color('green')
            else:
                bars[i].set_color('skyblue')
        
        # Add labels
        axes[t].set_yticks(range(len(top_indices)))
        axes[t].set_yticklabels(top_labels)
        axes[t].set_xlabel('Probability')
        axes[t].set_title(f'Timestep {t+1} - Top {top_k} Predicted Actions')
        axes[t].set_xlim(0, 1.0)
        
        # Add ground truth annotation
        if len(true_labels) > 0:
            truth_text = "Ground Truth: " + ", ".join(true_labels)
        else:
            truth_text = "Ground Truth: None"
        axes[t].annotate(truth_text, xy=(0.5, -0.3), xycoords='axes fraction', 
                         ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_prediction_heatmap(predictions, targets, action_labels, output_path, max_actions=15):
    """
    Create a heatmap visualization of action predictions over time.
    
    Args:
        predictions: Predicted action probabilities [seq_len, num_classes]
        targets: Target action labels [seq_len, num_classes]
        action_labels: List of action class names
        output_path: Path to save visualization
        max_actions: Maximum number of actions to display
    """
    seq_length, num_classes = predictions.shape
    
    # Find most relevant actions (highest probability across sequence)
    action_importance = np.max(predictions, axis=0)
    top_action_indices = np.argsort(action_importance)[-max_actions:][::-1]
    
    # Extract relevant predictions and targets
    relevant_predictions = predictions[:, top_action_indices]
    relevant_targets = targets[:, top_action_indices]
    relevant_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in top_action_indices]
    
    # Create figure with side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot predictions heatmap
    im1 = ax1.imshow(relevant_predictions.T, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Predicted Actions Over Time', fontsize=14)
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Action', fontsize=12)
    ax1.set_yticks(range(len(relevant_labels)))
    ax1.set_yticklabels(relevant_labels)
    plt.colorbar(im1, ax=ax1, label='Prediction Probability')
    
    # Plot ground truth heatmap
    im2 = ax2.imshow(relevant_targets.T, cmap='binary', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Ground Truth Actions Over Time', fontsize=14)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_yticks(range(len(relevant_labels)))
    ax2.set_yticklabels([])  # Hide labels on second plot
    plt.colorbar(im2, ax=ax2, label='Present (1) / Absent (0)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_action_trajectory(predictions, targets, action_labels, output_path, top_k=3):
    """
    Visualize how the most important actions evolve over time.
    
    Args:
        predictions: Predicted action probabilities [seq_len, num_classes]
        targets: Target action labels [seq_len, num_classes]
        action_labels: List of action class names
        output_path: Path to save visualization
        top_k: Number of top actions to track
    """
    seq_length, num_classes = predictions.shape
    
    # Find top actions across all timesteps (most frequently predicted)
    average_probs = np.mean(predictions, axis=0)
    top_indices = np.argsort(average_probs)[-top_k:][::-1]
    
    # Extract trajectories for top actions
    trajectories = predictions[:, top_indices]
    target_trajectories = targets[:, top_indices]
    
    # Get labels for top actions
    top_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in top_indices]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot prediction trajectories
    for i in range(top_k):
        plt.plot(trajectories[:, i], '-o', linewidth=2, label=f"Pred: {top_labels[i]}")
    
    # Plot ground truth as shaded areas
    for i in range(top_k):
        for t in range(seq_length):
            if target_trajectories[t, i] > 0.5:
                plt.axvspan(t-0.4, t+0.4, alpha=0.2, color=f'C{i}')
    
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title('Top Action Prediction Trajectories', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    # Mark ground truth with symbols in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    for i in range(top_k):
        patch = patches.Patch(color=f'C{i}', alpha=0.2, label=f"True: {top_labels[i]}")
        handles.append(patch)
        labels.append(f"True: {top_labels[i]}")
    
    plt.legend(handles, labels, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_confidence_over_time(predictions, targets, action_labels, output_path):
    """
    Visualize how prediction confidence changes with prediction horizon.
    
    Args:
        predictions: Predicted action probabilities [seq_len, num_classes]
        targets: Target action labels [seq_len, num_classes]
        action_labels: List of action class names
        output_path: Path to save visualization
    """
    seq_length, num_classes = predictions.shape
    
    # Calculate confidence metrics for each timestep
    top1_confidence = np.max(predictions, axis=1)
    entropy = -np.sum(predictions * np.log2(predictions + 1e-10), axis=1)
    max_entropy = np.log2(num_classes)
    normalized_entropy = entropy / max_entropy
    
    # Calculate accuracy at each timestep
    accuracy = []
    for t in range(seq_length):
        pred_top = np.argmax(predictions[t])
        true_classes = np.where(targets[t] > 0.5)[0]
        
        # If no true classes, count as incorrect
        if len(true_classes) == 0:
            accuracy.append(0)
        # If top prediction is among true classes, count as correct
        elif pred_top in true_classes:
            accuracy.append(1)
        else:
            accuracy.append(0)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot confidence
    ax1.plot(top1_confidence, 'o-', linewidth=2, label='Top-1 Confidence')
    ax1.plot(1 - normalized_entropy, 's--', linewidth=2, label='Certainty (1 - Norm. Entropy)')
    
    ax1.set_ylabel('Confidence', fontsize=12)
    ax1.set_title('Prediction Confidence vs Horizon', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    ax1.set_ylim(0, 1.05)
    
    # Plot accuracy
    ax2.plot(accuracy, 'o-', linewidth=2, color='green', label='Top-1 Accuracy')
    
    # Add moving average for better visualization
    window_size = min(3, seq_length)
    if seq_length > window_size:
        moving_avg = np.convolve(accuracy, np.ones(window_size)/window_size, mode='valid')
        padding = np.ones(window_size-1) * moving_avg[0]
        moving_avg = np.concatenate((padding, moving_avg))
        ax2.plot(moving_avg, '--', linewidth=2, color='red', label=f'{window_size}-step Moving Avg')
    
    ax2.set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Prediction Accuracy vs Horizon', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_prediction_animation(predictions, targets, action_labels, output_path, top_k=5):
    """
    Create an animated GIF showing how predictions evolve over the sequence.
    
    Args:
        predictions: Predicted action probabilities [seq_len, num_classes]
        targets: Target action labels [seq_len, num_classes]
        action_labels: List of action class names
        output_path: Path to save visualization
        top_k: Number of top actions to display
    """
    seq_length, num_classes = predictions.shape
    
    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Function to update the frame
    def update_frame(t):
        ax.clear()
        
        # Get top predictions for this timestep
        probs = predictions[t]
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        top_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in top_indices]
        
        # Get ground truth actions
        true_indices = np.where(targets[t] > 0.5)[0]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top_indices)), top_probs, align='center')
        
        # Color bars based on whether they match ground truth
        for i, idx in enumerate(top_indices):
            if idx in true_indices:
                bars[i].set_color('green')
            else:
                bars[i].set_color('skyblue')
        
        # Add labels
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(top_labels)
        ax.set_xlabel('Probability')
        ax.set_title(f'Timestep {t+1}/{seq_length} - Top {top_k} Predicted Actions')
        ax.set_xlim(0, 1.0)
        
        # Add ground truth annotation
        true_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in true_indices]
        if len(true_labels) > 0:
            truth_text = "Ground Truth: " + ", ".join(true_labels)
        else:
            truth_text = "Ground Truth: None"
        ax.annotate(truth_text, xy=(0.5, -0.1), xycoords='axes fraction', 
                     ha='center', va='center', fontsize=10, color='red')
        
        return bars
    
    # Create animation
    frames = min(seq_length, 15)  # Limit to 15 frames to keep GIF size reasonable
    anim = FuncAnimation(fig, update_frame, frames=range(frames), interval=500)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=2, dpi=100)
    plt.close()


def create_summary_visualization(results, action_labels, output_dir, cfg):
    """
    Create a summary visualization of all samples.
    
    Args:
        results: List of sample results
        action_labels: List of action class names
        output_dir: Directory to save outputs
        cfg: Configuration dictionary
    """
    # Create a summary plot showing aggregate metrics
    plt.figure(figsize=(15, 10))
    
    # Create a GridSpec to organize subplots
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # 1. Top predicted actions across all samples
    ax1 = plt.subplot(gs[0, 0])
    visualize_top_actions_summary(results, action_labels, ax1)
    
    # 2. Prediction accuracy vs horizon
    ax2 = plt.subplot(gs[0, 1])
    visualize_accuracy_vs_horizon(results, ax2)
    
    # 3. Action co-occurrence
    ax3 = plt.subplot(gs[1, 0])
    visualize_action_co_occurrence(results, action_labels, ax3)
    
    # 4. Confidence calibration
    ax4 = plt.subplot(gs[1, 1])
    visualize_confidence_calibration(results, ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary.png"), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_top_actions_summary(results, action_labels, ax):
    """Visualize top predicted actions across all samples."""
    # Collect all predictions
    all_predictions = []
    for sample in results:
        all_predictions.append(sample['predicted_actions'])
    
    if all_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Get average probability for each action
        avg_probs = np.mean(all_predictions, axis=0)
        
        # Get top 10 actions
        top_indices = np.argsort(avg_probs)[-10:][::-1]
        top_probs = avg_probs[top_indices]
        top_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in top_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top_indices)), top_probs, align='center')
        
        # Add labels
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(top_labels)
        ax.set_xlabel('Average Probability')
        ax.set_title('Top 10 Predicted Actions (All Samples)')
        ax.set_xlim(0, min(1.0, max(top_probs) * 1.2))
    else:
        ax.text(0.5, 0.5, "No prediction data available", 
                ha='center', va='center', transform=ax.transAxes)


def visualize_accuracy_vs_horizon(results, ax):
    """Visualize how prediction accuracy changes with horizon."""
    if not results:
        ax.text(0.5, 0.5, "No prediction data available", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate accuracy at each timestep
    max_horizon = min(15, max(len(sample['predicted_actions']) for sample in results))
    accuracy = np.zeros(max_horizon)
    count = np.zeros(max_horizon)
    
    for sample in results:
        preds = sample['predicted_actions']
        targets = sample['target_actions']
        
        for t in range(min(max_horizon, len(preds))):
            if t < len(targets):
                pred_top = np.argmax(preds[t])
                true_classes = np.where(targets[t] > 0.5)[0]
                
                if len(true_classes) > 0 and pred_top in true_classes:
                    accuracy[t] += 1
                count[t] += 1
    
    # Calculate accuracy percentage
    accuracy = np.divide(accuracy, count, out=np.zeros_like(accuracy), where=count!=0)
    
    # Plot accuracy vs horizon
    ax.plot(range(1, max_horizon+1), accuracy, 'o-', linewidth=2)
    ax.set_xlabel('Prediction Horizon (timesteps)')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Prediction Accuracy vs Horizon')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)


def visualize_action_co_occurrence(results, action_labels, ax):
    """Visualize which actions tend to co-occur."""
    if not results:
        ax.text(0.5, 0.5, "No prediction data available", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Collect all ground truth labels
    all_targets = []
    for sample in results:
        all_targets.append(sample['target_actions'])
    
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Find most common actions
    action_frequency = np.sum(all_targets, axis=0)
    top_indices = np.argsort(action_frequency)[-10:][::-1]
    
    # Calculate co-occurrence matrix
    co_occurrence = np.zeros((len(top_indices), len(top_indices)))
    for i, idx1 in enumerate(top_indices):
        for j, idx2 in enumerate(top_indices):
            if i == j:
                co_occurrence[i, j] = 1.0
            else:
                co_occurrence[i, j] = np.sum(all_targets[:, idx1] * all_targets[:, idx2]) / max(1, np.sum(all_targets[:, idx1]))
    
    # Create heatmap
    im = ax.imshow(co_occurrence, cmap='viridis', vmin=0, vmax=1)
    
    # Add labels
    top_labels = [action_labels[i] if i < len(action_labels) else f"Action {i}" for i in top_indices]
    ax.set_xticks(range(len(top_indices)))
    ax.set_yticks(range(len(top_indices)))
    ax.set_xticklabels(top_labels, rotation=45, ha='right')
    ax.set_yticklabels(top_labels)
    
    ax.set_title('Action Co-occurrence Matrix')
    plt.colorbar(im, ax=ax, label='Co-occurrence Probability')


def visualize_confidence_calibration(results, ax):
    """Visualize how well-calibrated the model's confidence is."""
    if not results:
        ax.text(0.5, 0.5, "No prediction data available", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Collect all predictions and targets
    all_preds = []
    all_targets = []
    
    for sample in results:
        preds = sample['predicted_actions']
        targets = sample['target_actions']
        
        for t in range(min(len(preds), len(targets))):
            top_class = np.argmax(preds[t])
            confidence = preds[t][top_class]
            
            is_correct = 0
            true_classes = np.where(targets[t] > 0.5)[0]
            if len(true_classes) > 0 and top_class in true_classes:
                is_correct = 1
            
            all_preds.append(confidence)
            all_targets.append(is_correct)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Create confidence bins
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate accuracy in each bin
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bin_edges) - 1):
        mask = (all_preds >= bin_edges[i]) & (all_preds < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(all_targets[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)
    
    # Plot calibration curve
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    ax.bar(bin_centers, bin_accuracies, width=0.1, alpha=0.5, label='Model Calibration')
    
    # Plot histogram of confidence values
    ax_twin = ax.twinx()
    ax_twin.hist(all_preds, bins=bin_edges, alpha=0.3, color='blue', label='Confidence Histogram')
    ax_twin.set_ylabel('Count')
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Confidence Calibration')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add both legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')


def run_inference_demo(model, dataset, cfg, device='cuda', output_dir=None):
    """
    Run a comprehensive inference demo.
    
    Args:
        model: Trained model
        dataset: Dataset object
        cfg: Configuration dictionary
        device: Device to run inference on
        output_dir: Directory to save outputs
        
    Returns:
        Path to demo outputs
    """
    from torch.utils.data import DataLoader
    
    # Create a small data loader with a few samples
    demo_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Run the qualitative demo
    return create_qualitative_demo(
        model, demo_loader, cfg, device, output_dir, num_samples=3
    )


# Example usage:
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate qualitative demo')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model = CausalGPT2ForFrameEmbeddings(
        hidden_dim=cfg['models']['world_model']['hidden_dim'],
        embedding_dim=cfg['models']['world_model']['embedding_dim'],
        n_layer=cfg['models']['world_model']['n_layer'],
        use_head=cfg['models']['world_model']['use_head'],
        targets_dims=cfg['models']['world_model']['targets_dims'],
        target_heads=cfg['models']['world_model']['target_heads'],
        loss_weights=cfg['models']['world_model']['loss_weights']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset and loader
    # (This part depends on your specific dataset implementation)
    
    # Run demo
    output_dir = create_qualitative_demo(model, test_loader, cfg, device, args.output, num_samples=3)
    print(f"Demo results saved to: {output_dir}")