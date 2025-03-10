import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
from datetime import datetime


def plot_action_prediction_results(results, save_dir=None, experiment_name=None):
    """
    Create comprehensive visualizations of action prediction accuracy results.
    
    Args:
        results: Dictionary with keys like 'horizon_{h}_top_{k}' and accuracy values,
                or path to a checkpoint file containing evaluation results
        save_dir: Directory to save plots (default: 'plots')
        experiment_name: Name for the experiment (used in filenames)
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    # Setup
    if save_dir is None:
        save_dir = 'plots'
    
    if experiment_name is None:
        experiment_name = f"action_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results if a checkpoint path is provided
    if isinstance(results, str) and os.path.exists(results):
        checkpoint = torch.load(results, map_location='cpu')
        if 'evaluation_results' in checkpoint:
            results = checkpoint['evaluation_results']
        elif 'action_accuracies' in checkpoint:
            results = checkpoint['action_accuracies']
    
    # Extract horizons and top-k values from results
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
    
    # Create data matrix
    data_matrix = np.zeros((len(horizons), len(top_ks)))
    
    for i, h in enumerate(horizons):
        for j, k in enumerate(top_ks):
            key = f"horizon_{h}_top_{k}"
            value = results.get(key, 0.0)
            
            # Handle case where value is a list (average the values)
            if isinstance(value, list):
                if value:  # Check if list is non-empty
                    data_matrix[i, j] = np.mean(value)
                else:
                    data_matrix[i, j] = 0.0
            else:
                data_matrix[i, j] = value
    
    # Generate plots
    plot_files = {}
    
    # 1. Line plot: Accuracy vs Horizon for different top-k values
    plot_files['line_plot'] = create_line_plot(
        horizons, top_ks, data_matrix, 
        save_dir, experiment_name
    )
    
    # 2. Bar plot: Comparing top-k performance for each horizon
    plot_files['bar_plot'] = create_bar_plot(
        horizons, top_ks, data_matrix, 
        save_dir, experiment_name
    )
    
    # 3. Heatmap visualization
    plot_files['heatmap'] = create_heatmap(
        horizons, top_ks, data_matrix, 
        save_dir, experiment_name
    )
    
    # 4. Combined multi-plot figure for papers/presentations
    plot_files['combined_plot'] = create_combined_plot(
        horizons, top_ks, data_matrix, 
        save_dir, experiment_name
    )
    
    return plot_files


def create_line_plot(horizons, top_ks, data_matrix, save_dir, experiment_name):
    """Create a line plot showing accuracy vs horizon for each top-k value."""
    plt.figure(figsize=(10, 6))
    
    # Color map for different top-k values
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_ks)))
    
    for j, k in enumerate(top_ks):
        plt.plot(horizons, data_matrix[:, j], 'o-', 
                 color=colors[j], linewidth=2, 
                 label=f'Top-{k}')
    
    plt.title('Action Prediction Accuracy vs Prediction Horizon', fontsize=16)
    plt.xlabel('Prediction Horizon (frames)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add horizontal line at y=0.5 for reference
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    # Format x-axis to show all horizons
    plt.xticks(horizons)
    
    # Format y-axis to show percentages
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    # Add data point values
    for j, k in enumerate(top_ks):
        for i, h in enumerate(horizons):
            plt.annotate(f'{data_matrix[i, j]:.3f}', 
                         (h, data_matrix[i, j]),
                         textcoords="offset points", 
                         xytext=(0, 5), 
                         ha='center',
                         fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{experiment_name}_line_plot.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def create_bar_plot(horizons, top_ks, data_matrix, save_dir, experiment_name):
    """Create bar plots comparing top-k performance for each horizon."""
    # Create a separate bar plot for each horizon
    all_filepaths = []
    
    # Also create a combined figure with subplots
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 6), sharey=True)
    
    if len(horizons) == 1:
        axes = [axes]  # Make sure axes is iterable
    
    for i, h in enumerate(horizons):
        # Plot on the combined figure
        ax = axes[i]
        bars = ax.bar(range(len(top_ks)), data_matrix[i, :], width=0.6)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_title(f'Horizon = {h}', fontsize=14)
        ax.set_xticks(range(len(top_ks)))
        ax.set_xticklabels([f'Top-{k}' for k in top_ks])
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        if i == 0:
            ax.set_ylabel('Accuracy', fontsize=14)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Action Prediction Accuracy by Top-k', fontsize=16)
    plt.tight_layout()
    
    # Save combined figure
    filename = f"{experiment_name}_bar_plots.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    all_filepaths.append(filepath)
    
    return all_filepaths


def create_heatmap(horizons, top_ks, data_matrix, save_dir, experiment_name):
    """Create a heatmap visualization of the horizon/top-k accuracy matrix."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=[f'Top-{k}' for k in top_ks],
                yticklabels=[f'H={h}' for h in horizons],
                cbar_kws={'label': 'Accuracy'})
    
    plt.title('Action Prediction Accuracy Heatmap', fontsize=16)
    plt.xlabel('Top-k Value', fontsize=14)
    plt.ylabel('Prediction Horizon', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{experiment_name}_heatmap.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def create_combined_plot(horizons, top_ks, data_matrix, save_dir, experiment_name):
    """Create a comprehensive figure combining multiple visualization types."""
    fig = plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid of subplots
    gs = fig.add_gridspec(2, 2)
    
    # 1. Line plot in top-left
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_ks)))
    for j, k in enumerate(top_ks):
        ax1.plot(horizons, data_matrix[:, j], 'o-', 
                color=colors[j], linewidth=2, 
                label=f'Top-{k}')
    
    ax1.set_title('Accuracy vs Horizon', fontsize=14)
    ax1.set_xlabel('Prediction Horizon', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.set_xticks(horizons)
    ax1.set_ylim(0, 1.0)
    
    # 2. Bar plot for selected horizons in top-right
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Select representative horizons (first, middle, last)
    if len(horizons) >= 3:
        selected_horizons = [horizons[0], horizons[len(horizons)//2], horizons[-1]]
        selected_indices = [0, len(horizons)//2, len(horizons)-1]
    else:
        selected_horizons = horizons
        selected_indices = list(range(len(horizons)))
    
    bar_width = 0.8 / len(selected_horizons)
    for i, (h_idx, h) in enumerate(zip(selected_indices, selected_horizons)):
        positions = np.arange(len(top_ks)) + i * bar_width - (len(selected_horizons)-1) * bar_width / 2
        bars = ax2.bar(positions, data_matrix[h_idx, :], width=bar_width, 
                      label=f'H={h}')
    
    ax2.set_title('Top-k Comparison for Selected Horizons', fontsize=14)
    ax2.set_xticks(range(len(top_ks)))
    ax2.set_xticklabels([f'Top-{k}' for k in top_ks])
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', axis='y', alpha=0.7)
    ax2.set_ylim(0, 1.0)
    
    # 3. Heatmap in bottom-left
    ax3 = fig.add_subplot(gs[1, 0])
    heatmap = ax3.imshow(data_matrix, cmap='viridis', aspect='auto')
    
    # Add text annotations
    for i in range(len(horizons)):
        for j in range(len(top_ks)):
            ax3.text(j, i, f'{data_matrix[i, j]:.3f}', 
                    ha='center', va='center', 
                    color='w' if data_matrix[i, j] < 0.7 else 'black',
                    fontsize=9)
    
    ax3.set_title('Accuracy Heatmap', fontsize=14)
    ax3.set_xticks(range(len(top_ks)))
    ax3.set_xticklabels([f'Top-{k}' for k in top_ks])
    ax3.set_yticks(range(len(horizons)))
    ax3.set_yticklabels([f'H={h}' for h in horizons])
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax3)
    cbar.set_label('Accuracy', fontsize=12)
    
    # 4. Horizon impact analysis in bottom-right
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate relative performance degradation with increasing horizon
    degradation = np.zeros((len(horizons)-1, len(top_ks)))
    for i in range(len(horizons)-1):
        for j in range(len(top_ks)):
            degradation[i, j] = (data_matrix[i+1, j] - data_matrix[i, j]) / max(0.001, data_matrix[i, j]) * 100
    
    # Plot relative degradation
    for j, k in enumerate(top_ks):
        ax4.plot(horizons[1:], degradation[:, j], 'o--', 
                label=f'Top-{k}')
    
    ax4.set_title('Performance Change Between Horizons', fontsize=14)
    ax4.set_xlabel('Target Horizon', fontsize=12)
    ax4.set_ylabel('Percent Change (%)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(fontsize=10)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Overall title and layout
    plt.suptitle(f'Action Prediction Performance Analysis', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    filename = f"{experiment_name}_combined_analysis.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def parse_console_output(output_text):
    """
    Parse action prediction results from console output.
    
    Example:
    ```
    results = parse_console_output('''
    Action Prediction Accuracy:
    --------------------------------------------------
    Horizon    Top-k      Accuracy  
    --------------------------------------------------
    1          1          0.6007
    1          3          0.8163
    ... etc ...
    --------------------------------------------------
    ''')
    plot_from_results(results)
    ```
    """
    results = {}
    lines = output_text.strip().split('\n')
    
    # Find the start of the results table
    start_idx = None
    for i, line in enumerate(lines):
        if "Action Prediction Accuracy:" in line:
            start_idx = i
            break
    
    if start_idx is None:
        raise ValueError("Could not find results table in console output")
    
    # Skip header lines
    data_start = start_idx + 3
    
    # Parse results until the end or another separator
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if '---' in line:
            break
            
        parts = line.split()
        if len(parts) >= 3:
            horizon = int(parts[0])
            topk = int(parts[1])
            accuracy = float(parts[2])
            
            results[f"horizon_{horizon}_top_{topk}"] = accuracy
    
    return results


def plot_results_from_console(console_output, save_dir=None, experiment_name=None):
    """
    Parse console output and create visualizations.
    
    Args:
        console_output: String containing the console output with accuracy table
        save_dir: Directory to save plots
        experiment_name: Name for the experiment
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    results = parse_console_output(console_output)
    return plot_action_prediction_results(results, save_dir, experiment_name)


# Example of direct use with your console output
def main():
    # Copy your console output
    console_output = """
    Action Prediction Accuracy:
    --------------------------------------------------
    Horizon    Top-k      Accuracy  
    --------------------------------------------------
    1          1          0.6007
    1          3          0.8163
    1          5          0.8377
    1          10         0.8691
    3          1          0.5027
    3          3          0.7861
    3          5          0.8277
    3          10         0.8609
    5          1          0.4680
    5          3          0.7571
    5          5          0.8180
    5          10         0.8575
    10         1          0.4295
    10         3          0.7012
    10         5          0.8031
    10         10         0.8588
    15         1          0.4085
    15         3          0.6564
    15         5          0.7903
    15         10         0.8515
    --------------------------------------------------
    """
    
    # Set up directory
    os.makedirs("plots", exist_ok=True)
    
    # Parse and plot
    plot_files = plot_results_from_console(
        console_output,
        save_dir="plots",
        experiment_name="action_prediction_results"
    )
    
    print(f"Created visualization files:")
    for plot_type, filepath in plot_files.items():
        print(f"- {plot_type}: {filepath}")


if __name__ == "__main__":
    main()