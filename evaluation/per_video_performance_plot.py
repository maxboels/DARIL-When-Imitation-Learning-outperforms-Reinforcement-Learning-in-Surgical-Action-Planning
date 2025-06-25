#!/usr/bin/env python3
"""
Per-Video Performance Visualization for Surgical Action Recognition
Adapted from SwinT ensemble paper style for Method 1 vs Method 2 vs Method 3 comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


def create_per_video_performance_plot(
    experiment_results: Dict[str, Any],
    output_dir: str = "results/plots",
    save_data: bool = True,
    figure_size: tuple = (14, 10),
    dpi: int = 300
):
    """
    Create per-video performance plots similar to SwinT ensemble paper.
    
    Args:
        experiment_results: Dictionary containing results from all methods
        output_dir: Directory to save plots
        save_data: Whether to save underlying data
        figure_size: Figure size tuple
        dpi: DPI for saved figure
    """
    
    # Extract per-video results from experimental data
    video_results = extract_per_video_results(experiment_results)
    
    if not video_results:
        print("❌ No per-video results found in experiment data")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size, height_ratios=[2, 1])
    
    # Plot (a): Per-video mAP scores
    create_per_video_scatter_plot(video_results, ax1)
    
    # Plot (b): Box plots of mAP distributions
    create_method_boxplots(video_results, ax2)
    
    # Styling and layout
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / "per_video_performance_analysis.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    # Save additional formats
    plt.savefig(Path(output_dir) / "per_video_performance_analysis.pdf", bbox_inches='tight')
    plt.savefig(Path(output_dir) / "per_video_performance_analysis.svg", bbox_inches='tight')
    
    plt.show()
    
    # Save underlying data
    if save_data:
        save_performance_data(video_results, output_dir)
    
    print(f"✅ Per-video performance plots saved to: {plot_path}")
    
    # Print summary statistics
    print_performance_summary(video_results)


def extract_per_video_results(experiment_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract per-video results from experimental data.
    
    Args:
        experiment_results: Complete experiment results dictionary
        
    Returns:
        Dictionary mapping method_name -> {video_id: mAP_score}
    """
    
    video_results = {}
    
    # Method 1: Autoregressive IL
    method1 = experiment_results.get('method_1_autoregressive_il', {})
    if method1.get('status') == 'success':
        evaluation = method1.get('evaluation', {})
        detailed_metrics = evaluation.get('detailed_video_metrics', {})
        
        if detailed_metrics:
            video_results['Autoregressive IL'] = {
                video_id: metrics.get('mAP', 0.0) * 100  # Convert to percentage
                for video_id, metrics in detailed_metrics.items()
            }
    
    # Method 2: Conditional World Model + RL
    method2 = experiment_results.get('method_2_conditional_world_model', {})
    if method2.get('status') == 'success':
        # For RL methods, we might need to use different metrics
        # This is a placeholder - adapt based on your actual RL results structure
        world_model_eval = method2.get('world_model_evaluation', {})
        if 'detailed_video_metrics' in world_model_eval:
            video_results['World Model + RL'] = {
                video_id: metrics.get('mAP', 0.0) * 100
                for video_id, metrics in world_model_eval['detailed_video_metrics'].items()
            }
    
    # Method 3: Direct Video RL
    method3 = experiment_results.get('method_3_direct_video_rl', {})
    if method3.get('status') == 'success':
        # Placeholder for direct RL results
        # Adapt based on your actual results structure
        pass
    
    # If you have different model configurations, add them here
    # Example: Different architectures, ensemble methods, etc.
    
    return video_results


def create_per_video_scatter_plot(video_results: Dict[str, Dict[str, float]], ax):
    """Create the per-video scatter plot (top panel)."""
    
    # Define colors for each method (similar to original paper)
    method_colors = {
        'Autoregressive IL': '#d62728',              # Red
        'Autoregressive IL + MultiT': '#ff7f0e',     # Orange  
        'Autoregressive IL + SelfD': '#2ca02c',      # Green
        'World Model + RL': '#1f77b4',               # Blue
        'Direct Video RL': '#9467bd',                # Purple
        'Ensemble': '#8c564b'                        # Brown
    }
    
    # Get all unique videos and sort by best method performance
    all_videos = set()
    for method_results in video_results.values():
        all_videos.update(method_results.keys())
    
    # Sort videos by performance of best method (for nice visualization)
    if video_results:
        best_method = max(video_results.keys(), 
                         key=lambda m: np.mean(list(video_results[m].values())))
        video_order = sorted(all_videos, 
                           key=lambda v: video_results[best_method].get(v, 0), 
                           reverse=True)
    else:
        video_order = sorted(all_videos)
    
    # Create x-axis positions
    x_positions = np.arange(len(video_order))
    
    # Plot each method
    for method_name, method_data in video_results.items():
        y_values = [method_data.get(video, 0) for video in video_order]
        color = method_colors.get(method_name, '#333333')
        
        ax.scatter(x_positions, y_values, 
                  color=color, alpha=0.7, s=30, 
                  label=method_name, edgecolors='white', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Videos: Ranked by Performance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Average Precision (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Average Precision scores per video', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_positions[::max(1, len(video_order)//20)])  # Show every nth label
    ax.set_xticklabels([video_order[i] for i in range(0, len(video_order), max(1, len(video_order)//20))], 
                      rotation=45, ha='right', fontsize=8)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set y-axis limits with some padding
    if video_results:
        all_values = []
        for method_data in video_results.values():
            all_values.extend(method_data.values())
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Add subplot label
    ax.text(-0.1, 1.02, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')


def create_method_boxplots(video_results: Dict[str, Dict[str, float]], ax):
    """Create box plots showing distribution of mAP scores (bottom panel)."""
    
    # Prepare data for box plots
    methods = []
    scores = []
    
    for method_name, method_data in video_results.items():
        for score in method_data.values():
            methods.append(method_name)
            scores.append(score)
    
    if not methods:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({'Method': methods, 'mAP': scores})
    
    # Define method order and colors
    method_order = list(video_results.keys())
    method_colors = {
        'Autoregressive IL': '#d62728',
        'Autoregressive IL + MultiT': '#ff7f0e',
        'Autoregressive IL + SelfD': '#2ca02c', 
        'World Model + RL': '#1f77b4',
        'Direct Video RL': '#9467bd',
        'Ensemble': '#8c564b'
    }
    
    # Create box plots
    box_data = []
    colors = []
    labels = []
    
    for method in method_order:
        method_scores = df[df['Method'] == method]['mAP'].values
        if len(method_scores) > 0:
            box_data.append(method_scores)
            colors.append(method_colors.get(method, '#333333'))
            labels.append(method)
    
    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, 
                       notch=True, showmeans=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style the plot elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', alpha=0.8)
        
        # Mean markers
        plt.setp(bp['means'], marker='D', markerfacecolor='white', 
                markeredgecolor='black', markersize=4)
        
        # Scatter individual points
        for i, (method, color) in enumerate(zip(labels, colors)):
            method_scores = df[df['Method'] == method]['mAP'].values
            y_positions = method_scores
            x_positions = np.random.normal(i+1, 0.04, size=len(y_positions))
            ax.scatter(x_positions, y_positions, alpha=0.6, s=20, color=color)
    
    # Styling
    ax.set_xlabel('Method Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Average Precision (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of mAP scores across videos', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=15)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add subplot label
    ax.text(-0.1, 1.02, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')


def save_performance_data(video_results: Dict[str, Dict[str, float]], output_dir: str):
    """Save the underlying performance data."""
    
    # Create comprehensive data structure
    data_export = {
        'per_video_results': video_results,
        'summary_statistics': {},
        'method_comparisons': {}
    }
    
    # Calculate summary statistics for each method
    for method_name, method_data in video_results.items():
        scores = list(method_data.values())
        if scores:
            data_export['summary_statistics'][method_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
                'num_videos': len(scores)
            }
    
    # Create CSV for easy analysis
    rows = []
    for method_name, method_data in video_results.items():
        for video_id, score in method_data.items():
            rows.append({
                'Method': method_name,
                'Video_ID': video_id,
                'mAP_Percentage': score
            })
    
    df = pd.DataFrame(rows)
    
    # Save files
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(Path(output_dir) / 'per_video_performance_data.json', 'w') as f:
        json.dump(data_export, f, indent=2)
    
    # Save CSV
    df.to_csv(Path(output_dir) / 'per_video_performance_data.csv', index=False)
    
    print(f"✅ Performance data saved to: {output_dir}")


def print_performance_summary(video_results: Dict[str, Dict[str, float]]):
    """Print a summary of performance across methods."""
    
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for method_name, method_data in video_results.items():
        if method_data:
            scores = list(method_data.values())
            print(f"\n🎯 {method_name}:")
            print(f"   Mean mAP: {np.mean(scores):.2f}% ± {np.std(scores):.2f}%")
            print(f"   Median:   {np.median(scores):.2f}%")
            print(f"   Range:    {np.min(scores):.2f}% - {np.max(scores):.2f}%")
            print(f"   Videos:   {len(scores)}")
    
    # Best performing method
    if video_results:
        best_method = max(video_results.keys(), 
                         key=lambda m: np.mean(list(video_results[m].values())))
        best_score = np.mean(list(video_results[best_method].values()))
        print(f"\n🏆 Best Method: {best_method} ({best_score:.2f}% mAP)")


# Example usage function
def create_plots_from_experiment_results(results_file: str = "results/complete_results.json"):
    """
    Create plots from saved experiment results file.
    
    Args:
        results_file: Path to the complete results JSON file
    """
    
    try:
        with open(results_file, 'r') as f:
            experiment_results = json.load(f)
        
        create_per_video_performance_plot(
            experiment_results=experiment_results,
            output_dir="results/publication_plots"
        )
        
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("Make sure to run the complete experiment first!")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")


# Enhanced version with ensemble simulation
def simulate_ensemble_results(base_results: Dict[str, Dict[str, float]], 
                            improvement_factor: float = 1.1) -> Dict[str, Dict[str, float]]:
    """
    Simulate ensemble results by improving base method performance.
    
    Args:
        base_results: Base method results
        improvement_factor: Factor by which ensemble improves performance
        
    Returns:
        Enhanced results including ensemble
    """
    
    enhanced_results = base_results.copy()
    
    # Create ensemble results (simulate improvement over best base method)
    if base_results:
        best_method_name = max(base_results.keys(), 
                              key=lambda m: np.mean(list(base_results[m].values())))
        best_method_data = base_results[best_method_name]
        
        # Simulate ensemble improvement
        ensemble_results = {}
        for video_id, score in best_method_data.items():
            # Add some improvement with slight randomness
            improvement = improvement_factor + np.random.normal(0, 0.02)
            ensemble_score = min(score * improvement, 75.0)  # Cap at reasonable max
            ensemble_results[video_id] = ensemble_score
        
        enhanced_results['Ensemble'] = ensemble_results
    
    return enhanced_results


if __name__ == "__main__":
    print("📊 PER-VIDEO PERFORMANCE VISUALIZATION")
    print("=" * 50)
    print("🎯 Adapted from SwinT ensemble paper style")
    print("📈 Shows per-video mAP scores and distributions")
    print("🔄 Supports multiple method configurations")
    
    # Example of how to use with your experiment results
    print("\n📝 Usage:")
    print("# From experiment results file:")
    print("create_plots_from_experiment_results('results/complete_results.json')")
    print()
    print("# From experiment results dictionary:")
    print("create_per_video_performance_plot(experiment_results)")
    
    # Example with simulated data
    print("\n🧪 Creating example plot with simulated data...")
    
    # Simulate some example results
    video_ids = ['VID02', 'VID06', 'VID111', 'VID14', 'VID23', 'VID25', 'VID50', 'VID51', 'VID66', 'VID79']
    
    example_results = {
        'Autoregressive IL': {
            vid: np.random.normal(32, 8) for vid in video_ids  # Around 32% like your results
        },
        'World Model + RL': {
            vid: np.random.normal(28, 6) for vid in video_ids  # Slightly lower for RL
        }
    }
    
    # Add ensemble simulation
    example_results = simulate_ensemble_results(example_results, improvement_factor=1.15)
    
    create_per_video_performance_plot(
        experiment_results={'method_1_autoregressive_il': {
            'status': 'success',
            'evaluation': {
                'detailed_video_metrics': {
                    vid: {'mAP': score/100} for vid, score in example_results['Autoregressive IL'].items()
                }
            }
        }},
        output_dir="example_plots"
    )
