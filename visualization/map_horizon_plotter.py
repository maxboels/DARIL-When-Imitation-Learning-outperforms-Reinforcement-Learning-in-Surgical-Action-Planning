import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def plot_map_vs_horizon(planning_results: Dict, 
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (14, 8), # Dims: (width, height)
                       style: str = 'paper',
                       show_confidence_intervals: bool = True,
                       include_overall_ivt: bool = True,
                       include_additional_metrics: bool = True):
    """
    Plot mAP scores for triplet components vs planning horizon for MICCAI paper.
    
    Args:
        planning_results: Dictionary containing detailed planning results
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
        style: Plotting style ('paper' for publication, 'presentation' for slides)
        show_confidence_intervals: Whether to show error bars/confidence intervals
        include_overall_ivt: Whether to include overall IVT mAP line
        include_additional_metrics: Whether to include additional metrics subplot
        
    Returns:
        matplotlib Figure object
    """
    
    # Set style based on target
    if style == 'paper':
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        font_size = 12
        title_size = 14
        legend_size = 11
    else:  # presentation
        plt.style.use('seaborn-v0_8-dark')
        font_size = 14
        title_size = 16
        legend_size = 13
    
    # Extract data from detailed results
    detailed_results = planning_results['detailed_video_results']
    video_ids = list(detailed_results.keys())
    num_videos = len(video_ids)
    
    # Extract horizon keys from the first video and sort them
    first_video_results = detailed_results[video_ids[0]]['horizon_results']
    horizons = sorted(first_video_results.keys(), key=lambda x: float(x.rstrip('s')))
    horizon_seconds = [float(h.rstrip('s')) for h in horizons]
    
    # Aggregate data across videos
    horizon_data = aggregate_video_results(detailed_results, horizons)
    
    # Define triplet components and their styling
    metrics_config = {
        'Overall IVT': {
            'key': 'ivt_mAP',
            'color': '#2E86AB',
            'linestyle': '-',
            'linewidth': 3,
            'marker': 'o',
            'markersize': 8,
            'alpha': 0.9
        },
        'Instrument (I)': {
            'key': 'ivt_i_mAP',
            'color': '#A23B72',
            'linestyle': '-',
            'linewidth': 2.5,
            'marker': 's',
            'markersize': 7,
            'alpha': 0.8
        },
        'Verb (V)': {
            'key': 'ivt_v_mAP',
            'color': '#F18F01',
            'linestyle': '-',
            'linewidth': 2.5,
            'marker': '^',
            'markersize': 7,
            'alpha': 0.8
        },
        'Target (T)': {
            'key': 'ivt_t_mAP',
            'color': '#C73E1D',
            'linestyle': '-',
            'linewidth': 2.5,
            'marker': 'D',
            'markersize': 6,
            'alpha': 0.8
        },
        'Instrument-Verb (IV)': {
            'key': 'ivt_iv_mAP',
            'color': '#7209B7',
            'linestyle': '--',
            'linewidth': 2,
            'marker': 'v',
            'markersize': 6,
            'alpha': 0.7
        },
        'Instrument-Target (IT)': {
            'key': 'ivt_it_mAP',
            'color': '#0B6E4F',
            'linestyle': '--',
            'linewidth': 2,
            'marker': '<',
            'markersize': 6,
            'alpha': 0.7
        }
    }
    
    # Create figure with optional additional metrics panel
    if include_additional_metrics:
        fig = plt.figure(figsize=figsize, dpi=300 if style == 'paper' else 100)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1], 
                             hspace=0.35, wspace=0.3)
        ax = fig.add_subplot(gs[0, :])  # Main plot spans top
        ax_additional = fig.add_subplot(gs[1, 0])  # Additional metrics
        ax_stats = fig.add_subplot(gs[1, 1])  # Statistics
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=300 if style == 'paper' else 100)
    
    # Extract data for each metric
    for metric_name, config in metrics_config.items():
        if not include_overall_ivt and metric_name == 'Overall IVT':
            continue
            
        means = []
        stds = []
        
        for horizon in horizons:
            if horizon in horizon_data:
                mean_val = horizon_data[horizon]['mean'][config['key']]
                std_val = horizon_data[horizon]['std'][config['key']] if num_videos > 1 else 0
                means.append(mean_val)
                stds.append(std_val)
            else:
                means.append(0)
                stds.append(0)
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot main line
        line = ax.plot(horizon_seconds, means, 
                      color=config['color'],
                      linestyle=config['linestyle'],
                      linewidth=config['linewidth'],
                      marker=config['marker'],
                      markersize=config['markersize'],
                      alpha=config['alpha'],
                      label=metric_name,
                      markerfacecolor=config['color'],
                      markeredgecolor='white',
                      markeredgewidth=1)
        
        # Add confidence intervals if requested and std > 0
        if show_confidence_intervals and np.any(stds > 0):
            ax.fill_between(horizon_seconds, 
                          means - stds, 
                          means + stds,
                          color=config['color'], 
                          alpha=0.2,
                          linewidth=0)
    
    # Customize plot
    ax.set_xlabel('Planning Horizon (seconds)', fontsize=font_size, fontweight='bold')
    ax.set_ylabel('mAP Score', fontsize=font_size, fontweight='bold')
    ax.set_title('Triplet Component mAP Deterioration over Planning Horizon', 
                fontsize=title_size, fontweight='bold', pad=20)
    
    # Set axis properties
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(horizon_seconds)
    ax.set_xticklabels([f'{h}s' for h in horizon_seconds])
    
    # Format y-axis
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 1.1, 0.1)])
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=legend_size, 
                      frameon=True, fancybox=True, shadow=True,
                      framealpha=0.9, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # Add performance annotations
    add_performance_annotations(ax, horizon_data, horizons, horizon_seconds, font_size)
    
    # Add statistical significance indicators (if multiple videos)
    if num_videos > 1:
        add_significance_indicators(ax, horizon_data, horizons, horizon_seconds)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Add subtle background shading for different time ranges
    add_background_shading(ax, horizon_seconds)
    
    # Add additional metrics subplot if requested
    if include_additional_metrics:
        plot_additional_metrics(ax_additional, horizon_data, horizons, horizon_seconds, font_size)
        create_enhanced_stats_table(ax_stats, horizon_data, horizons, num_videos, font_size)
        
        # Add video information
        fig.text(0.02, 0.02, f"Videos analyzed: {num_videos} | Video IDs: {', '.join(video_ids[:3])}{'...' if len(video_ids) > 3 else ''}", 
                fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    
    return

def aggregate_video_results(detailed_results: Dict, horizons: List[str]) -> Dict:
    """
    Aggregate results across multiple videos to compute means and standard deviations.
    
    Args:
        detailed_results: Dictionary with video_id -> horizon_results structure
        horizons: List of horizon keys to process
        
    Returns:
        Dictionary with aggregated statistics per horizon
    """
    
    aggregated = {}
    
    for horizon in horizons:
        # Collect values across all videos for this horizon
        video_values = {}
        
        for video_id, video_data in detailed_results.items():
            if horizon in video_data['horizon_results']:
                horizon_data = video_data['horizon_results'][horizon]
                
                for metric, value in horizon_data.items():
                    if isinstance(value, (int, float)) and metric != 'num_sequences':
                        if metric not in video_values:
                            video_values[metric] = []
                        video_values[metric].append(value)
        
        # Compute mean and std for each metric
        aggregated[horizon] = {
            'mean': {},
            'std': {},
            'num_videos': len(detailed_results)
        }
        
        for metric, values in video_values.items():
            aggregated[horizon]['mean'][metric] = np.mean(values)
            aggregated[horizon]['std'][metric] = np.std(values) if len(values) > 1 else 0.0
    
    return aggregated

def plot_additional_metrics(ax, horizon_data, horizons, horizon_seconds, font_size):
    """Plot additional metrics like exact match rate, sparsity, etc."""
    
    # Additional metrics to plot
    additional_metrics = {
        'Exact Match': {'key': 'exact_match_rate', 'color': '#FF6B6B', 'marker': 'o'},
        'Hamming Accuracy': {'key': 'hamming_accuracy', 'color': '#4ECDC4', 'marker': 's'},
        'Action Consistency': {'key': 'action_consistency', 'color': '#45B7D1', 'marker': '^'},
        # 'Temporal Smoothness': {'key': 'temporal_smoothness', 'color': '#96CEB4', 'marker': 'D'}
    }
    
    for metric_name, config in additional_metrics.items():
        means = [horizon_data[h]['mean'][config['key']] for h in horizons]
        stds = [horizon_data[h]['std'][config['key']] for h in horizons]
        
        # Plot main line
        ax.plot(horizon_seconds, means, 
               color=config['color'], marker=config['marker'],
               linewidth=2, markersize=6, label=metric_name, alpha=0.8)
        
        # Add error bars if multiple videos
        if any(s > 0 for s in stds):
            ax.errorbar(horizon_seconds, means, yerr=stds, 
                       color=config['color'], alpha=0.3, capsize=3)
    
    ax.set_xlabel('Planning Horizon (seconds)', fontsize=font_size, fontweight='bold')
    ax.set_ylabel('Score', fontsize=font_size, fontweight='bold')
    ax.set_title('Additional Performance Metrics', fontsize=font_size, fontweight='bold')
    ax.legend(fontsize=font_size-1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

def create_enhanced_stats_table(ax, horizon_data, horizons, num_videos, font_size):
    """Create an enhanced statistics table with key insights."""
    
    ax.axis('off')
    
    # Calculate key statistics
    overall_1s = horizon_data['1s']['mean']['ivt_mAP']
    overall_10s = horizon_data['10s']['mean']['ivt_mAP']
    degradation = (overall_1s - overall_10s) / overall_1s * 100
    
    exact_match_1s = horizon_data['1s']['mean']['exact_match_rate']
    exact_match_10s = horizon_data['10s']['mean']['exact_match_rate']
    match_degradation = (exact_match_1s - exact_match_10s) / exact_match_1s * 100
    
    # Find most robust component
    components = {'I': 'ivt_i_mAP', 'V': 'ivt_v_mAP', 'T': 'ivt_t_mAP'}
    component_degradations = {}
    for comp, key in components.items():
        val_1s = horizon_data['1s']['mean'][key]
        val_10s = horizon_data['10s']['mean'][key]
        component_degradations[comp] = (val_1s - val_10s) / val_1s * 100
    
    most_robust = min(component_degradations.keys(), key=lambda x: component_degradations[x])
    
    # Statistics text
    stats_text = f"""
KEY INSIGHTS

Overall IVT mAP:
1s: {overall_1s:.1%}
10s: {overall_10s:.1%}
Degradation: {degradation:.1f}%

Exact Match Rate:
1s: {exact_match_1s:.1%}
10s: {exact_match_10s:.1%}
Degradation: {match_degradation:.1f}%

Most Robust Component:
{most_robust} ({component_degradations[most_robust]:.1f}% loss)

Videos Analyzed: {num_videos}

Sequences per horizon:
{horizon_data['1s']['mean'].get('num_sequences', 'N/A')}
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=font_size-1,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

def add_performance_annotations(ax, horizon_data, horizons, horizon_seconds, font_size):
    """Add performance annotations to highlight key insights."""
    
    # Find steepest decline
    overall_scores = [horizon_data[h]['mean']['ivt_mAP'] for h in horizons]
    max_decline_idx = np.argmax(np.diff(overall_scores) * -1)  # Most negative diff
    
    if max_decline_idx < len(horizon_seconds) - 1:
        x_pos = horizon_seconds[max_decline_idx + 1]
        y_pos = overall_scores[max_decline_idx + 1]
        
        # Add annotation for steepest decline
        ax.annotate('Steepest\nDecline', 
                   xy=(x_pos, y_pos), 
                   xytext=(x_pos + 1.5, y_pos + 0.15),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=font_size-2, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                           edgecolor='red', alpha=0.8))

def add_significance_indicators(ax, horizon_data, horizons, horizon_seconds):
    """Add statistical significance indicators for multi-video analysis."""
    
    # This would be implemented when you have multiple videos
    # For now, we'll add a placeholder for when std > threshold
    for i, horizon in enumerate(horizons):
        std_val = horizon_data[horizon]['std'].get('ivt_mAP', 0)
        if std_val > 0.05:  # Significant variability
            ax.scatter(horizon_seconds[i], 0.95, marker='*', 
                      s=50, c='orange', alpha=0.8, zorder=10)

def add_background_shading(ax, horizon_seconds):
    """Add subtle background shading to separate time ranges."""
    
    # Short-term (1-2s)
    ax.axvspan(0.5, 2.5, alpha=0.05, color='green', zorder=0)
    ax.text(1.5, 0.02, 'Short-term', ha='center', va='bottom', 
           fontsize=9, alpha=0.6, style='italic')
    
    # Medium-term (3-5s)
    ax.axvspan(2.5, 5.5, alpha=0.05, color='orange', zorder=0)
    ax.text(4, 0.02, 'Medium-term', ha='center', va='bottom', 
           fontsize=9, alpha=0.6, style='italic')
    
    # Long-term (10s)
    ax.axvspan(5.5, 10.5, alpha=0.05, color='red', zorder=0)
    ax.text(8, 0.02, 'Long-term', ha='center', va='bottom', 
           fontsize=9, alpha=0.6, style='italic')

def plot_sparsity_analysis(planning_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a focused sparsity analysis plot showing prediction vs ground truth sparsity.
    """    
    detailed_results = planning_results['detailed_video_results']
    
    # Extract horizon keys from the first video and sort them
    video_ids = list(detailed_results.keys())
    first_video_results = detailed_results[video_ids[0]]['horizon_results']
    horizons = sorted(first_video_results.keys(), key=lambda x: float(x.rstrip('s')))
    horizon_seconds = [float(h.rstrip('s')) for h in horizons]
    
    horizon_data = aggregate_video_results(detailed_results, horizons)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Sparsity comparison
    pred_sparsity = [horizon_data[h]['mean']['pred_sparsity'] for h in horizons]
    gt_sparsity = [horizon_data[h]['mean']['gt_sparsity'] for h in horizons]
    sparsity_similarity = [horizon_data[h]['mean']['sparsity_similarity'] for h in horizons]
    
    ax1.plot(horizon_seconds, pred_sparsity, 'o-', label='Predicted Sparsity', 
            linewidth=2, markersize=6, color='#FF6B6B')
    ax1.plot(horizon_seconds, gt_sparsity, 's-', label='Ground Truth Sparsity', 
            linewidth=2, markersize=6, color='#4ECDC4')
    ax1.set_xlabel('Planning Horizon (seconds)', fontweight='bold')
    ax1.set_ylabel('Sparsity Ratio', fontweight='bold')
    ax1.set_title('Action Sparsity: Predicted vs Ground Truth', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sparsity similarity
    ax2.plot(horizon_seconds, sparsity_similarity, 'D-', label='Sparsity Similarity', 
            linewidth=2, markersize=6, color='#96CEB4')
    ax2.set_xlabel('Planning Horizon (seconds)', fontweight='bold')
    ax2.set_ylabel('Similarity Score', fontweight='bold')
    ax2.set_title('Sparsity Pattern Similarity', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.99, 1.001)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Sparsity analysis saved to: {save_path}")
    return
    
# Demo usage function
def demo_plot_with_sample_data():
    """Demo function showing how to use the plotting functions."""
    
    # Sample data structure matching the user's detailed results format
    sample_results = {
        'detailed_video_results': {
            'VID02': {
                'video_id': 'VID02',
                'horizon_results': {
                    '1s': {
                        'num_sequences': 2839, 'total_frames': 2839, 'horizon_frames': 1,
                        'ivt_mAP': 0.4243, 'ivt_i_mAP': 0.8518, 'ivt_v_mAP': 0.6296, 'ivt_t_mAP': 0.5168,
                        'ivt_iv_mAP': 0.4254, 'ivt_it_mAP': 0.4980,
                        'exact_match_rate': 0.4033, 'hamming_accuracy': 0.9901, 
                        'action_consistency': 0.9965, 
                        # 'temporal_smoothness': 0.9961,
                        'pred_sparsity': 0.0111, 'gt_sparsity': 0.0143, 'sparsity_similarity': 0.9967
                    },
                    '2s': {
                        'num_sequences': 2839, 'total_frames': 2839, 'horizon_frames': 2,
                        'ivt_mAP': 0.4025, 'ivt_i_mAP': 0.8097, 'ivt_v_mAP': 0.5803, 'ivt_t_mAP': 0.5169,
                        'ivt_iv_mAP': 0.3937, 'ivt_it_mAP': 0.4789,
                        'exact_match_rate': 0.3868, 'hamming_accuracy': 0.9898, 
                        'action_consistency': 0.9969, 
                        # 'temporal_smoothness': 0.9968,
                        'pred_sparsity': 0.0115, 'gt_sparsity': 0.0143, 'sparsity_similarity': 0.9972
                    },
                    '3s': {
                        'num_sequences': 2839, 'total_frames': 2839, 'horizon_frames': 3,
                        'ivt_mAP': 0.3753, 'ivt_i_mAP': 0.7683, 'ivt_v_mAP': 0.5383, 'ivt_t_mAP': 0.5044,
                        'ivt_iv_mAP': 0.3590, 'ivt_it_mAP': 0.4499,
                        'exact_match_rate': 0.3713, 'hamming_accuracy': 0.9895, 
                        'action_consistency': 0.9971, 
                        # 'temporal_smoothness': 0.9972,
                        'pred_sparsity': 0.0119, 'gt_sparsity': 0.0143, 'sparsity_similarity': 0.9975
                    },
                    '5s': {
                        'num_sequences': 2839, 'total_frames': 2839, 'horizon_frames': 5,
                        'ivt_mAP': 0.3443, 'ivt_i_mAP': 0.7223, 'ivt_v_mAP': 0.4903, 'ivt_t_mAP': 0.4771,
                        'ivt_iv_mAP': 0.3404, 'ivt_it_mAP': 0.4131,
                        'exact_match_rate': 0.3526, 'hamming_accuracy': 0.9890, 
                        'action_consistency': 0.9977, 
                        # 'temporal_smoothness': 0.9976,
                        'pred_sparsity': 0.0123, 'gt_sparsity': 0.0143, 'sparsity_similarity': 0.9980
                    },
                    '10s': {
                        'num_sequences': 2839, 'total_frames': 2839, 'horizon_frames': 10,
                        'ivt_mAP': 0.2917, 'ivt_i_mAP': 0.6529, 'ivt_v_mAP': 0.4200, 'ivt_t_mAP': 0.4037,
                        'ivt_iv_mAP': 0.3114, 'ivt_it_mAP': 0.3471,
                        'exact_match_rate': 0.3142, 'hamming_accuracy': 0.9879, 
                        'action_consistency': 0.9981, 
                        # 'temporal_smoothness': 0.9979,
                        'pred_sparsity': 0.0126, 'gt_sparsity': 0.0143, 'sparsity_similarity': 0.9982
                    }
                }
            }
        }
    }
    
    # Create the main plot with additional metrics
    print("Creating enhanced mAP vs horizon plot...")
    fig1 = plot_map_vs_horizon(sample_results, 
                              save_path='enhanced_map_vs_horizon.png',
                              style='paper',
                              include_additional_metrics=True)
    
    # Create simple plot without additional metrics
    print("Creating simple mAP vs horizon plot...")
    fig2 = plot_map_vs_horizon(sample_results,
                              save_path='simple_map_vs_horizon.png', 
                              style='paper',
                              include_additional_metrics=False)
    print("Demo plots created successfully!")

if __name__ == "__main__":
    # Run demo
    demo_plot_with_sample_data()
    
    # Example usage with your actual data:
    """
    # Load your planning results with detailed video results
    import pickle
    with open('planning_results.pkl', 'rb') as f:
        planning_results = pickle.load(f)
    
    # Create enhanced publication plot with additional metrics
    fig = plot_map_vs_horizon(planning_results, 
                             save_path='miccai_enhanced_analysis.png',
                             style='paper',
                             include_additional_metrics=True)
    
    # Create simple mAP plot for main paper figure
    fig_simple = plot_map_vs_horizon(planning_results,
                                    save_path='miccai_map_simple.png',
                                    style='paper', 
                                    include_additional_metrics=False)
    """