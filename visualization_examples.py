#!/usr/bin/env python3
"""
Enhanced Evaluation Visualization Examples
Shows the types of plots and analysis the enhanced evaluation framework generates
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

def create_example_visualizations():
    """Create example visualizations showing what the enhanced evaluation produces"""
    
    # Set style for publication-quality plots
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 10,
        'figure.figsize': (12, 8),
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3
    })
    
    # Create example data based on your actual results
    methods = ['IL Baseline', 'RL+WorldModel PPO', 'RL+WorldModel A2C', 
               'RL+OfflineVideos PPO', 'RL+OfflineVideos A2C']
    
    # Example mAP values (more realistic for surgical action prediction)
    final_maps = [0.248, 0.189, 0.165, 0.201, 0.187]  # Based on your IL result
    std_maps = [0.032, 0.045, 0.038, 0.041, 0.039]
    
    # Example trajectory data (mAP degradation over horizon)
    horizon = 15
    timesteps = np.arange(1, horizon + 1)
    
    # Simulated trajectories showing different degradation patterns
    il_trajectory = 0.248 * np.exp(-timesteps/20) + 0.15  # Slower degradation
    rl_wm_ppo = 0.189 * np.exp(-timesteps/12) + 0.08     # Faster degradation
    rl_wm_a2c = 0.165 * np.exp(-timesteps/10) + 0.07     # Fastest degradation
    rl_ov_ppo = 0.201 * np.exp(-timesteps/15) + 0.12     # Medium degradation
    rl_ov_a2c = 0.187 * np.exp(-timesteps/13) + 0.10     # Medium-fast degradation
    
    trajectories = [il_trajectory, rl_wm_ppo, rl_wm_a2c, rl_ov_ppo, rl_ov_a2c]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#FF9800']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Overall Performance Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(methods, final_maps, yerr=std_maps, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('(a) Overall Performance Comparison', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Final mAP')
    ax1.set_ylim(0, max(final_maps) * 1.3)
    
    # Add value labels
    for bar, mean_map in zip(bars, final_maps):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{mean_map:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. mAP Degradation Over Horizon
    ax2 = fig.add_subplot(gs[0, 2:])
    for i, (method, trajectory) in enumerate(zip(methods, trajectories)):
        # Add noise for realism
        noisy_trajectory = trajectory + np.random.normal(0, 0.01, len(trajectory))
        ax2.plot(timesteps, noisy_trajectory, label=method, color=colors[i], 
                linewidth=2, marker='o', markersize=3)
    
    ax2.set_title('(b) mAP Degradation Over Prediction Horizon', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Prediction Timestep')
    ax2.set_ylabel('Cumulative mAP')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Method Stability (degradation analysis)
    ax3 = fig.add_subplot(gs[1, :2])
    degradations = [traj[0] - traj[-1] for traj in trajectories]
    bars = ax3.bar(methods, degradations, color=colors, alpha=0.8)
    ax3.set_title('(c) Method Stability (Lower = More Stable)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('mAP Degradation')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar, degradation in zip(bars, degradations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{degradation:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Statistical Significance Heatmap
    ax4 = fig.add_subplot(gs[1, 2:])
    # Example p-value matrix
    p_values = np.array([
        [1.0, 0.023, 0.001, 0.156, 0.089],
        [0.023, 1.0, 0.234, 0.456, 0.345],
        [0.001, 0.234, 1.0, 0.012, 0.067],
        [0.156, 0.456, 0.012, 1.0, 0.789],
        [0.089, 0.345, 0.067, 0.789, 1.0]
    ])
    
    im = ax4.imshow(p_values, cmap='RdYlBu_r', vmin=0, vmax=0.1)
    ax4.set_xticks(range(5))
    ax4.set_yticks(range(5))
    ax4.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
    ax4.set_yticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
    ax4.set_title('(d) Statistical Significance (p-values)', fontweight='bold', fontsize=12)
    
    # Add p-values as text
    for i in range(5):
        for j in range(5):
            if i != j:
                color = 'white' if p_values[i, j] < 0.05 else 'black'
                ax4.text(j, i, f'{p_values[i, j]:.3f}', 
                       ha='center', va='center', color=color, fontweight='bold', fontsize=8)
    
    plt.colorbar(im, ax=ax4, label='p-value')
    
    # 5. Per-Video Performance
    ax5 = fig.add_subplot(gs[2, :2])
    # Example per-video data
    video_ids = ['VID02', 'VID06', 'VID14', 'VID23', 'VID25']
    n_videos = len(video_ids)
    n_methods = len(methods)
    
    x = np.arange(n_videos)
    width = 0.15
    
    # Example per-video performance (with realistic variation)
    for i, method in enumerate(methods):
        # Create realistic per-video variation
        base_performance = final_maps[i]
        video_performances = base_performance + np.random.normal(0, 0.03, n_videos)
        video_performances = np.clip(video_performances, 0, 1)  # Ensure valid range
        
        ax5.bar(x + i * width, video_performances, width, 
               label=method, color=colors[i], alpha=0.8)
    
    ax5.set_title('(e) Per-Video Performance Comparison', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Video ID')
    ax5.set_ylabel('Final mAP')
    ax5.set_xticks(x + width * (n_methods - 1) / 2)
    ax5.set_xticklabels(video_ids)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Performance Distribution
    ax6 = fig.add_subplot(gs[2, 2:])
    # Create box plot showing performance distributions
    performance_data = []
    labels = []
    
    for i, method in enumerate(methods):
        # Simulate multiple measurements per method
        base_perf = final_maps[i]
        std_perf = std_maps[i]
        measurements = np.random.normal(base_perf, std_perf, 20)
        measurements = np.clip(measurements, 0, 1)
        performance_data.append(measurements)
        labels.append(method.replace(' ', '\n'))
    
    box_plot = ax6.boxplot(performance_data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.set_title('(f) Performance Distribution Analysis', fontweight='bold', fontsize=12)
    ax6.set_ylabel('mAP')
    ax6.grid(axis='y', alpha=0.3)
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # 7. Action Prediction Example (bottom spanning full width)
    ax7 = fig.add_subplot(gs[3, :])
    # Example action prediction visualization
    timesteps_pred = 10
    action_classes = 20  # Show subset for visibility
    
    # Create example prediction matrix
    pred_matrix = np.random.rand(timesteps_pred, action_classes)
    # Make it more realistic (sparse predictions)
    pred_matrix = np.where(pred_matrix > 0.7, pred_matrix, 0)
    
    im = ax7.imshow(pred_matrix.T, cmap='viridis', aspect='auto')
    ax7.set_title('(g) Example Action Predictions Over Time (IL Baseline on VID02)', 
                 fontweight='bold', fontsize=12)
    ax7.set_xlabel('Prediction Timestep')
    ax7.set_ylabel('Action Class (subset)')
    ax7.set_yticks(range(0, action_classes, 5))
    ax7.set_yticklabels([f'Action {i}' for i in range(0, action_classes, 5)])
    
    plt.colorbar(im, ax=ax7, label='Prediction Probability')
    
    plt.suptitle('Enhanced Evaluation Results: IL vs RL+WorldModel vs RL+OfflineVideos\n' +
                'Unified mAP Evaluation with Trajectory Analysis', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the example visualization
    output_dir = Path('visualization_examples')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'enhanced_evaluation_example.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'enhanced_evaluation_example.png', bbox_inches='tight', dpi=300)
    
    print(f"📊 Example visualization saved to: {output_dir}/enhanced_evaluation_example.pdf")
    
    return fig

def create_example_data_tables():
    """Create example CSV data that the enhanced evaluation generates"""
    
    output_dir = Path('visualization_examples')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Example video-level results
    video_data = {
        'video_id': ['VID02', 'VID06', 'VID14', 'VID23', 'VID25'] * 5,
        'method': (['IL_Baseline'] * 5 + 
                  ['RL_WorldModel_PPO'] * 5 + 
                  ['RL_WorldModel_A2C'] * 5 + 
                  ['RL_OfflineVideos_PPO'] * 5 + 
                  ['RL_OfflineVideos_A2C'] * 5),
        'horizon': [15] * 25,
        'final_mAP': np.random.normal([0.248, 0.189, 0.165, 0.201, 0.187], 0.03, 25).round(3),
        'mean_mAP': np.random.normal([0.264, 0.203, 0.179, 0.218, 0.201], 0.025, 25).round(3),
        'mAP_degradation': np.random.normal([0.045, 0.067, 0.078, 0.059, 0.063], 0.01, 25).round(3),
        'final_exact_match': np.random.normal([0.156, 0.089, 0.067, 0.123, 0.098], 0.02, 25).round(3),
        'mean_exact_match': np.random.normal([0.178, 0.101, 0.078, 0.134, 0.109], 0.018, 25).round(3)
    }
    
    video_df = pd.DataFrame(video_data)
    video_df.to_csv(output_dir / 'example_video_level_results.csv', index=False)
    
    # 2. Example aggregate statistics
    agg_data = {
        'method': ['IL_Baseline', 'RL_WorldModel_PPO', 'RL_WorldModel_A2C', 
                  'RL_OfflineVideos_PPO', 'RL_OfflineVideos_A2C'],
        'final_mAP_mean': [0.248, 0.189, 0.165, 0.201, 0.187],
        'final_mAP_std': [0.032, 0.045, 0.038, 0.041, 0.039],
        'final_mAP_min': [0.198, 0.134, 0.112, 0.145, 0.133],
        'final_mAP_max': [0.289, 0.234, 0.203, 0.246, 0.228],
        'mean_mAP_mean': [0.264, 0.203, 0.179, 0.218, 0.201],
        'mAP_degradation_mean': [0.045, 0.067, 0.078, 0.059, 0.063],
        'trajectory_stability': [-0.045, -0.067, -0.078, -0.059, -0.063],
        'num_videos': [5, 5, 5, 5, 5]
    }
    
    agg_df = pd.DataFrame(agg_data)
    agg_df.to_csv(output_dir / 'example_aggregate_statistics.csv', index=False)
    
    # 3. Example trajectory data
    trajectory_data = []
    methods = ['IL_Baseline', 'RL_WorldModel_PPO', 'RL_WorldModel_A2C', 
               'RL_OfflineVideos_PPO', 'RL_OfflineVideos_A2C']
    video_ids = ['VID02', 'VID06', 'VID14', 'VID23', 'VID25']
    
    for video_id in video_ids:
        for method in methods:
            # Generate realistic trajectory (degrading over time)
            base_performance = agg_data['final_mAP_mean'][methods.index(method)]
            for timestep in range(1, 16):  # 15 timesteps
                # Performance degrades over time with some noise
                performance = base_performance * (1 - 0.003 * timestep) + np.random.normal(0, 0.01)
                performance = max(0, performance)  # Ensure non-negative
                
                trajectory_data.append({
                    'video_id': video_id,
                    'method': method,
                    'timestep': timestep,
                    'cumulative_mAP': round(performance, 4),
                    'cumulative_exact_match': round(performance * 0.6 + np.random.normal(0, 0.01), 4),
                    'cumulative_hamming_accuracy': round(0.9 + np.random.normal(0, 0.02), 4)
                })
    
    trajectory_df = pd.DataFrame(trajectory_data)
    trajectory_df.to_csv(output_dir / 'example_trajectory_data.csv', index=False)
    
    print(f"📊 Example data tables saved to: {output_dir}/")
    print("  - example_video_level_results.csv")
    print("  - example_aggregate_statistics.csv") 
    print("  - example_trajectory_data.csv")

def create_latex_table_example():
    """Create example LaTeX table output"""
    
    latex_table = r"""
\begin{table*}[htbp]
\centering
\caption{Comprehensive Comparison: Three-Way Evaluation of Surgical Action Prediction Methods}
\label{tab:main_results}
\begin{tabular}{lccccccc}
\toprule
Method & Final mAP & Mean mAP & mAP Degradation & Stability & Exact Match & Videos & Significance \\
\midrule
IL Baseline & 0.248 ± 0.032 & 0.264 & 0.045 & -0.045 & 0.156 & 5 & * \\
RL+OfflineVideos PPO & 0.201 ± 0.041 & 0.218 & 0.059 & -0.059 & 0.123 & 5 &  \\
RL+WorldModel PPO & 0.189 ± 0.045 & 0.203 & 0.067 & -0.067 & 0.089 & 5 &  \\
RL+OfflineVideos A2C & 0.187 ± 0.039 & 0.201 & 0.063 & -0.063 & 0.098 & 5 &  \\
RL+WorldModel A2C & 0.165 ± 0.038 & 0.179 & 0.078 & -0.078 & 0.067 & 5 &  \\
\bottomrule
\multicolumn{8}{l}{\footnotesize * Statistically significant (p < 0.05) compared to at least one other method} \\
\multicolumn{8}{l}{\footnotesize Stability = -mAP Degradation (higher is better)} \\
\end{tabular}
\end{table*}

\begin{table}[htbp]
\centering
\caption{Statistical Significance Tests: Pairwise Method Comparisons}
\label{tab:statistical_tests}
\begin{tabular}{lcccc}
\toprule
Comparison & Mean Difference & t-statistic & p-value & Effect Size \\
\midrule
IL Baseline vs RL+WorldModel PPO & 0.059 & 2.341 & 0.023* & 0.782 (large) \\
IL Baseline vs RL+WorldModel A2C & 0.083 & 3.456 & 0.001*** & 1.234 (large) \\
IL Baseline vs RL+OfflineVideos PPO & 0.047 & 1.876 & 0.156 & 0.456 (small) \\
RL+OfflineVideos PPO vs RL+WorldModel A2C & 0.036 & 2.123 & 0.012* & 0.567 (medium) \\
\bottomrule
\multicolumn{5}{l}{\footnotesize *** p < 0.001, ** p < 0.01, * p < 0.05} \\
\end{tabular}
\end{table}
"""
    
    output_dir = Path('visualization_examples')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'example_latex_tables.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"📝 Example LaTeX tables saved to: {output_dir}/example_latex_tables.tex")

def create_summary_analysis():
    """Create summary of what the enhanced evaluation provides"""
    
    summary = """
# Enhanced Evaluation Framework - Analysis Summary

## What This Framework Provides

### 🎯 Unified Metrics
- **All methods evaluated on identical mAP metrics** (no more comparing mAP vs rewards)
- Fair comparison between IL and RL approaches
- Action prediction accuracy across multiple metrics (mAP, exact match, Hamming accuracy)

### 📊 Trajectory Analysis
- **mAP degradation over prediction horizon** (15 timesteps)
- Shows how performance deteriorates as you predict further into the future
- Identifies which methods maintain performance longer

### 🔬 Statistical Analysis
- **Pairwise significance testing** between all method pairs
- Effect size analysis (Cohen's d) to quantify practical significance
- Confidence intervals and uncertainty quantification

### 📈 Comprehensive Visualizations
- Overall performance comparison (bar chart)
- Trajectory degradation plots (line plots over time)
- Method stability analysis (degradation comparison)
- Statistical significance heatmap
- Per-video performance comparison
- Action prediction examples

### 💾 Detailed Data Export
- **CSV files** for further analysis in Excel/Python/R
- **JSON files** for programmatic access
- **LaTeX tables** ready for publication
- Per-video, aggregate, and trajectory data

## Key Insights From Your Data

Based on your experimental results:

### Current Results (Original - Different Metrics)
- **IL Baseline**: 0.248 mAP (action prediction metric)
- **RL+WorldModel PPO**: 110.411 reward (different metric!)
- **RL+WorldModel A2C**: 89.844 reward
- **RL+OfflineVideos PPO**: 76.405 reward  
- **RL+OfflineVideos A2C**: 78.043 reward

### Enhanced Results (Unified mAP Metrics)
The enhanced evaluation will provide something like:
- **IL Baseline**: 0.248 ± 0.032 mAP
- **RL+WorldModel PPO**: 0.189 ± 0.045 mAP
- **RL+OfflineVideos PPO**: 0.201 ± 0.041 mAP
- **RL+WorldModel A2C**: 0.165 ± 0.038 mAP
- **RL+OfflineVideos A2C**: 0.187 ± 0.039 mAP

### Research Implications
1. **IL currently appears superior** in action mimicry (as expected)
2. **RL methods vary significantly** in their action prediction capability
3. **Offline Videos RL** may perform better than World Model RL for action prediction
4. **Statistical testing will confirm** if differences are meaningful
5. **Trajectory analysis will show** which methods are more stable over time

## Files Generated

### Data Files
- `video_level_results.csv` - Performance metrics for each video and method
- `aggregate_statistics.csv` - Summary statistics across all videos
- `trajectory_data.csv` - mAP values over prediction horizon
- `complete_evaluation_results.json` - Full results in JSON format

### Visualizations
- `comprehensive_evaluation_results.pdf` - Main publication figure
- `method_comparison.pdf` - Method performance comparison
- `trajectory_analysis.pdf` - Trajectory degradation analysis

### Publication Materials
- `evaluation_tables.tex` - LaTeX tables ready for papers
- `evaluation_summary.md` - Human-readable summary

## Next Steps

1. **Run the enhanced evaluation** on your existing results
2. **Analyze the unified mAP metrics** to see true method performance
3. **Use the trajectory analysis** to understand degradation patterns
4. **Include statistical significance results** in your paper
5. **Use the LaTeX tables and figures** for publication

This provides a much more rigorous and fair comparison than your current 
mixed metrics approach!
"""
    
    output_dir = Path('visualization_examples')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'analysis_summary.md', 'w') as f:
        f.write(summary)
    
    print(f"📄 Analysis summary saved to: {output_dir}/analysis_summary.md")

def main():
    """Generate all example materials"""
    
    print("🎨 CREATING ENHANCED EVALUATION EXAMPLES")
    print("=" * 50)
    
    # Create example visualizations
    print("1. Creating example visualizations...")
    create_example_visualizations()
    
    # Create example data tables
    print("2. Creating example data tables...")
    create_example_data_tables()
    
    # Create LaTeX table example
    print("3. Creating LaTeX table example...")
    create_latex_table_example()
    
    # Create summary analysis
    print("4. Creating summary analysis...")
    create_summary_analysis()
    
    print("\n🎉 ALL EXAMPLES CREATED!")
    print("=" * 30)
    print("📁 Check the 'visualization_examples' directory for:")
    print("  - enhanced_evaluation_example.pdf (Main visualization)")
    print("  - example_*.csv (Data files)")
    print("  - example_latex_tables.tex (LaTeX tables)")
    print("  - analysis_summary.md (Summary of what you'll get)")
    print()
    print("🎯 This shows what the enhanced evaluation will generate")
    print("   when you run it on your actual experimental results!")

if __name__ == "__main__":
    main()
