#!/usr/bin/env python3
"""
Reward Values Visualization Tool for CholecT50 Dataset
Plots and analyzes reward values during data loading to verify scaling and correctness
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import json
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RewardAnalyzer:
    """
    Comprehensive reward analysis and visualization tool.
    """
    
    def __init__(self, save_dir: str = "reward_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Storage for all reward data
        self.all_rewards = {}
        self.all_actions = {}
        self.all_phases = {}
        self.video_metadata = {}
        
        print(f"📊 Reward Analyzer initialized")
        print(f"💾 Saving plots to: {self.save_dir}")

    def add_video_data(self, video_data: Dict[str, Any]):
        """
        Add a video's data for analysis.
        
        Args:
            video_data: Single video dictionary from load_cholect50_data
        """
        
        video_id = video_data.get('video_id', 'unknown')
        
        # Extract rewards
        rewards = video_data.get('rewards', video_data.get('next_rewards', {}))
        actions = video_data.get('actions_binaries', np.array([]))
        phases = video_data.get('phase_binaries', np.array([]))
        
        # Store data
        self.all_rewards[video_id] = rewards
        self.all_actions[video_id] = actions
        self.all_phases[video_id] = phases
        
        # Store metadata
        self.video_metadata[video_id] = {
            'num_frames': video_data.get('num_frames', len(actions) if len(actions) > 0 else 0),
            'embedding_dim': video_data.get('frame_embeddings', np.array([])).shape[-1] if len(video_data.get('frame_embeddings', [])) > 0 else 0
        }
        
        print(f"✅ Added video {video_id} to reward analysis")

    def analyze_all_videos(self, video_data_list: List[Dict[str, Any]]):
        """
        Analyze reward values for all videos.
        
        Args:
            video_data_list: List of video dictionaries from load_cholect50_data
        """
        
        print(f"\n📊 ANALYZING REWARDS FOR {len(video_data_list)} VIDEOS")
        print("=" * 50)
        
        # Add all video data
        for video_data in video_data_list:
            self.add_video_data(video_data)
        
        # Run comprehensive analysis
        self._plot_reward_distributions()
        self._plot_reward_time_series()
        self._plot_reward_statistics()
        self._plot_action_reward_correlations()
        self._analyze_reward_scales()
        self._save_summary_statistics()
        
        print(f"✅ Reward analysis complete!")
        print(f"📁 All plots saved to: {self.save_dir}")

    def _plot_reward_distributions(self):
        """Plot distribution of reward values across all videos."""
        
        print("📊 Plotting reward distributions...")
        
        # Collect all reward types
        all_reward_types = set()
        for rewards in self.all_rewards.values():
            all_reward_types.update(rewards.keys())
        
        all_reward_types = sorted(list(all_reward_types))
        
        if not all_reward_types:
            print("⚠️ No reward types found!")
            return
        
        # Create subplots
        n_rewards = len(all_reward_types)
        n_cols = 3
        n_rows = (n_rewards + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, reward_type in enumerate(all_reward_types):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Collect all values for this reward type
            all_values = []
            for video_id, rewards in self.all_rewards.items():
                if reward_type in rewards:
                    values = np.array(rewards[reward_type])
                    if values.size > 0:
                        all_values.extend(values.flatten())
            
            if all_values:
                all_values = np.array(all_values)
                
                # Plot histogram
                ax.hist(all_values, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'{reward_type}\nMean: {np.mean(all_values):.4f}, Std: {np.std(all_values):.4f}')
                ax.set_xlabel('Reward Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f"Min: {np.min(all_values):.4f}\nMax: {np.max(all_values):.4f}\nMed: {np.median(all_values):.4f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{reward_type}\n(No data)')
        
        # Remove empty subplots
        for i in range(len(all_reward_types), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'reward_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Reward distributions saved")

    def _plot_reward_time_series(self):
        """Plot time series of rewards for each video."""
        
        print("📊 Plotting reward time series...")
        
        for video_id, rewards in self.all_rewards.items():
            if not rewards:
                continue
                
            reward_types = list(rewards.keys())
            n_rewards = len(reward_types)
            
            if n_rewards == 0:
                continue
            
            # Create subplots for this video
            n_cols = 2
            n_rows = (n_rewards + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for i, reward_type in enumerate(reward_types):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                reward_values = np.array(rewards[reward_type])
                
                if reward_values.size > 0:
                    # Plot time series
                    ax.plot(reward_values, linewidth=1, alpha=0.8)
                    ax.set_title(f'{reward_type}')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Reward Value')
                    ax.grid(True, alpha=0.3)
                    
                    # Highlight non-zero values
                    non_zero_frames = np.where(np.abs(reward_values) > 1e-6)[0]
                    if len(non_zero_frames) > 0:
                        ax.scatter(non_zero_frames, reward_values[non_zero_frames], 
                                 color='red', s=10, alpha=0.6, label='Non-zero')
                        ax.legend()
                    
                    # Add statistics
                    stats_text = f"Mean: {np.mean(reward_values):.4f}\nNon-zero: {len(non_zero_frames)}/{len(reward_values)}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{reward_type} (No data)')
            
            # Remove empty subplots
            for i in range(len(reward_types), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.suptitle(f'Reward Time Series - {video_id}', fontsize=16, y=0.98)
            plt.savefig(self.save_dir / f'reward_timeseries_{video_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Time series plots saved for {len(self.all_rewards)} videos")

    def _plot_reward_statistics(self):
        """Plot comprehensive reward statistics."""
        
        print("📊 Plotting reward statistics...")
        
        # Collect statistics for all reward types
        reward_stats = {}
        
        all_reward_types = set()
        for rewards in self.all_rewards.values():
            all_reward_types.update(rewards.keys())
        
        for reward_type in all_reward_types:
            stats = {
                'means': [],
                'stds': [],
                'mins': [],
                'maxs': [],
                'non_zero_counts': [],
                'total_counts': []
            }
            
            for video_id, rewards in self.all_rewards.items():
                if reward_type in rewards:
                    values = np.array(rewards[reward_type])
                    if values.size > 0:
                        stats['means'].append(np.mean(values))
                        stats['stds'].append(np.std(values))
                        stats['mins'].append(np.min(values))
                        stats['maxs'].append(np.max(values))
                        stats['non_zero_counts'].append(np.sum(np.abs(values) > 1e-6))
                        stats['total_counts'].append(len(values))
            
            reward_stats[reward_type] = stats
        
        # Create summary plot
        if reward_stats:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Mean values across videos
            ax1 = axes[0, 0]
            reward_names = []
            mean_values = []
            std_errors = []
            
            for reward_type, stats in reward_stats.items():
                if stats['means']:
                    reward_names.append(reward_type.replace('_r_', '').replace('_', '\n'))
                    mean_values.append(np.mean(stats['means']))
                    std_errors.append(np.std(stats['means']))
            
            if reward_names:
                bars = ax1.bar(range(len(reward_names)), mean_values, yerr=std_errors, 
                              alpha=0.7, capsize=5)
                ax1.set_xlabel('Reward Type')
                ax1.set_ylabel('Mean Reward Value')
                ax1.set_title('Average Reward Values Across Videos')
                ax1.set_xticks(range(len(reward_names)))
                ax1.set_xticklabels(reward_names, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                
                # Color bars based on magnitude
                for bar, val in zip(bars, mean_values):
                    if abs(val) > 1.0:
                        bar.set_color('red')
                    elif abs(val) > 0.1:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
            
            # 2. Value ranges
            ax2 = axes[0, 1]
            min_values = []
            max_values = []
            
            for reward_type in reward_names:
                original_name = '_r_' + reward_type.replace('\n', '_')
                if original_name in reward_stats and reward_stats[original_name]['mins']:
                    min_values.append(np.min(reward_stats[original_name]['mins']))
                    max_values.append(np.max(reward_stats[original_name]['maxs']))
                else:
                    min_values.append(0)
                    max_values.append(0)
            
            if min_values and max_values:
                x_pos = range(len(reward_names))
                ax2.bar(x_pos, max_values, alpha=0.7, label='Max', color='lightcoral')
                ax2.bar(x_pos, min_values, alpha=0.7, label='Min', color='lightblue')
                ax2.set_xlabel('Reward Type')
                ax2.set_ylabel('Reward Value Range')
                ax2.set_title('Reward Value Ranges')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(reward_names, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. Non-zero frequency
            ax3 = axes[1, 0]
            non_zero_fractions = []
            
            for reward_type in reward_names:
                original_name = '_r_' + reward_type.replace('\n', '_')
                if original_name in reward_stats:
                    stats = reward_stats[original_name]
                    if stats['non_zero_counts'] and stats['total_counts']:
                        total_non_zero = sum(stats['non_zero_counts'])
                        total_frames = sum(stats['total_counts'])
                        non_zero_fractions.append(total_non_zero / total_frames if total_frames > 0 else 0)
                    else:
                        non_zero_fractions.append(0)
                else:
                    non_zero_fractions.append(0)
            
            if non_zero_fractions:
                bars = ax3.bar(range(len(reward_names)), non_zero_fractions, alpha=0.7)
                ax3.set_xlabel('Reward Type')
                ax3.set_ylabel('Fraction of Non-Zero Values')
                ax3.set_title('Reward Sparsity (Non-Zero Frequency)')
                ax3.set_xticks(range(len(reward_names)))
                ax3.set_xticklabels(reward_names, rotation=45, ha='right')
                ax3.set_ylim([0, 1])
                ax3.grid(True, alpha=0.3)
                
                # Color bars based on sparsity
                for bar, frac in zip(bars, non_zero_fractions):
                    if frac > 0.5:
                        bar.set_color('green')
                    elif frac > 0.1:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            
            # 4. Standard deviations
            ax4 = axes[1, 1]
            std_values = []
            
            for reward_type in reward_names:
                original_name = '_r_' + reward_type.replace('\n', '_')
                if original_name in reward_stats and reward_stats[original_name]['stds']:
                    std_values.append(np.mean(reward_stats[original_name]['stds']))
                else:
                    std_values.append(0)
            
            if std_values:
                ax4.bar(range(len(reward_names)), std_values, alpha=0.7, color='purple')
                ax4.set_xlabel('Reward Type')
                ax4.set_ylabel('Average Standard Deviation')
                ax4.set_title('Reward Variability')
                ax4.set_xticks(range(len(reward_names)))
                ax4.set_xticklabels(reward_names, rotation=45, ha='right')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'reward_statistics_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Reward statistics summary saved")

    def _plot_action_reward_correlations(self):
        """Plot correlations between actions and rewards."""
        
        print("📊 Plotting action-reward correlations...")
        
        for video_id in self.all_rewards.keys():
            rewards = self.all_rewards[video_id]
            actions = self.all_actions.get(video_id, np.array([]))
            
            if len(rewards) == 0 or actions.size == 0:
                continue
            
            # Calculate action counts per frame
            action_counts = np.sum(actions, axis=1) if actions.ndim > 1 else actions
            
            # Create correlation plot
            n_rewards = len(rewards)
            if n_rewards == 0:
                continue
            
            n_cols = 2
            n_rows = (n_rewards + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for i, (reward_type, reward_values) in enumerate(rewards.items()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                reward_values = np.array(reward_values)
                
                if reward_values.size > 0 and len(action_counts) == len(reward_values):
                    # Scatter plot
                    ax.scatter(action_counts, reward_values, alpha=0.6, s=10)
                    ax.set_xlabel('Number of Actions')
                    ax.set_ylabel('Reward Value')
                    ax.set_title(f'{reward_type}')
                    ax.grid(True, alpha=0.3)
                    
                    # Calculate correlation
                    if np.std(action_counts) > 0 and np.std(reward_values) > 0:
                        correlation = np.corrcoef(action_counts, reward_values)[0, 1]
                        ax.text(0.02, 0.98, f'Corr: {correlation:.3f}', transform=ax.transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Add trend line
                    if len(np.unique(action_counts)) > 1:
                        z = np.polyfit(action_counts, reward_values, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(np.min(action_counts), np.max(action_counts), 100)
                        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)
                else:
                    ax.text(0.5, 0.5, 'Data mismatch', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{reward_type} (Data mismatch)')
            
            # Remove empty subplots
            for i in range(len(rewards), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.suptitle(f'Action-Reward Correlations - {video_id}', fontsize=16, y=0.98)
            plt.savefig(self.save_dir / f'action_reward_correlations_{video_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Action-reward correlation plots saved")

    def _analyze_reward_scales(self):
        """Analyze and report reward scaling issues."""
        
        print("📊 Analyzing reward scales...")
        
        scale_analysis = {
            'summary': {},
            'potential_issues': [],
            'recommendations': []
        }
        
        # Analyze each reward type
        all_reward_types = set()
        for rewards in self.all_rewards.values():
            all_reward_types.update(rewards.keys())
        
        for reward_type in all_reward_types:
            all_values = []
            for rewards in self.all_rewards.values():
                if reward_type in rewards:
                    values = np.array(rewards[reward_type])
                    if values.size > 0:
                        all_values.extend(values.flatten())
            
            if all_values:
                all_values = np.array(all_values)
                
                stats = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values),
                    'min': np.min(all_values),
                    'max': np.max(all_values),
                    'abs_max': np.max(np.abs(all_values)),
                    'non_zero_fraction': np.sum(np.abs(all_values) > 1e-6) / len(all_values),
                    'range': np.max(all_values) - np.min(all_values)
                }
                
                scale_analysis['summary'][reward_type] = stats
                
                # Check for potential issues
                if stats['abs_max'] > 100:
                    scale_analysis['potential_issues'].append(
                        f"⚠️ {reward_type}: Very large values (max: {stats['abs_max']:.2f}) - may cause RL instability"
                    )
                    scale_analysis['recommendations'].append(
                        f"Consider normalizing {reward_type} to [-1, 1] or [-10, 10] range"
                    )
                
                if stats['abs_max'] < 0.001:
                    scale_analysis['potential_issues'].append(
                        f"⚠️ {reward_type}: Very small values (max: {stats['abs_max']:.6f}) - may be ignored by RL"
                    )
                    scale_analysis['recommendations'].append(
                        f"Consider scaling up {reward_type} by factor of 100-1000"
                    )
                
                if stats['non_zero_fraction'] < 0.01:
                    scale_analysis['potential_issues'].append(
                        f"⚠️ {reward_type}: Very sparse ({stats['non_zero_fraction']*100:.1f}% non-zero) - may not provide enough signal"
                    )
                
                if stats['range'] > 1000:
                    scale_analysis['potential_issues'].append(
                        f"⚠️ {reward_type}: Very large range ({stats['range']:.2f}) - may cause value function issues"
                    )
        
        # Save scale analysis
        with open(self.save_dir / 'reward_scale_analysis.json', 'w') as f:
            json.dump(scale_analysis, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📋 REWARD SCALE ANALYSIS SUMMARY")
        print("=" * 40)
        
        if scale_analysis['potential_issues']:
            print("🚨 POTENTIAL ISSUES FOUND:")
            for issue in scale_analysis['potential_issues']:
                print(f"   {issue}")
        else:
            print("✅ No major scaling issues detected")
        
        if scale_analysis['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in scale_analysis['recommendations']:
                print(f"   {rec}")
        
        print(f"\n📄 Detailed analysis saved to: reward_scale_analysis.json")

    def _save_summary_statistics(self):
        """Save comprehensive summary statistics."""
        
        print("📊 Saving summary statistics...")
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_videos': len(self.all_rewards),
            'video_metadata': self.video_metadata,
            'reward_types_found': [],
            'overall_statistics': {}
        }
        
        # Collect all reward types
        all_reward_types = set()
        for rewards in self.all_rewards.values():
            all_reward_types.update(rewards.keys())
        
        summary['reward_types_found'] = sorted(list(all_reward_types))
        
        # Calculate overall statistics
        for reward_type in all_reward_types:
            all_values = []
            video_stats = []
            
            for video_id, rewards in self.all_rewards.items():
                if reward_type in rewards:
                    values = np.array(rewards[reward_type])
                    if values.size > 0:
                        all_values.extend(values.flatten())
                        video_stats.append({
                            'video_id': video_id,
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'non_zero_count': np.sum(np.abs(values) > 1e-6),
                            'total_count': len(values)
                        })
            
            if all_values:
                all_values = np.array(all_values)
                summary['overall_statistics'][reward_type] = {
                    'global_mean': float(np.mean(all_values)),
                    'global_std': float(np.std(all_values)),
                    'global_min': float(np.min(all_values)),
                    'global_max': float(np.max(all_values)),
                    'global_median': float(np.median(all_values)),
                    'global_non_zero_fraction': float(np.sum(np.abs(all_values) > 1e-6) / len(all_values)),
                    'per_video_stats': video_stats
                }
        
        # Save summary
        with open(self.save_dir / 'reward_summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Summary statistics saved")
        print(f"📄 File: reward_summary_statistics.json")

    def print_quick_summary(self):
        """Print a quick summary of reward analysis findings."""
        
        print(f"\n📊 QUICK REWARD ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Videos analyzed: {len(self.all_rewards)}")
        
        # Get all reward types
        all_reward_types = set()
        for rewards in self.all_rewards.values():
            all_reward_types.update(rewards.keys())
        
        print(f"Reward types found: {len(all_reward_types)}")
        
        for reward_type in sorted(all_reward_types):
            all_values = []
            for rewards in self.all_rewards.values():
                if reward_type in rewards:
                    values = np.array(rewards[reward_type])
                    if values.size > 0:
                        all_values.extend(values.flatten())
            
            if all_values:
                all_values = np.array(all_values)
                non_zero_frac = np.sum(np.abs(all_values) > 1e-6) / len(all_values)
                
                print(f"\n  {reward_type}:")
                print(f"    Range: [{np.min(all_values):.4f}, {np.max(all_values):.4f}]")
                print(f"    Mean: {np.mean(all_values):.4f} ± {np.std(all_values):.4f}")
                print(f"    Non-zero: {non_zero_frac*100:.1f}%")
                
                # Flag potential issues
                if np.max(np.abs(all_values)) > 100:
                    print(f"    ⚠️ VALUES TOO LARGE - may cause RL instability")
                elif np.max(np.abs(all_values)) < 0.001:
                    print(f"    ⚠️ VALUES TOO SMALL - may be ignored by RL")
                elif non_zero_frac < 0.01:
                    print(f"    ⚠️ TOO SPARSE - insufficient signal")
                else:
                    print(f"    ✅ Scale looks reasonable")


def analyze_rewards_during_loading(video_data_list: List[Dict[str, Any]], 
                                 save_dir: str = "reward_analysis_loading"):
    """
    Convenience function to analyze rewards immediately after loading data.
    
    Args:
        video_data_list: List of video dictionaries from load_cholect50_data
        save_dir: Directory to save analysis results
    """
    
    print(f"\n🔍 ANALYZING REWARD VALUES AFTER DATA LOADING")
    print("=" * 50)
    
    if not video_data_list:
        print("❌ No video data provided for analysis")
        return None
    
    # Create analyzer
    analyzer = RewardAnalyzer(save_dir)
    
    # Analyze all videos
    analyzer.analyze_all_videos(video_data_list)
    
    # Print quick summary
    analyzer.print_quick_summary()
    
    return analyzer


# Integration function for cholect50.py
def add_reward_analysis_to_loading():
    """
    Example of how to integrate reward analysis into the data loading process.
    """
    
    print("💡 TO INTEGRATE REWARD ANALYSIS INTO YOUR DATA LOADING:")
    print("=" * 50)
    print("1. Add this import to cholect50.py:")
    print("   from reward_visualization_tool import analyze_rewards_during_loading")
    print()
    print("2. Add this line after data loading in load_cholect50_data():")
    print("   # Analyze rewards after loading")
    print("   if data:  # Only if data was successfully loaded")
    print("       analyze_rewards_during_loading(data, 'reward_analysis_' + split)")
    print()
    print("3. Or run analysis separately:")
    print("   train_data = load_cholect50_data(...)")
    print("   analyze_rewards_during_loading(train_data, 'train_reward_analysis')")


if __name__ == "__main__":
    print("📊 REWARD VISUALIZATION TOOL")
    print("=" * 40)
    print("Use this tool to analyze reward values during data loading")
    print()
    add_reward_analysis_to_loading()
