# ===================================================================
# File: evaluate_rl_results.py  
# Script to analyze and visualize RL experiment results
# ===================================================================

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List

def load_results(results_path: str = 'rl_comparison_results.json') -> Dict:
    """Load experiment results"""
    with open(results_path, 'r') as f:
        return json.load(f)

def create_comparison_plot(results: Dict, save_path: str = 'rl_comparison.png'):
    """Create comparison plot of different approaches"""
    
    # Extract data for plotting
    methods = []
    rewards = []
    errors = []
    
    # Baseline IL
    if 'baseline_imitation' in results:
        il_metrics = results['baseline_imitation']['environment_metrics']
        methods.append('Imitation Learning')
        rewards.append(il_metrics['avg_episode_reward'])
        errors.append(il_metrics.get('std_episode_reward', 0))
    
    # RL algorithms
    if 'rl_algorithms' in results:
        for alg_name, alg_results in results['rl_algorithms'].items():
            if 'evaluation' in alg_results:
                methods.append(alg_name.upper())
                rewards.append(alg_results['evaluation']['avg_reward'])
                errors.append(alg_results['evaluation']['std_reward'])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, rewards, yerr=errors, capsize=5, alpha=0.7)
    
    # Color bars
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    plt.title('Surgical World Model: RL vs Imitation Learning Comparison')
    plt.ylabel('Average Episode Reward')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Add value labels on bars
    for i, (method, reward, error) in enumerate(zip(methods, rewards, errors)):
        plt.text(i, reward + error + 0.1, f'{reward:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to {save_path}")

def create_detailed_analysis(results: Dict):
    """Create detailed analysis report"""
    
    print("\n" + "="*60)
    print("DETAILED RL EXPERIMENT ANALYSIS")
    print("="*60)
    
    # Baseline performance
    if 'baseline_imitation' in results:
        il_metrics = results['baseline_imitation']['environment_metrics']
        print(f"\nBaseline Imitation Learning:")
        print(f"  Average Reward: {il_metrics['avg_episode_reward']:.4f}")
        print(f"  Std Reward: {il_metrics['std_episode_reward']:.4f}")
        print(f"  Episodes: {il_metrics['num_episodes']}")
        
        baseline_reward = il_metrics['avg_episode_reward']
    else:
        baseline_reward = 0
    
    # RL algorithm performance
    if 'rl_algorithms' in results:
        print(f"\nRL Algorithm Results:")
        
        best_rl_method = None
        best_rl_reward = float('-inf')
        
        for alg_name, alg_results in results['rl_algorithms'].items():
            print(f"\n{alg_name.upper()}:")
            
            if 'error' in alg_results:
                print(f"  Status: ERROR - {alg_results['error']}")
                continue
                
            if 'evaluation' not in alg_results:
                print(f"  Status: No evaluation data available")
                continue
            
            eval_data = alg_results['evaluation']
            avg_reward = eval_data['avg_reward']
            
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Std Reward: {eval_data['std_reward']:.4f}")
            print(f"  Min/Max Reward: {eval_data['min_reward']:.4f}/{eval_data['max_reward']:.4f}")
            print(f"  Average Length: {eval_data['avg_length']:.1f}")
            
            # Calculate improvement over baseline
            if baseline_reward != 0:
                improvement = ((avg_reward - baseline_reward) / abs(baseline_reward)) * 100
                print(f"  Improvement over IL: {improvement:+.2f}%")
            
            # Track best RL method
            if avg_reward > best_rl_reward:
                best_rl_reward = avg_reward
                best_rl_method = alg_name
        
        # Summary
        print(f"\n" + "-"*40)
        print("SUMMARY:")
        print(f"Best RL Method: {best_rl_method.upper() if best_rl_method else 'None'}")
        
        if best_rl_method and baseline_reward != 0:
            overall_improvement = ((best_rl_reward - baseline_reward) / abs(baseline_reward)) * 100
            if overall_improvement > 5:
                print(f"Result: RL shows significant improvement ({overall_improvement:+.2f}%)")
            elif overall_improvement > 0:
                print(f"Result: RL shows modest improvement ({overall_improvement:+.2f}%)")
            else:
                print(f"Result: Imitation Learning is competitive ({overall_improvement:+.2f}%)")
        
    print(f"\n" + "="*60)

def main():
    """Main analysis function"""
    
    try:
        # Load results
        results = load_results()
        
        # Create visualizations
        create_comparison_plot(results)
        
        # Detailed analysis
        create_detailed_analysis(results)
        
        # Save summary report
        summary = {
            'experiment_date': str(pd.Timestamp.now()),
            'methods_tested': list(results.keys()),
            'key_findings': "RL vs IL comparison completed"
        }
        
        with open('experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nAnalysis complete! Check experiment_summary.json for key findings.")
        
    except FileNotFoundError:
        print("Error: rl_comparison_results.json not found.")
        print("Run the RL experiments first using: python run_integrated_rl_experiment.py")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
