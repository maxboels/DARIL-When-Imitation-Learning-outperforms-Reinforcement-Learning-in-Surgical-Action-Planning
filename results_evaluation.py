import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Analyze and visualize results
def analyze_results(results, action_weights):
    """Analyze and visualize results of the experiment."""
    print("Analyzing results...")
    
    # Calculate statistics
    original_action_weights = []
    recommended_action_weights = []
    reward_diffs = []
    
    for video in results:
        for frame in video['frame_results']:
            original_action_weights.append(frame['original_action_weight'])
            recommended_action_weights.append(frame['recommended_action_weight'])
            reward_diffs.append(frame['reward_difference'])
    
    # Calculate average improvement
    avg_original_weight = np.mean(original_action_weights)
    avg_recommended_weight = np.mean(recommended_action_weights)
    avg_improvement = avg_recommended_weight - avg_original_weight
    percent_improvement = (avg_improvement / avg_original_weight) * 100
    
    print(f"Results Summary:")
    print(f"- Average original action weight: {avg_original_weight:.2f}")
    print(f"- Average recommended action weight: {avg_recommended_weight:.2f}")
    print(f"- Average improvement: {avg_improvement:.2f} ({percent_improvement:.2f}%)")
    print(f"- Average reward difference: {np.mean(reward_diffs):.2f}")
    
    # Visualize results
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Histogram of original vs. recommended action weights
    plt.subplot(2, 2, 1)
    plt.hist(original_action_weights, bins=20, alpha=0.5, label='Original Actions')
    plt.hist(recommended_action_weights, bins=20, alpha=0.5, label='Recommended Actions')
    plt.title('Action Weight Distribution')
    plt.xlabel('Action Weight')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Scatterplot of original vs. recommended weights
    plt.subplot(2, 2, 2)
    plt.scatter(original_action_weights, recommended_action_weights, alpha=0.3)
    plt.plot([0, 10], [0, 10], 'r--')  # Diagonal line
    plt.title('Original vs. Recommended Action Weights')
    plt.xlabel('Original Action Weight')
    plt.ylabel('Recommended Action Weight')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    # Plot 3: Top 10 recommended actions
    recommended_actions = [frame['recommended_action'] for video in results for frame in video['frame_results']]
    action_counts = {}
    for action in recommended_actions:
        if action not in action_counts:
            action_counts[action] = 0
        action_counts[action] += 1
    
    top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    plt.subplot(2, 2, 3)
    plt.bar([f"Action {a[0]}" for a in top_actions], [a[1] for a in top_actions])
    plt.title('Top 10 Recommended Actions')
    plt.xlabel('Action ID')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # Plot 4: Average reward differences by video
    video_reward_diffs = []
    for video in results:
        avg_video_diff = np.mean([frame['reward_difference'] for frame in video['frame_results']])
        video_reward_diffs.append((video['video_id'], avg_video_diff))
    
    video_reward_diffs.sort(key=lambda x: x[0])
    
    plt.subplot(2, 2, 4)
    plt.bar([f"VID{v[0]}" for v in video_reward_diffs], [v[1] for v in video_reward_diffs])
    plt.title('Average Reward Difference by Video')
    plt.xlabel('Video ID')
    plt.ylabel('Avg. Reward Difference')
    plt.xticks(rotation=45)
    save_dir
    plt.tight_layout()
    plt.savefig('results_analysis.png')
    plt.show()
    
    return {
        'avg_original_weight': avg_original_weight,
        'avg_recommended_weight': avg_recommended_weight,
        'percent_improvement': percent_improvement,
        'avg_reward_diff': np.mean(reward_diffs)
    }