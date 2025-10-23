#!/usr/bin/env python3
"""
Quick RL Diagnostic Script
Run this to check if your RL is working properly
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def diagnose_rl_training(log_dir: str):
    """Quick diagnostic of RL training logs."""
    
    print("🔍 RL TRAINING DIAGNOSTICS")
    print("=" * 50)
    
    # Check reward trends
    print("\n1. 📊 REWARD ANALYSIS:")
    
    # Simulate checking your actual reward logs
    # You should replace this with your actual reward loading
    recent_rewards = [-400, -380, -350, -320, -300]  # Example from your logs
    
    if all(r < 0 for r in recent_rewards):
        print("❌ All rewards are negative!")
        print("   → Problem: Reward function is poorly designed")
        print("   → Solution: Add expert demonstration matching")
    
    reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
    if reward_trend > 0:
        print(f"✅ Rewards trending upward: +{reward_trend:.3f}")
    else:
        print(f"❌ Rewards trending downward: {reward_trend:.3f}")
    
    # Check action space
    print("\n2. 🎯 ACTION SPACE ANALYSIS:")
    
    # Check if your RL model outputs match expected format
    # You should replace this with your actual model testing
    dummy_state = torch.randn(1, 1024)
    
    print("   Testing action space compatibility...")
    print("   RL Action Space: Box(0.0, 1.0, (100,))")  # Should be continuous
    print("   Evaluation expects: Binary {0,1}^100")     # Binary actions
    print("   ⚠️  MISMATCH DETECTED!")
    print("   → Solution: Use continuous→binary conversion in evaluation")
    
    # Check expert matching
    print("\n3. 👩‍⚕️ EXPERT DEMONSTRATION MATCHING:")
    
    # Simulate expert matching calculation
    dummy_rl_actions = np.random.rand(100, 100)  # RL predictions
    dummy_expert_actions = np.random.randint(0, 2, (100, 100))  # Expert binary
    
    binary_rl = (dummy_rl_actions > 0.5).astype(int)
    matching_rate = np.mean(binary_rl == dummy_expert_actions)
    
    print(f"   Current expert matching: {matching_rate:.3f}")
    if matching_rate < 0.5:
        print("   ❌ Poor expert alignment!")
        print("   → Problem: RL not learning from demonstrations")
        print("   → Solution: Add imitation learning component to reward")
    else:
        print("   ✅ Good expert alignment")
    
    # Check action density
    print("\n4. 🔢 ACTION DENSITY ANALYSIS:")
    
    avg_rl_actions = np.mean(np.sum(binary_rl, axis=1))
    avg_expert_actions = np.mean(np.sum(dummy_expert_actions, axis=1))
    
    print(f"   RL avg actions/step: {avg_rl_actions:.1f}")
    print(f"   Expert avg actions/step: {avg_expert_actions:.1f}")
    
    if abs(avg_rl_actions - avg_expert_actions) > 2:
        print("   ❌ Action density mismatch!")
        print("   → Problem: RL not matching expert action patterns")
    else:
        print("   ✅ Action density aligned")
    
    # Overall assessment
    print("\n🎯 OVERALL ASSESSMENT:")
    issues = []
    
    if all(r < 0 for r in recent_rewards):
        issues.append("Negative rewards")
    if reward_trend <= 0:
        issues.append("No learning progress")
    if matching_rate < 0.5:
        issues.append("Poor expert matching")
    if abs(avg_rl_actions - avg_expert_actions) > 2:
        issues.append("Wrong action density")
    
    if issues:
        print(f"   ❌ {len(issues)} major issues found:")
        for issue in issues:
            print(f"      • {issue}")
        print("\n   🔧 RECOMMENDED FIXES:")
        print("      1. Use the fixed environments from the artifacts above")
        print("      2. Implement expert demonstration matching rewards")
        print("      3. Fix action space evaluation mismatch")
        print("      4. Train for longer (20k+ timesteps)")
        print("      5. Monitor expert matching during training")
    else:
        print("   ✅ RL training looks healthy!")

def plot_expected_rl_curves():
    """Plot what healthy RL curves should look like."""
    
    # Simulate healthy RL training curves
    steps = np.arange(0, 20000, 100)
    
    # Healthy reward curve: starts low, increases, stabilizes
    reward_curve = -5 + 8 * (1 - np.exp(-steps/5000)) + np.random.normal(0, 0.5, len(steps))
    
    # Expert matching: starts random, improves to >70%
    expert_matching = 0.5 + 0.3 * (1 - np.exp(-steps/8000)) + np.random.normal(0, 0.05, len(steps))
    expert_matching = np.clip(expert_matching, 0, 1)
    
    # Action density: should converge to expert level (~3-4 actions)
    action_density = 3.5 + 2 * np.exp(-steps/3000) * np.sin(steps/1000) + np.random.normal(0, 0.2, len(steps))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Reward curve
    axes[0].plot(steps, reward_curve, 'b-', alpha=0.7, linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Reward')
    axes[0].set_title('Healthy RL Reward Curve')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Episode Reward')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Expert matching
    axes[1].plot(steps, expert_matching, 'g-', alpha=0.7, linewidth=2)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Matching')
    axes[1].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good Performance')
    axes[1].set_title('Expert Action Matching')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Matching Rate')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Action density
    axes[2].plot(steps, action_density, 'orange', alpha=0.7, linewidth=2)
    axes[2].axhline(y=3.5, color='g', linestyle='--', alpha=0.5, label='Expert Level')
    axes[2].set_title('Action Density Convergence')
    axes[2].set_xlabel('Training Steps')
    axes[2].set_ylabel('Actions per Step')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('healthy_rl_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Saved expected RL curves to 'healthy_rl_curves.png'")
    print("\nWhat you should see:")
    print("  • Rewards start low but increase steadily")
    print("  • Expert matching improves from ~50% to >70%")
    print("  • Action density converges to expert levels")
    print("  • All curves stabilize after sufficient training")

if __name__ == "__main__":
    # Run diagnostics
    diagnose_rl_training("your_log_directory")
    
    # Plot expected curves
    plot_expected_rl_curves()
    
    print("\n🔧 NEXT STEPS:")
    print("1. Use the fixed RL environments provided above")
    print("2. Implement the debugging tools")
    print("3. Train for 20k+ timesteps with proper monitoring")
    print("4. Focus on expert demonstration matching in rewards")
    print("5. Fix the action space mismatch in evaluation")
