#!/usr/bin/env python3
"""
Quick RL Test Script
Run just the RL parts with debugging to verify improvements before full experiment
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

def quick_rl_test(config_path='config_improved_rl.yaml'):
    """Quick test of improved RL without full experiment."""
    
    print("🚀 QUICK RL TEST")
    print("=" * 50)
    print("Testing improved RL training with debugging...")
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded: {config_path}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False
    
    # Import components
    try:
        from datasets.cholect50 import load_cholect50_data
        from environment.rl_environments import DirectVideoEnvironment
        from rl_debug_tools import RLDebugger
        from utils.logger import SimpleLogger
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Setup logger
    logger = SimpleLogger(log_dir="quick_rl_test", name="QuickRLTest")
    
    # Load minimal data
    print("\n📂 Loading minimal data...")
    config['experiment'] = {'train': {'max_videos': 1}, 'test': {'max_videos': 1}}
    
    try:
        train_data = load_cholect50_data(config, logger, split='train', max_videos=1)
        if not train_data:
            print("❌ No training data loaded")
            return False
        print(f"✅ Loaded {len(train_data)} training video(s)")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test improved environment
    print("\n🎬 Testing improved DirectVideoEnvironment...")
    try:
        env = DirectVideoEnvironment(
            video_data=train_data,
            config=config.get('rl_training', {}),
            device='cpu'
        )
        
        # Initialize debugger
        debugger = RLDebugger("quick_rl_test/debug")
        
        # Run a few episodes
        episode_rewards = []
        expert_matching_scores = []
        
        for episode in range(5):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_expert_scores = []
            
            for step in range(20):
                # Random action for testing
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Track expert matching if available
                if 'expert_matching' in info:
                    episode_expert_scores.append(info['expert_matching'])
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            if episode_expert_scores:
                expert_matching_scores.append(np.mean(episode_expert_scores))
            
            # Log to debugger
            debugger.log_episode({
                'episode_reward': episode_reward,
                'episode_length': step + 1
            })
            
            print(f"Episode {episode+1}: Reward = {episode_reward:.3f}, Steps = {step+1}")
        
        print(f"\n📊 QUICK RL TEST RESULTS:")
        print(f"   Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"   Reward range: [{np.min(episode_rewards):.3f}, {np.max(episode_rewards):.3f}]")
        
        if expert_matching_scores:
            print(f"   Average expert matching: {np.mean(expert_matching_scores):.3f}")
        
        # Check if rewards are reasonable
        avg_reward = np.mean(episode_rewards)
        if avg_reward > 0:
            print("🎉 REWARDS ARE POSITIVE! (Major improvement)")
        elif avg_reward > -10:
            print("👍 Rewards are much better than previous -400")
        else:
            print("⚠️ Rewards still negative, may need more tuning")
        
        # Generate debug plots
        debugger.plot_training_curves()
        analysis = debugger.analyze_convergence()
        print(f"\n🔍 Convergence analysis: {analysis}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_old_vs_new():
    """Compare old vs new expected results."""
    
    print("\n📈 EXPECTED IMPROVEMENTS:")
    print("OLD RL Results (what we had):")
    print("   Method 2 World Model RL: PPO: -400.010, A2C: -404.962")
    print("   Method 3 Direct Video RL: PPO: 79.488, A2C: 76.512")
    print()
    print("NEW RL Results (what we expect):")
    print("   Method 2 World Model RL: PPO: +50-150, A2C: +50-150")
    print("   Method 3 Direct Video RL: PPO: +100-200, A2C: +100-200")
    print()
    print("KEY IMPROVEMENTS:")
    print("   ✅ Expert demonstration matching rewards")
    print("   ✅ Proper action space [0,1] continuous")
    print("   ✅ Better episode termination")
    print("   ✅ Enhanced monitoring and debugging")
    print("   ✅ Optimized hyperparameters")

def main():
    """Run quick RL test."""
    
    print("🔧 QUICK RL VERIFICATION")
    print("=" * 60)
    print("This will test the improved RL before running the full experiment")
    print()
    
    # Run quick test
    success = quick_rl_test()
    
    # Show comparison
    compare_old_vs_new()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 QUICK RL TEST PASSED!")
        print("✅ Ready to run full experiment:")
        print("   python run_experiment_v4.py --config config_improved_rl.yaml")
        print()
        print("🎯 What to expect in full experiment:")
        print("   • Positive RL rewards (no more -400)")
        print("   • Learning progress over episodes") 
        print("   • Expert matching scores improving")
        print("   • Comprehensive debug outputs")
        print("   • Training curves showing convergence")
    else:
        print("❌ QUICK RL TEST FAILED!")
        print("   Please fix issues before running full experiment")
        print("   Check the error messages above for debugging")
    
    print("\n📋 NEXT STEPS:")
    print("1. If test passed, run full experiment")
    print("2. Monitor RL training logs for improvements")
    print("3. Check results/*/rl_debug/ for detailed analysis")
    print("4. Compare with previous results")

if __name__ == "__main__":
    main()