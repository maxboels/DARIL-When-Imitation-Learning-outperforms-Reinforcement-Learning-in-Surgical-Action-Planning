#!/usr/bin/env python3
"""
Standalone test script for Method 3: RL with Offline Video Episodes
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try multiple import paths to find the direct_video_env module
try:
    from environment.direct_video_env import DirectVideoSB3Trainer, test_direct_video_environment
    print("✅ Imported from environment.direct_video_env")
except ImportError:
    try:
        from direct_video_env import DirectVideoSB3Trainer, test_direct_video_environment
        print("✅ Imported from direct_video_env")
    except ImportError:
        print("❌ Could not import DirectVideoSB3Trainer and test_direct_video_environment")
        print("Please ensure direct_video_env.py is in the environment/ directory or current directory")
        sys.exit(1)

from datasets.cholect50 import load_cholect50_data
from utils.logger import SimpleLogger


def test_method3_standalone():
    """Test Method 3 (RL with Offline Video Episodes) in isolation."""
    
    print("🎬 TESTING METHOD 3: RL WITH OFFLINE VIDEO EPISODES")
    print("=" * 60)
    print("🎯 This tests direct RL on video sequences without world model simulation")
    print()
    
    # Load configuration
    config_path = 'config_local_debug.yaml'
    if not os.path.exists(config_path):
        config_path = 'config.yaml'
        print(f"⚠️ Using fallback config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = SimpleLogger(log_dir="logs", name="method3_test")
    logger.info("🧪 Starting Method 3 standalone test")
    
    try:
        # Load training data (small subset for testing)
        logger.info("📊 Loading training data...")
        train_data = load_cholect50_data(
            config, logger, split='train', max_videos=2
        )
        logger.info(f"✅ Loaded {len(train_data)} training videos")
        
        # Test the direct video environment
        logger.info("🎬 Testing Direct Video Environment...")
        test_success = test_direct_video_environment(train_data, config)
        
        if not test_success:
            logger.error("❌ Direct Video Environment test failed")
            return False
        
        # Create trainer
        logger.info("🔧 Creating Direct Video RL Trainer...")
        trainer = DirectVideoSB3Trainer(
            video_data=train_data,
            config=config,
            logger=logger,
            device='cpu'  # Use CPU for testing
        )
        
        # Train with reduced timesteps for testing
        test_timesteps = 2000  # Reduced for faster testing
        logger.info(f"🚀 Training RL algorithms for {test_timesteps} timesteps...")
        
        # Train all algorithms
        results = trainer.train_all_algorithms(timesteps=test_timesteps)
        
        # Print results
        print("\n🎉 METHOD 3 TEST RESULTS")
        print("=" * 40)
        
        for alg_name, result in results.items():
            if result.get('status') == 'success':
                print(f"✅ {alg_name.upper()}: Success")
                print(f"   Mean Reward: {result.get('mean_reward', 0):.3f}")
                print(f"   Std Reward: {result.get('std_reward', 0):.3f}")
                print(f"   Uses Real Frames: {result.get('uses_real_frames', False)}")
                print(f"   Uses World Model: {result.get('uses_world_model', False)}")
                
                # Episode stats
                if 'episode_stats' in result:
                    stats = result['episode_stats']
                    print(f"   Episode Stats: {stats}")
                print()
            else:
                print(f"❌ {alg_name.upper()}: Failed")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                print()
        
        # Summary
        successful_algos = [alg for alg, res in results.items() if res.get('status') == 'success']
        
        print(f"📊 SUMMARY:")
        print(f"   Successful Algorithms: {len(successful_algos)}/{len(results)}")
        print(f"   Success Rate: {len(successful_algos)/len(results)*100:.1f}%")
        print(f"   Algorithms: {successful_algos}")
        print(f"   Results saved to: {trainer.save_dir}")
        
        if successful_algos:
            print("\n✅ METHOD 3 TEST SUCCESSFUL!")
            print("🎯 Direct Video RL is working and ready for full experiment")
            return True
        else:
            print("\n❌ METHOD 3 TEST FAILED!")
            print("🔧 No algorithms trained successfully")
            return False
            
    except Exception as e:
        logger.error(f"❌ Method 3 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_method2_concept():
    """Show the conceptual difference between Method 2 and Method 3."""
    
    print("\n🔍 CONCEPTUAL COMPARISON: METHOD 2 vs METHOD 3")
    print("=" * 60)
    
    print("📊 Method 2: RL with World Model Simulation")
    print("   • Uses trained world model as environment simulator")
    print("   • Agent interacts with world_model.rl_state_prediction()")
    print("   • Can simulate beyond original video data")
    print("   • Model-based RL approach")
    print("   • Limited by world model accuracy")
    print()
    
    print("🎬 Method 3: RL with Offline Video Episodes") 
    print("   • Steps through actual video frames sequentially")
    print("   • Agent interacts with real frame embeddings")
    print("   • Limited to existing video sequences")
    print("   • Model-free RL on offline data")
    print("   • No simulation or world model dependency")
    print()
    
    print("🎯 Research Question:")
    print("   Does using a world model for simulation (Method 2)")
    print("   outperform direct interaction with real data (Method 3)?")
    print()
    
    print("📈 Expected Outcomes:")
    print("   • Method 2 might be better if world model is accurate")
    print("   • Method 3 might be better if world model has errors")
    print("   • Trade-offs in sample efficiency vs. data fidelity")


if __name__ == "__main__":
    # Show conceptual comparison first
    compare_with_method2_concept()
    
    # Run the test
    success = test_method3_standalone()
    
    if success:
        print("\n🚀 READY FOR FULL THREE-WAY COMPARISON!")
        print("   You can now run the complete experiment with all three methods:")
        print("   python run_experiment_v2.py")
    else:
        print("\n🔧 Please fix the issues before running the full experiment.")
    
    exit(0 if success else 1)