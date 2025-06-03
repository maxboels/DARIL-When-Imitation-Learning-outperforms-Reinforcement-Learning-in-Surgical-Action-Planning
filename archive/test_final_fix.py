#!/usr/bin/env python3
"""
Quick test to verify the final fix works
"""

import os
import sys
import yaml
import numpy as np

# Add your project paths
sys.path.append('.')

def test_final_fix():
    """Test the final fixed trainer quickly."""
    
    print("🧪 TESTING FINAL FIX")
    print("=" * 40)
    
    try:
        # Load config
        config_path = 'config_local_debug.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load data (minimal)
        from datasets.cholect50 import load_cholect50_data
        from utils.logger import SimpleLogger
        
        logger = SimpleLogger(log_dir="test_logs", name="test_final_fix")
        train_data = load_cholect50_data(config, logger, split='train', max_videos=2)
        
        print(f"✅ Loaded {len(train_data)} videos")
        
        # Test the final fixed environment
        from final_fixed_trainer import test_final_fixed_environment
        
        success = test_final_fixed_environment(train_data, config)
        
        if success:
            print("\n🚀 TESTING RL TRAINING")
            print("-" * 30)
            
            # Test actual RL training
            from final_fixed_trainer import SB3Trainer
            from models.dual_world_model import DualWorldModel
            
            # Load or create world model
            il_model_path = config.get('experiment', {}).get('il_experiments', {}).get('il_model_path')
            if il_model_path and os.path.exists(il_model_path):
                world_model = DualWorldModel.load_model(il_model_path, device='cpu')
            else:
                model_config = config['models']['dual_world_model']
                world_model = DualWorldModel(**model_config)
            
            # Create trainer
            trainer = SB3Trainer(world_model, config, logger, device='cpu')
            
            # Test very short training
            print("🔧 Testing PPO with 1000 timesteps...")
            result = trainer.train_ppo_final(train_data, timesteps=1000)
            
            if result['status'] == 'success':
                print("✅ PPO training successful!")
                print(f"📊 Mean reward: {result['mean_reward']:.3f}")
                return True
            else:
                print(f"❌ PPO training failed: {result.get('error', 'Unknown')}")
                return False
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_final_fix()
    
    if success:
        print("\n🎉 FINAL FIX SUCCESSFUL!")
        print("=" * 40)
        print("✅ Action space issue resolved")
        print("✅ Environment working correctly")
        print("✅ RL training functioning")
        print("\n📋 Next Steps:")
        print("1. Use SB3Trainer in your main experiment")
        print("2. Increase timesteps for longer training")
        print("3. Monitor tensorboard for training curves")
        print("\n🔧 Usage:")
        print("from final_fixed_trainer import SB3Trainer")
        print("trainer = SB3Trainer(world_model, config, logger)")
        print("results = trainer.train_ppo_final(train_data, timesteps=10000)")
    else:
        print("\n❌ FINAL FIX FAILED")
        print("Check the error messages above for debugging")
