# ===================================================================
# File: diagnose_action_spaces.py
# Diagnostic script to understand action space mismatches
# ===================================================================

import numpy as np
import torch
from pathlib import Path

def diagnose_models():
    """
    Diagnose what action spaces and outputs your models actually use
    """
    
    print("🔍 DIAGNOSING ACTION SPACES")
    print("=" * 50)
    
    # 1. Test World Model Action Output
    print("\n1. Testing World Model (Imitation Learning):")
    try:
        from models import WorldModel
        import yaml
        
        with open('config_rl.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load world model
        world_model_path = config['experiment']['world_model']['best_model_path'] 
        checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
        
        model_config = config['models']['world_model']
        world_model = WorldModel(**model_config)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        world_model.eval()
        
        # Test with dummy input
        dummy_state = torch.randn(1, 1024)  # Assuming 1024 embedding dim
        
        with torch.no_grad():
            action_probs = world_model.predict_next_action(dummy_state)
            
        print(f"   ✅ World model loaded successfully")
        print(f"   📐 Input shape: {dummy_state.shape}")
        print(f"   📐 Output shape: {action_probs.shape}")
        print(f"   📊 Output range: [{action_probs.min():.3f}, {action_probs.max():.3f}]")
        print(f"   🎯 Expected: Binary predictions for 100 action classes")
        print(f"   ✅ World model produces {action_probs.shape[-1]} action predictions")
        
    except Exception as e:
        print(f"   ❌ Error testing world model: {e}")
    
    # 2. Test RL Models
    print("\n2. Testing RL Models:")
    
    # Test PPO
    try:
        from stable_baselines3 import PPO
        
        if Path('surgical_ppo_policy.zip').exists():
            ppo_model = PPO.load('surgical_ppo_policy.zip')
            
            print(f"   PPO Model Info:")
            print(f"   📐 Observation space: {ppo_model.observation_space}")
            print(f"   📐 Action space: {ppo_model.action_space}")
            
            # Test prediction
            dummy_obs = np.random.randn(1024)  # Match world model embedding dim
            action, _ = ppo_model.predict(dummy_obs, deterministic=True)
            
            print(f"   📊 Action output shape: {action.shape}")
            print(f"   📊 Action output type: {type(action)}")
            print(f"   📊 Action output: {action}")
            print(f"   🎯 Need: 100-dimensional binary vector")
            
        else:
            print("   ⚠️  PPO model file not found")
            
    except Exception as e:
        print(f"   ❌ Error testing PPO: {e}")
    
    # Test SAC
    try:
        from stable_baselines3 import SAC
        
        if Path('surgical_sac_policy.zip').exists():
            sac_model = SAC.load('surgical_sac_policy.zip')
            
            print(f"\n   SAC Model Info:")
            print(f"   📐 Observation space: {sac_model.observation_space}")
            print(f"   📐 Action space: {sac_model.action_space}")
            
            # Test prediction
            dummy_obs = np.random.randn(1024)
            action, _ = sac_model.predict(dummy_obs, deterministic=True)
            
            print(f"   📊 Action output shape: {action.shape}")
            print(f"   📊 Action output type: {type(action)}")
            print(f"   📊 Action range: [{action.min():.3f}, {action.max():.3f}]")
            print(f"   🎯 Need: 100-dimensional binary vector")
            
        else:
            print("   ⚠️  SAC model file not found")
            
    except Exception as e:
        print(f"   ❌ Error testing SAC: {e}")
    
    # 3. Test Ground Truth Data
    print("\n3. Testing Ground Truth Data:")
    try:
        from datasets.cholect50 import load_cholect50_data
        import yaml
        import logging
        
        with open('config_rl.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger = logging.getLogger(__name__)
        test_data = load_cholect50_data(config, logger, split='test', max_videos=1)
        
        if test_data:
            video = test_data[0]
            actions = video['actions_binaries']
            embeddings = video['frame_embeddings']
            
            print(f"   ✅ Test data loaded successfully")
            print(f"   📐 Frame embeddings shape: {embeddings.shape}")
            print(f"   📐 Actions shape: {actions.shape}")
            print(f"   📊 Actions dtype: {actions.dtype}")
            print(f"   📊 Actions range: [{actions.min()}, {actions.max()}]")
            print(f"   🎯 This is what models need to predict")
            
    except Exception as e:
        print(f"   ❌ Error testing ground truth: {e}")
    
    # 4. Recommendations
    print("\n4. RECOMMENDATIONS:")
    print("   🔧 Action Space Alignment:")
    print("      - World Model: Should output 100-dim binary vector")
    print("      - PPO: Should be trained with MultiBinary(100) action space")
    print("      - SAC: Should be trained with Box(0, 1, (100,)) action space")
    print("   🔧 Inference Alignment:")
    print("      - All models should predict 100-dimensional action vectors")
    print("      - Apply thresholding: action > 0.5 for binary decisions")
    print("      - Ensure consistent shape handling in evaluation")

def check_rl_training_config():
    """Check if RL models were trained with correct action spaces"""
    
    print("\n🔍 CHECKING RL TRAINING CONFIGURATION")
    print("=" * 50)
    
    try:
        import yaml
        with open('config_rl.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("Training Configuration:")
        if 'rl_experiments' in config:
            rl_config = config['rl_experiments']
            print(f"   Algorithms: {rl_config.get('algorithms', 'Not specified')}")
            print(f"   Timesteps: {rl_config.get('timesteps', 'Not specified')}")
            print(f"   Horizon: {rl_config.get('horizon', 'Not specified')}")
        
        if 'models' in config and 'world_model' in config['models']:
            wm_config = config['models']['world_model']
            print(f"   Action classes: {wm_config.get('num_action_classes', 'Not specified')}")
            print(f"   Action learning: {wm_config.get('action_learning', 'Not specified')}")
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")

if __name__ == "__main__":
    diagnose_models()
    check_rl_training_config()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")
    print("Use this information to fix action space mismatches")
    print("Then run the corrected evaluation framework")
