#!/usr/bin/env python3
"""
Debug Experiment Runner - Use this to test and fix the RL training
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add your project paths
sys.path.append('.')

def debug_rl_training_pipeline():
    """
    Complete debug pipeline for RL training issues.
    """
    
    print("🔧 DEBUGGING RL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load configuration
    config_path = 'config_local_debug.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded from {config_path}")
    
    # Step 1: Test data loading
    print("\n📊 Step 1: Testing Data Loading")
    print("-" * 30)
    
    try:
        from datasets.cholect50 import load_cholect50_data
        from utils.logger import SimpleLogger
        
        # Create logger
        logger = SimpleLogger(log_dir="debug_logs", name="debug_rl")
        
        # Load minimal data for testing
        config['experiment']['train']['max_videos'] = 2  # Keep small for debugging
        train_data = load_cholect50_data(config, logger, split='train', max_videos=2)
        
        print(f"✅ Loaded {len(train_data)} training videos")
        
        # Validate data structure
        for i, video in enumerate(train_data):
            print(f"   Video {i}: {video['video_id']} - {len(video['frame_embeddings'])} frames")
            
            # Check required fields
            required_fields = ['frame_embeddings', 'actions_binaries']
            for field in required_fields:
                if field not in video:
                    print(f"   ⚠️ Missing field: {field}")
                else:
                    print(f"   ✅ {field}: shape {np.array(video[field]).shape}")
    
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Test world model loading
    print("\n🤖 Step 2: Testing World Model")
    print("-" * 30)
    
    try:
        from models.dual_world_model import DualWorldModel
        
        # Try to load existing model or create new one
        il_model_path = config.get('experiment', {}).get('il_experiments', {}).get('il_model_path')
        
        if il_model_path and os.path.exists(il_model_path):
            print(f"🔧 Loading existing model from: {il_model_path}")
            world_model = DualWorldModel.load_model(il_model_path, device='cpu')
        else:
            print("🔧 Creating new world model for testing")
            model_config = config['models']['dual_world_model']
            world_model = DualWorldModel(**model_config)
        
        print(f"✅ World model loaded: {type(world_model)}")
        print(f"   Parameters: {sum(p.numel() for p in world_model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ World model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test environment creation
    print("\n🌍 Step 3: Testing Environment")
    print("-" * 30)
    
    try:
        # Import the fixed environment from our artifact
        from fixed_sb3_trainer import FixedSurgicalActionEnv, test_environment_standalone
        
        # Test environment standalone
        success = test_environment_standalone(train_data, config)
        
        if not success:
            print("❌ Environment test failed")
            return False
        
        print("✅ Environment test passed")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test RL training
    print("\n🚀 Step 4: Testing RL Training")
    print("-" * 30)
    
    try:
        from fixed_sb3_trainer import DebuggedSB3Trainer
        
        # Create trainer
        trainer = DebuggedSB3Trainer(world_model, config, logger, device='cpu')
        
        # Test with very short training for debugging
        debug_timesteps = 1000  # Very short for debugging
        
        print(f"🔧 Testing PPO training with {debug_timesteps} timesteps...")
        ppo_result = trainer.train_ppo_debug(train_data, timesteps=debug_timesteps)
        
        if ppo_result['status'] == 'success':
            print("✅ PPO debug training successful!")
            print(f"   Mean reward: {ppo_result.get('mean_reward', 'N/A')}")
        else:
            print(f"❌ PPO debug training failed: {ppo_result.get('error', 'Unknown error')}")
            return False
        
        print(f"\n🔧 Testing DQN training with {debug_timesteps} timesteps...")
        dqn_result = trainer.train_dqn_debug(train_data, timesteps=debug_timesteps)
        
        if dqn_result['status'] == 'success':
            print("✅ DQN debug training successful!")
            print(f"   Mean reward: {dqn_result.get('mean_reward', 'N/A')}")
        else:
            print(f"❌ DQN debug training failed: {dqn_result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ RL training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Summary and recommendations
    print("\n📋 Step 5: Summary and Recommendations")
    print("-" * 30)
    
    print("✅ All debug tests passed!")
    print("\n🔧 Key Fixes Applied:")
    print("   • Fixed environment termination logic")
    print("   • Added proper error handling and debugging")
    print("   • Fixed reward calculation (removed duplicate calculation)")
    print("   • Added comprehensive logging and monitoring")
    print("   • Reduced training parameters for faster debugging")
    print("   • Added episode statistics tracking")
    
    print("\n📈 Next Steps:")
    print("   1. Run with longer timesteps (10,000 - 50,000)")
    print("   2. Monitor tensorboard logs for training curves")
    print("   3. Adjust hyperparameters based on performance")
    print("   4. Scale up to more videos once stable")
    
    print("\n🎯 To run full training:")
    print("   1. Update config timesteps: 'timesteps: 50000'")
    print("   2. Increase max_videos: 'max_videos: 5'")
    print("   3. Use the DebuggedSB3Trainer in your main script")
    
    return True


def create_fixed_main_experiment():
    """Create a fixed version of the main experiment runner."""
    
    fixed_code = '''#!/usr/bin/env python3
"""
FIXED Main Experiment Runner - Uses debugged RL training
"""

import numpy as np
import torch
import os
import yaml
from pathlib import Path

# Import fixed components
from fixed_sb3_trainer import DebuggedSB3Trainer
from datasets.cholect50 import load_cholect50_data, NextFramePredictionDataset, create_video_dataloaders
from utils.logger import SimpleLogger
from models.dual_world_model import DualWorldModel
from trainer.dual_trainer import train_dual_world_model
from torch.utils.data import DataLoader

class FixedComparisonExperiment:
    """Fixed IL vs RL comparison with proper RL training."""
    
    def __init__(self, config_path: str = 'config_local_debug.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = SimpleLogger(log_dir="logs", name="il_vs_rl_comparison_fixed")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.results = {
            'il_results': None,
            'rl_results': {},
            'comparison_results': None,
            'model_paths': {},
            'config': self.config
        }
        
        self.results_dir = Path(self.logger.log_dir) / 'comparison_results'
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("🚀 Starting FIXED IL vs RL Comparison Experiment")
    
    def _load_data(self):
        """Load training and test data."""
        train_videos = self.config.get('experiment', {}).get('train', {}).get('max_videos', 2)
        train_data = load_cholect50_data(
            self.config, self.logger, split='train', max_videos=train_videos
        )
        
        test_videos = self.config.get('experiment', {}).get('test', {}).get('max_videos', 1)
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=test_videos
        )
        
        self.logger.info(f"Loaded {len(train_data)} training videos and {len(test_data)} test videos")
        return train_data, test_data
    
    def _train_imitation_learning(self, train_data, test_data):
        """Train IL model if needed."""
        il_model_path = self.config.get('experiment', {}).get('il_experiments', {}).get('il_model_path')
        
        if il_model_path and os.path.exists(il_model_path):
            self.logger.info(f"✅ Using existing IL model: {il_model_path}")
            return il_model_path
        
        self.logger.info("🎓 Training new IL model...")
        
        # Create datasets
        train_dataset = NextFramePredictionDataset(self.config['data'], train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
        
        test_video_loaders = create_video_dataloaders(
            self.config, test_data, batch_size=16, shuffle=False
        )
        
        # Initialize model
        model_config = self.config['models']['dual_world_model']
        model = DualWorldModel(**model_config).to(self.device)
        
        # Train
        self.config['training_mode'] = 'supervised'
        il_model_path = train_dual_world_model(
            self.config, self.logger, model, train_loader, test_video_loaders, self.device
        )
        
        return il_model_path
    
    def _train_rl_models_fixed(self, train_data, world_model_path):
        """Train RL models using the fixed trainer."""
        
        # Load world model
        if world_model_path and os.path.exists(world_model_path):
            world_model = DualWorldModel.load_model(world_model_path, self.device)
        else:
            # Create new model if needed
            model_config = self.config['models']['dual_world_model']
            world_model = DualWorldModel(**model_config).to(self.device)
        
        # Create fixed trainer
        trainer = DebuggedSB3Trainer(world_model, self.config, self.logger, self.device)
        
        rl_results = {}
        
        # Get timesteps from config
        timesteps = self.config.get('experiment', {}).get('rl_experiments', {}).get('timesteps', 10000)
        
        # Train PPO
        self.logger.info("🤖 Training PPO...")
        ppo_result = trainer.train_ppo_debug(train_data, timesteps=timesteps)
        rl_results['ppo'] = ppo_result
        
        # Train DQN
        self.logger.info("🤖 Training DQN...")
        dqn_result = trainer.train_dqn_debug(train_data, timesteps=timesteps)
        rl_results['dqn'] = dqn_result
        
        return rl_results
    
    def run_fixed_comparison(self):
        """Run the fixed comparison experiment."""
        
        try:
            # Load data
            self.logger.info("📊 Loading dataset...")
            train_data, test_data = self._load_data()
            
            # Train IL
            if self.config['experiment']['il_experiments']['enabled']:
                il_model_path = self._train_imitation_learning(train_data, test_data)
                self.results['model_paths']['imitation_learning'] = il_model_path
            
            # Train RL with fixed trainer
            if self.config.get('experiment', {}).get('rl_experiments', {}).get('enabled', False):
                self.logger.info("🤖 Training RL Models (FIXED)...")
                rl_results = self._train_rl_models_fixed(train_data, self.results['model_paths'].get('imitation_learning'))
                self.results['rl_results'] = rl_results
            
            # Log results
            self.logger.info("📊 EXPERIMENT RESULTS:")
            for alg, result in self.results['rl_results'].items():
                if result['status'] == 'success':
                    self.logger.info(f"   {alg.upper()}: Mean reward = {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
                    self.logger.info(f"      Episode stats: {result.get('episode_stats', {})}")
                else:
                    self.logger.info(f"   {alg.upper()}: FAILED - {result.get('error', 'Unknown error')}")
            
            self.logger.info("✅ Fixed IL vs RL comparison completed!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'partial_results': self.results}


def main():
    """Run the fixed experiment."""
    print("🔧 FIXED IL vs RL COMPARISON")
    print("=" * 50)
    
    experiment = FixedComparisonExperiment()
    results = experiment.run_fixed_comparison()
    
    if 'error' not in results:
        print("🎉 Experiment completed successfully!")
    else:
        print(f"❌ Experiment failed: {results['error']}")


if __name__ == "__main__":
    main()
'''
    
    # Save the fixed code
    with open('fixed_main_experiment.py', 'w') as f:
        f.write(fixed_code)
    
    print("✅ Created fixed_main_experiment.py")
    print("   Run this file to test the complete fixed pipeline")


if __name__ == "__main__":
    # Run debug pipeline
    success = debug_rl_training_pipeline()
    
    if success:
        print("\n📁 Creating fixed experiment file...")
        create_fixed_main_experiment()
        
        print("\n🎯 DEBUGGING COMPLETE!")
        print("=" * 50)
        print("✅ All tests passed")
        print("📁 Files created:")
        print("   • fixed_sb3_trainer.py (use this for RL training)")
        print("   • fixed_main_experiment.py (use this to run experiments)")
        print("\n🚀 Next steps:")
        print("   1. python fixed_main_experiment.py")
        print("   2. Monitor the training progress and logs")
        print("   3. Check tensorboard logs for training curves")
    else:
        print("\n❌ Debugging failed - check the errors above")
