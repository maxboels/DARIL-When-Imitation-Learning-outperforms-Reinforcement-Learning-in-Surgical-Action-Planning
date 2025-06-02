#!/usr/bin/env python3
"""
Usage Example: Dual World Model for Surgical Video Analysis

This script demonstrates how to:
1. Train the model in supervised mode for autoregressive action prediction
2. Fine-tune the model in RL mode for state and reward prediction
3. Evaluate both modes and compare performance
4. Use the model for inference in both modes

Run this script to see the dual world model in action!
"""

import sys
import os
import torch
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import your modules
from models.dual_world_model import DualWorldModel
from trainer.dual_trainer import DualTrainer
from evaluation.dual_evaluator import DualModelEvaluator
from rl.improved_environment import SurgicalWorldModelEnv
from datasets.cholect50 import load_cholect50_data, NextFramePredictionDataset
from utils.model_analyzer import ModelAnalyzer, DataAnalyzer, create_comprehensive_analysis_report
from utils.logger import SimpleLogger

def create_sample_config():
    """Create a sample configuration for demonstration."""
    config = {
        'training_mode': 'supervised',  # Start with supervised
        
        'experiment': {
            'max_videos': 3,  # Small number for demo
            'dual_world_model': {
                'train': True,
                'inference': True,
                'best_model_path': None
            }
        },
        
        'training': {
            'epochs': 3,  # Short training for demo
            'batch_size': 8,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'gradient_clip_val': 1.0,
            'num_workers': 2,
            'pin_memory': True,
            'log_every_n_steps': 10,
            'eval_epoch_interval': 1
        },
        
        'data': {
            'context_length': 10,  # Shorter for demo
            'train_shift': 1,
            'padding_value': 0.0,
            'max_horizon': 5,  # Shorter for demo
            'paths': {
                'data_dir': "/path/to/your/CholecT50/data",
                'fold': 0,
                'metadata_file': "your_metadata_file.csv"
            }
        },
        
        'models': {
            'dual_world_model': {
                'hidden_dim': 256,  # Smaller for demo
                'embedding_dim': 512,  # Smaller for demo
                'action_embedding_dim': 64,
                'n_layer': 3,  # Fewer layers for demo
                'num_action_classes': 100,
                'num_phase_classes': 7,
                'autoregressive_action_prediction': True,
                'rl_state_prediction': True,
                'reward_prediction': True,
                'loss_weights': {
                    'state': 1.0,
                    'action': 1.0,
                    'reward': 0.5,
                    'phase': 0.3
                }
            }
        },
        
        'rl_training': {
            'rl_horizon': 20,
            'reward_mode': 'dense',
            'normalize_rewards': True,
            'reward_weights': {
                'phase_completion': 1.0,
                'phase_initiation': 0.5,
                'phase_progression': 1.0,
                'global_progression': 0.8,
                'action_probability': 0.3,
                'risk_penalty': -0.5
            }
        },
        
        'evaluation': {
            'supervised': {
                'action_prediction': {
                    'horizons': [1, 3, 5],
                    'top_ks': [1, 3, 5],
                    'temperature': 1.0
                }
            },
            'rl': {
                'rollout_horizon': 5,
                'eval_horizons': [1, 3, 5]
            }
        },
        
        'preprocess': {
            'extract_rewards': True,
            'rewards': {
                'expert_knowledge': {'risk_score': True},
                'grounded': {
                    'phase_progression': True,
                    'phase_completion': True
                }
            }
        }
    }
    
    return config

def demonstrate_supervised_training(config, logger):
    """Demonstrate supervised training for autoregressive action prediction."""
    logger.info("=== DEMONSTRATING SUPERVISED TRAINING ===")
    
    # Set training mode
    config['training_mode'] = 'supervised'
    
    print("🎯 Goal: Train model to predict future actions autoregressively")
    print("📚 Method: Teacher forcing with GPT-2 backbone")
    print("🔄 Process: Current states → Hidden representations → Future actions")
    
    try:
        # Create mock data for demonstration (replace with real data loading)
        print("\n📊 Loading demonstration data...")
        
        # In real usage, you would load actual data:
        # train_data = load_cholect50_data(config, logger, split='train')
        # test_data = load_cholect50_data(config, logger, split='test')
        
        # For demo, create mock data structure
        mock_data = create_mock_surgical_data(config, num_videos=2)
        
        # Create model
        print("🏗️  Creating Dual World Model...")
        model_config = config['models']['dual_world_model']
        model = DualWorldModel(**model_config)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model has {total_params:,} parameters")
        
        # Demonstrate forward pass
        print("\n🔄 Demonstrating supervised forward pass...")
        demonstrate_supervised_forward_pass(model, mock_data)
        
        # Create trainer
        print("\n🏋️  Creating trainer for supervised mode...")
        trainer = DualTrainer(model, config, logger, 'cpu')  # Use CPU for demo
        
        print("✅ Supervised training setup complete!")
        print("📈 In real training, this would learn action prediction patterns")
        
        return model, mock_data
        
    except Exception as e:
        logger.error(f"Supervised training demo failed: {str(e)}")
        return None, None

def demonstrate_rl_training(model, config, logger, mock_data):
    """Demonstrate RL training for state and reward prediction."""
    logger.info("=== DEMONSTRATING RL TRAINING ===")
    
    if model is None:
        print("❌ No model available for RL demonstration")
        return None
    
    # Set training mode
    config['training_mode'] = 'rl'
    
    print("🎯 Goal: Train model to predict next states and rewards given actions")
    print("🔧 Method: State prediction with reward modeling")
    print("🔄 Process: (Current states, Planned actions) → (Next states, Rewards)")
    
    try:
        print("\n🔄 Demonstrating RL forward pass...")
        demonstrate_rl_forward_pass(model, mock_data)
        
        print("\n🌍 Creating RL environment...")
        demonstrate_rl_environment(model, config, mock_data)
        
        print("✅ RL training setup complete!")
        print("🎮 In real training, this would learn state dynamics and rewards")
        
        return model
        
    except Exception as e:
        logger.error(f"RL training demo failed: {str(e)}")
        return None

def demonstrate_supervised_forward_pass(model, mock_data):
    """Demonstrate supervised mode forward pass."""
    print("   📥 Input: Sequence of frame embeddings")
    print("   🔄 Processing: GPT-2 autoregressive modeling")
    print("   📤 Output: Predicted next actions and states")
    
    model.eval()
    
    # Get sample data
    sample_video = mock_data[0]
    current_states = torch.tensor(sample_video['frame_embeddings'][:5], dtype=torch.float32).unsqueeze(0)
    next_states = torch.tensor(sample_video['frame_embeddings'][1:6], dtype=torch.float32).unsqueeze(0)
    next_actions = torch.tensor(sample_video['actions_binaries'][1:6], dtype=torch.float32).unsqueeze(0)
    
    print(f"   📊 Input shape: {current_states.shape}")
    
    with torch.no_grad():
        outputs = model(
            current_states=current_states,
            next_states=next_states,
            next_actions=next_actions,
            mode='supervised'
        )
    
    print(f"   📊 State predictions shape: {outputs['state_pred'].shape}")
    if 'action_pred' in outputs:
        print(f"   📊 Action predictions shape: {outputs['action_pred'].shape}")
    print(f"   📊 Total loss: {outputs['total_loss'].item():.4f}")
    
    # Demonstrate autoregressive generation
    print("\n   🔮 Demonstrating autoregressive action prediction...")
    generation_output = model.autoregressive_action_prediction(
        initial_states=current_states,
        horizon=3,
        temperature=0.8
    )
    
    print(f"   📊 Generated states shape: {generation_output['predicted_states'].shape}")
    print(f"   📊 Generated actions shape: {generation_output['predicted_actions'].shape}")

def demonstrate_rl_forward_pass(model, mock_data):
    """Demonstrate RL mode forward pass."""
    print("   📥 Input: Current states + Planned actions")
    print("   🔄 Processing: State transition modeling")
    print("   📤 Output: Next states + Reward predictions")
    
    model.eval()
    
    # Get sample data
    sample_video = mock_data[0]
    current_states = torch.tensor(sample_video['frame_embeddings'][:5], dtype=torch.float32).unsqueeze(0)
    planned_actions = torch.tensor(sample_video['actions_binaries'][:5], dtype=torch.float32).unsqueeze(0)
    
    print(f"   📊 Current states shape: {current_states.shape}")
    print(f"   📊 Planned actions shape: {planned_actions.shape}")
    
    with torch.no_grad():
        rl_outputs = model.rl_state_prediction(
            current_states=current_states,
            planned_actions=planned_actions,
            return_rewards=True
        )
    
    print(f"   📊 Predicted next states shape: {rl_outputs['next_states'].shape}")
    
    if 'rewards' in rl_outputs:
        print(f"   📊 Number of reward types predicted: {len(rl_outputs['rewards'])}")
        for reward_type, reward_tensor in rl_outputs['rewards'].items():
            print(f"      - {reward_type}: {reward_tensor.shape}")

def demonstrate_rl_environment(model, config, mock_data):
    """Demonstrate RL environment usage."""
    print("   🌍 Creating surgical RL environment...")
    
    env_config = config['rl_training']
    env = SurgicalWorldModelEnv(model, env_config, 'cpu')
    env.set_video_context(mock_data)
    
    print("   🎮 Testing environment interaction...")
    
    # Reset environment
    obs, info = env.reset(options={'video_id': mock_data[0]['video_id']})
    print(f"   📊 Initial observation shape: {obs.shape}")
    print(f"   📊 Initial info: {info}")
    
    # Take a few steps
    for step in range(3):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {step+1}: reward={reward:.3f}, phase={info.get('current_phase', 'unknown')}")
        
        if terminated or truncated:
            break
    
    print("   ✅ Environment interaction successful!")

def demonstrate_evaluation(model, config, logger, mock_data):
    """Demonstrate comprehensive model evaluation."""
    logger.info("=== DEMONSTRATING MODEL EVALUATION ===")
    
    if model is None:
        print("❌ No model available for evaluation")
        return
    
    print("🔍 Goal: Comprehensive evaluation of both training modes")
    print("📊 Methods: Action accuracy, state prediction, autoregressive performance")
    
    try:
        # Create mock test loaders
        print("\n📊 Creating mock test data...")
        mock_test_loaders = create_mock_test_loaders(mock_data)
        
        # Create evaluator
        print("🔍 Creating comprehensive evaluator...")
        evaluator = DualModelEvaluator(model, config, 'cpu', logger)
        
        # Quick evaluation demo (not full evaluation to save time)
        print("\n📈 Demonstrating evaluation components...")
        
        # 1. Model architecture analysis
        print("🏗️  Analyzing model architecture...")
        analyzer = ModelAnalyzer(model, 'cpu', logger)
        arch_analysis = analyzer.analyze_model_architecture()
        
        print(f"   📊 Total parameters: {arch_analysis['total_params']:,}")
        print(f"   🎯 Capabilities: {', '.join(arch_analysis['capabilities'])}")
        
        # 2. Quick prediction analysis
        print("\n🔮 Analyzing prediction patterns...")
        prediction_analysis = analyzer.analyze_prediction_patterns(mock_test_loaders['mock_video_1'], num_samples=10)
        
        if 'action_analysis' in prediction_analysis:
            action_stats = prediction_analysis['action_analysis']
            print(f"   🎯 Mean action accuracy: {action_stats['mean_class_accuracy']:.3f}")
            print(f"   🎯 Prediction confidence: {action_stats['average_prediction_confidence']:.3f}")
        
        if 'state_analysis' in prediction_analysis:
            state_stats = prediction_analysis['state_analysis']
            print(f"   🌍 State prediction correlation: {state_stats['mean_correlation']:.3f}")
        
        print("✅ Evaluation demonstration complete!")
        print("📈 In real evaluation, this would run comprehensive metrics")
        
    except Exception as e:
        logger.error(f"Evaluation demo failed: {str(e)}")

def create_mock_surgical_data(config, num_videos=2):
    """Create mock surgical data for demonstration."""
    print("🏥 Creating mock surgical video data...")
    
    embedding_dim = config['models']['dual_world_model']['embedding_dim']
    num_actions = config['models']['dual_world_model']['num_action_classes']
    num_phases = config['models']['dual_world_model']['num_phase_classes']
    
    mock_data = []
    
    for i in range(num_videos):
        video_length = np.random.randint(50, 100)  # Random video length
        
        # Create mock embeddings (random but realistic)
        frame_embeddings = np.random.randn(video_length, embedding_dim).astype(np.float32)
        
        # Create mock actions (sparse binary)
        actions_binaries = np.zeros((video_length, num_actions), dtype=np.float32)
        for t in range(video_length):
            # Randomly activate 1-3 actions per frame
            num_active = np.random.randint(1, 4)
            active_indices = np.random.choice(num_actions, num_active, replace=False)
            actions_binaries[t, active_indices] = 1.0
        
        # Create mock phases (sequential progression)
        phase_binaries = np.zeros((video_length, num_phases), dtype=np.float32)
        for t in range(video_length):
            # Simple phase progression
            phase = min(t // (video_length // num_phases), num_phases - 1)
            phase_binaries[t, phase] = 1.0
        
        # Create mock instruments
        instruments_binaries = np.zeros((video_length, 6), dtype=np.float32)
        for t in range(video_length):
            # Randomly activate 1-2 instruments
            num_active = np.random.randint(1, 3)
            active_indices = np.random.choice(6, num_active, replace=False)
            instruments_binaries[t, active_indices] = 1.0
        
        # Create mock rewards
        next_rewards = {
            '_r_phase_completion': np.random.uniform(0, 1, video_length-1),
            '_r_phase_initiation': np.random.uniform(0, 0.5, video_length-1),
            '_r_phase_progression': np.random.uniform(0, 1, video_length-1),
            '_r_global_progression': np.random.uniform(0, 1, video_length-1),
            '_r_action_probability': np.random.uniform(0, 0.8, video_length-1),
            '_r_risk': np.random.uniform(0.5, 1.0, video_length-1)  # Risk penalty
        }
        
        video_data = {
            'video_id': f'mock_video_{i+1}',
            'video_dir': f'/mock/path/video_{i+1}',
            'frame_embeddings': frame_embeddings,
            'actions_binaries': actions_binaries,
            'instruments_binaries': instruments_binaries,
            'phase_binaries': phase_binaries,
            'num_frames': video_length,
            'next_rewards': next_rewards,
            'outcomes': {}
        }
        
        mock_data.append(video_data)
    
    print(f"✅ Created {num_videos} mock videos with avg length {np.mean([v['num_frames'] for v in mock_data]):.0f} frames")
    
    return mock_data

def create_mock_test_loaders(mock_data):
    """Create mock test loaders for evaluation demo."""
    from torch.utils.data import DataLoader
    
    # Create simple mock dataset
    class MockDataset:
        def __init__(self, video_data):
            self.video_data = video_data
            
        def __len__(self):
            return min(10, self.video_data['num_frames'] - 1)  # Small for demo
            
        def __getitem__(self, idx):
            return {
                'current_states': torch.tensor(self.video_data['frame_embeddings'][idx:idx+5], dtype=torch.float32),
                'next_states': torch.tensor(self.video_data['frame_embeddings'][idx+1:idx+6], dtype=torch.float32),
                'next_actions': torch.tensor(self.video_data['actions_binaries'][idx+1:idx+6], dtype=torch.float32),
                'next_phases': torch.tensor(self.video_data['phase_binaries'][idx+1:idx+6], dtype=torch.float32)
            }
    
    mock_loaders = {}
    for video_data in mock_data:
        dataset = MockDataset(video_data)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        mock_loaders[video_data['video_id']] = loader
    
    return mock_loaders

def main():
    """Main function to run the demonstration."""
    print("🚀 DUAL WORLD MODEL DEMONSTRATION")
    print("=" * 50)
    print("This demo shows how to use the dual world model for:")
    print("1. 🎯 Supervised autoregressive action prediction")
    print("2. 🌍 RL state and reward prediction")
    print("3. 🔍 Comprehensive model evaluation")
    print("=" * 50)
    
    # Create logger
    logger = SimpleLogger(log_dir="demo_logs", name="dual_world_demo")
    
    # Create configuration
    config = create_sample_config()
    
    # Note about data
    print("\n📝 NOTE: This demo uses mock data for demonstration.")
    print("   In real usage, replace create_mock_surgical_data() with:")
    print("   train_data = load_cholect50_data(config, logger, split='train')")
    print("   test_data = load_cholect50_data(config, logger, split='test')")
    
    print("\n" + "=" * 50)
    
    # 1. Demonstrate Supervised Training
    model, mock_data = demonstrate_supervised_training(config, logger)
    
    print("\n" + "=" * 50)
    
    # 2. Demonstrate RL Training  
    model = demonstrate_rl_training(model, config, logger, mock_data)
    
    print("\n" + "=" * 50)
    
    # 3. Demonstrate Evaluation
    demonstrate_evaluation(model, config, logger, mock_data)
    
    print("\n" + "=" * 50)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("=" * 50)
    print("Next steps for real usage:")
    print("1. 📊 Set up your CholecT50 dataset paths in config")
    print("2. 🏋️  Run full training with: python updated_main_experiment.py")
    print("3. 🔍 Analyze results with the comprehensive evaluation tools")
    print("4. 🎮 Use trained model for RL experiments")
    print("\nHappy training! 🤖✨")

if __name__ == "__main__":
    main()