# ===================================================================
# File: setup_rl_experiment.py
# Script to integrate RL components with your existing codebase
# ===================================================================

import os
import shutil
import sys
from pathlib import Path

def setup_rl_experiment():
    """
    Setup script to integrate RL components with existing codebase
    """
    
    print("Setting up RL experiment integration...")
    
    # Create necessary directories
    directories = [
        'rl_logs',
        'ppo_surgical_logs', 
        'sac_surgical_logs',
        'rl_checkpoints',
        'rl_results'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    # Check if required files exist
    required_files = [
        'models/world_model.py',
        'datasets/cholect50.py',  # or wherever your cholect50.py is
        'config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("WARNING: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please ensure these files exist before running RL experiments.")
    
    # Create a simple integration script
    integration_script = '''
# Integration script for your existing codebase
import sys
import os

# Add your project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from world_model import WorldModel
from datasets.cholect50 import load_cholect50_data, NextFramePredictionDataset

# Import new RL components
from rl_environment import SurgicalWorldModelEnv
from rl_trainer import RLExperimentRunner

def main():
    """Run the integrated RL experiment"""
    from run_rl_experiments import run_rl_comparison_experiment
    
    # Run the experiment with your config
    results, comparison = run_rl_comparison_experiment('config_rl.yaml')
    
    print("Experiment completed!")
    print("Check rl_comparison_results.json for detailed results")

if __name__ == "__main__":
    main()
'''
    
    with open('run_integrated_rl_experiment.py', 'w') as f:
        f.write(integration_script)
    
    print("Created integration script: run_integrated_rl_experiment.py")
    
    # Create a test script
    test_script = '''
# Test script to verify RL integration
import torch
import numpy as np
from rl_environment import SurgicalWorldModelEnv
from world_model import WorldModel

def test_integration():
    """Test RL environment integration"""
    print("Testing RL integration...")
    
    # Create a dummy world model for testing
    config = {
        'hidden_dim': 768,
        'embedding_dim': 1024,
        'action_embedding_dim': 100,
        'n_layer': 6,
        'num_action_classes': 100,
        'num_phase_classes': 7,
        'reward_learning': True,
        'action_learning': True,
        'phase_learning': True,
        'imitation_learning': True,
        'action_conditioning': True
    }
    
    model = WorldModel(**config)
    
    # Create environment
    env_config = {
        'rl_horizon': 10,
        'context_length': 5,
        'reward_weights': {
            '_r_phase_completion': 1.0,
            '_r_risk': -0.5
        }
    }
    
    env = SurgicalWorldModelEnv(model, env_config, device='cpu')
    
    # Test basic functionality
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}")
        
        if done:
            break
    
    print("Integration test completed successfully!")

if __name__ == "__main__":
    test_integration()
'''
    
    with open('test_rl_integration.py', 'w') as f:
        f.write(test_script)
    
    print("Created test script: test_rl_integration.py")
    
    print("\n" + "="*50)
    print("RL INTEGRATION SETUP COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements_rl.txt")
    print("2. Test integration: python test_rl_integration.py")
    print("3. Run RL experiments: python run_integrated_rl_experiment.py")
    print("4. Check results in: rl_comparison_results.json")
    print("\nMake sure your world model is trained and the path in config_rl.yaml is correct!")

if __name__ == "__main__":
    setup_rl_experiment()