
# Test script to verify RL integration
import torch
import numpy as np
from models import WorldModel, SurgicalWorldModelEnv


def test_integration_fixed():
    """Test RL environment integration with fixed world model"""
    print("Testing RL integration with fixed world model...")
    
    # Create a world model for testing with proper config
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
        'action_conditioning': True,
        'outcome_learning': False  # Add this
    }
    
    model = WorldModel(**config)
    model.eval()  # Set to evaluation mode
    
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
        print(f"  Reward breakdown: {info.get('reward_breakdown', {})}")
        
        if done:
            break
    
    print("Fixed integration test completed successfully!")

if __name__ == "__main__":
    test_integration_fixed()

# def test_integration():
#     """Test RL environment integration"""
#     print("Testing RL integration...")
    
#     # Create a dummy world model for testing
#     config = {
#         'hidden_dim': 768,
#         'embedding_dim': 1024,
#         'action_embedding_dim': 100,
#         'n_layer': 6,
#         'num_action_classes': 100,
#         'num_phase_classes': 7,
#         'reward_learning': True,
#         'action_learning': True,
#         'phase_learning': True,
#         'imitation_learning': True,
#         'action_conditioning': True
#     }
    
#     model = WorldModel(**config)
    
#     # Create environment
#     env_config = {
#         'rl_horizon': 10,
#         'context_length': 5,
#         'reward_weights': {
#             '_r_phase_completion': 1.0,
#             '_r_risk': -0.5
#         }
#     }
    
#     env = SurgicalWorldModelEnv(model, env_config, device='cpu')
    
#     # Test basic functionality
#     obs, info = env.reset()
#     print(f"Initial observation shape: {obs.shape}")
    
#     for i in range(3):
#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         print(f"Step {i+1}: reward={reward:.3f}, done={done}")
        
#         if done:
#             break
    
#     print("Integration test completed successfully!")

# if __name__ == "__main__":
#     test_integration()
