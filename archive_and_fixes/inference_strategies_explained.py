# ===================================================================
# File: inference_strategies_explained.py
# Detailed explanation of different inference approaches for action prediction
# ===================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from enum import Enum

class InferenceStrategy(Enum):
    """Different inference strategies for action prediction"""
    SINGLE_STEP = "single_step"           # Predict only next action at each timestep
    RECEDING_HORIZON = "receding_horizon" # Predict short future, use first action, re-plan
    FULL_TRAJECTORY = "full_trajectory"   # Generate entire trajectory once
    AUTOREGRESSIVE = "autoregressive"     # Generate step-by-step using own predictions

class ActionPredictionInference:
    """
    Comprehensive explanation and implementation of different inference strategies
    """
    
    def __init__(self, video_length: int = 100, num_classes: int = 100, max_horizon: int = 15):
        self.video_length = video_length
        self.num_classes = num_classes  
        self.max_horizon = max_horizon
        
    def explain_inference_strategies(self):
        """
        Explain the different inference strategies and their use cases
        """
        
        print("🎯 INFERENCE STRATEGIES FOR SURGICAL ACTION PREDICTION")
        print("=" * 70)
        
        strategies = {
            "Single-Step Prediction": {
                "description": "At each timestep, predict only the immediate next action",
                "shape": "(video_length, num_classes)",
                "use_case": "Real-time systems, immediate action guidance",
                "pros": ["Fast inference", "No error accumulation", "Ground truth available for context"],
                "cons": ["No future planning", "Myopic decisions", "Cannot optimize for long-term goals"]
            },
            
            "Receding Horizon": {
                "description": "At each timestep, predict short future sequence, execute first action, re-plan",
                "shape": "(video_length, 1, num_classes) - only first action used",
                "use_case": "Model Predictive Control, online planning with replanning",
                "pros": ["Considers future consequences", "Robust to model errors", "Adaptive to changes"],
                "cons": ["Computationally expensive", "Still some myopia", "Frequent replanning overhead"]
            },
            
            "Full Trajectory Generation": {
                "description": "Generate entire action sequence once at the beginning",
                "shape": "(video_length, num_classes)",
                "use_case": "Offline planning, procedure preview, training data generation",
                "pros": ["Globally optimal", "Consistent planning", "Fast execution"],
                "cons": ["No adaptation to reality", "Error accumulation", "Unrealistic for real surgery"]
            },
            
            "Autoregressive": {
                "description": "Generate actions step-by-step, using own predictions as input for next step",
                "shape": "(video_length, num_classes)",
                "use_case": "Sequence modeling, language model style generation",
                "pros": ["Natural for sequence models", "Can capture dependencies", "Flexible length"],
                "cons": ["Error accumulation", "Slow inference", "Exposure bias problem"]
            }
        }
        
        for strategy_name, details in strategies.items():
            print(f"\n📊 {strategy_name.upper()}")
            print(f"   Description: {details['description']}")
            print(f"   Output Shape: {details['shape']}")
            print(f"   Use Case: {details['use_case']}")
            print(f"   Pros: {', '.join(details['pros'])}")
            print(f"   Cons: {', '.join(details['cons'])}")
    
    def single_step_inference(self, model, video_embeddings: np.ndarray, 
                             ground_truth_actions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single-step inference: predict next action at each timestep
        
        This is the most practical approach for real surgical systems.
        At each timestep t, we predict action for timestep t+1.
        
        Args:
            model: Trained model (IL or RL)
            video_embeddings: (video_length, embedding_dim)
            ground_truth_actions: (video_length, num_classes) - for teacher forcing in IL
            
        Returns:
            predictions: (video_length-1, num_classes) - one prediction per timestep
        """
        
        print("🔄 Running Single-Step Inference...")
        
        predictions = []
        video_length = len(video_embeddings)
        
        for t in range(video_length - 1):  # Predict t+1 from t
            
            # Current state at timestep t
            current_state = video_embeddings[t]  # (embedding_dim,)
            
            if hasattr(model, 'predict_next_action'):  # Imitation Learning
                # Use current state to predict next action
                with torch.no_grad():
                    state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
                    action_probs = model.predict_next_action(state_tensor)
                    action_pred = (action_probs.squeeze().cpu().numpy() > 0.5).astype(float)
                    
            else:  # RL Policy
                # Use current state to get action from policy
                action_pred, _ = model.predict(current_state.reshape(1, -1), deterministic=True)
                action_pred = action_pred.flatten()
            
            predictions.append(action_pred)
        
        predictions = np.array(predictions)  # Shape: (video_length-1, num_classes)
        
        print(f"   Output shape: {predictions.shape}")
        print(f"   Predictions per timestep: 1")
        print(f"   Total predictions made: {len(predictions)}")
        
        return predictions
    
    def receding_horizon_inference(self, model, video_embeddings: np.ndarray, 
                                  horizon: int = 5) -> np.ndarray:
        """
        Receding horizon inference: predict short future, use first action, re-plan
        
        This approach balances immediate needs with future planning.
        At each timestep, predict 'horizon' future actions but only execute the first one.
        
        Args:
            model: Trained model
            video_embeddings: (video_length, embedding_dim)
            horizon: Number of future steps to predict
            
        Returns:
            executed_actions: (video_length-1, num_classes) - first action from each plan
        """
        
        print(f"🔮 Running Receding Horizon Inference (horizon={horizon})...")
        
        executed_actions = []
        video_length = len(video_embeddings)
        
        for t in range(video_length - 1):
            
            # Current state
            current_state = video_embeddings[t]
            
            if hasattr(model, 'generate_conditional_future_states'):  # World Model
                # Generate future states and actions
                state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    future_predictions = model.generate_conditional_future_states(
                        input_embeddings=state_tensor,
                        horizon=min(horizon, video_length - t - 1),
                        do_sample=True
                    )
                    
                    # Extract first action from the planned sequence
                    if 'generated_actions' in future_predictions:
                        first_action = future_predictions['generated_actions'][0, 0].cpu().numpy()
                    else:
                        # Fallback to single prediction
                        action_probs = model.predict_next_action(state_tensor)
                        first_action = (action_probs.squeeze().cpu().numpy() > 0.5).astype(float)
                        
            else:  # RL Policy (simulate planning)
                # For RL, we could run multiple forward passes or use the policy directly
                action_pred, _ = model.predict(current_state.reshape(1, -1), deterministic=True)
                first_action = action_pred.flatten()
            
            executed_actions.append(first_action)
        
        executed_actions = np.array(executed_actions)  # Shape: (video_length-1, num_classes)
        
        print(f"   Output shape: {executed_actions.shape}")
        print(f"   Planning horizon: {horizon}")
        print(f"   Actions executed: {len(executed_actions)}")
        
        return executed_actions
    
    def full_trajectory_inference(self, model, initial_state: np.ndarray, 
                                 target_length: int) -> np.ndarray:
        """
        Full trajectory generation: create entire action sequence at once
        
        This generates the complete trajectory from start to finish without 
        intermediate corrections. Useful for offline planning or analysis.
        
        Args:
            model: Trained model
            initial_state: (embedding_dim,) - starting state
            target_length: Length of trajectory to generate
            
        Returns:
            full_trajectory: (target_length, num_classes)
        """
        
        print(f"🎬 Running Full Trajectory Generation (length={target_length})...")
        
        if hasattr(model, 'generate_conditional_future_states'):  # World Model
            
            state_tensor = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                trajectory_prediction = model.generate_conditional_future_states(
                    input_embeddings=state_tensor,
                    horizon=target_length,
                    do_sample=True,
                    temperature=0.8
                )
                
                if 'generated_actions' in trajectory_prediction:
                    full_trajectory = trajectory_prediction['generated_actions'][0].cpu().numpy()
                else:
                    # Fallback: generate step by step
                    full_trajectory = self._autoregressive_generation(model, initial_state, target_length)
                    
        else:  # RL Policy - simulate trajectory
            # For RL, we need to simulate the environment or use a simple rollout
            full_trajectory = self._rl_trajectory_simulation(model, initial_state, target_length)
        
        print(f"   Output shape: {full_trajectory.shape}")
        print(f"   Generated at once: True")
        print(f"   Total timesteps: {len(full_trajectory)}")
        
        return full_trajectory
    
    def autoregressive_inference(self, model, initial_state: np.ndarray, 
                                target_length: int) -> np.ndarray:
        """
        Autoregressive inference: generate step-by-step using own predictions
        
        This mimics language model generation where each prediction depends 
        on all previous predictions.
        
        Args:
            model: Trained world model
            initial_state: (embedding_dim,) - starting state  
            target_length: Length of sequence to generate
            
        Returns:
            autoregressive_trajectory: (target_length, num_classes)
        """
        
        print(f"🔄 Running Autoregressive Generation (length={target_length})...")
        
        trajectory = []
        current_state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)
        
        for step in range(target_length):
            
            with torch.no_grad():
                # Predict next action from current state
                action_probs = model.predict_next_action(current_state.unsqueeze(0))
                action_pred = (action_probs.squeeze().cpu().numpy() > 0.5).astype(float)
                trajectory.append(action_pred)
                
                # Update state using world model prediction
                if hasattr(model, 'forward'):
                    # Use the action to predict next state
                    action_tensor = torch.tensor(action_pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    
                    try:
                        output = model(
                            current_state=current_state.unsqueeze(0),
                            next_actions=action_tensor,
                            eval_mode='basic'
                        )
                        
                        if '_z_hat' in output:
                            current_state = output['_z_hat'].squeeze(0).squeeze(0)
                        else:
                            # Add noise to current state as fallback
                            current_state = current_state + torch.randn_like(current_state) * 0.01
                            
                    except:
                        # Fallback: add small random perturbation
                        current_state = current_state + torch.randn_like(current_state) * 0.01
        
        autoregressive_trajectory = np.array(trajectory)  # Shape: (target_length, num_classes)
        
        print(f"   Output shape: {autoregressive_trajectory.shape}")
        print(f"   Error accumulation: Possible")
        print(f"   State updates: {target_length}")
        
        return autoregressive_trajectory
    
    def _autoregressive_generation(self, model, initial_state: np.ndarray, 
                                  target_length: int) -> np.ndarray:
        """Helper for autoregressive generation"""
        return self.autoregressive_inference(model, initial_state, target_length)
    
    def _rl_trajectory_simulation(self, model, initial_state: np.ndarray, 
                                 target_length: int) -> np.ndarray:
        """Helper to simulate RL trajectory"""
        trajectory = []
        current_state = initial_state.copy()
        
        for step in range(target_length):
            action_pred, _ = model.predict(current_state.reshape(1, -1), deterministic=True)
            trajectory.append(action_pred.flatten())
            
            # Simple state transition (you'd use your actual environment here)
            current_state = current_state + np.random.normal(0, 0.01, current_state.shape)
        
        return np.array(trajectory)
    
    def compare_inference_strategies(self, model, video_embeddings: np.ndarray, 
                                   ground_truth_actions: np.ndarray):
        """
        Compare all inference strategies on the same video
        """
        
        print("\n🔍 COMPARING INFERENCE STRATEGIES")
        print("=" * 50)
        
        results = {}
        
        # 1. Single-step inference
        try:
            single_step_preds = self.single_step_inference(model, video_embeddings)
            results['single_step'] = {
                'predictions': single_step_preds,
                'shape': single_step_preds.shape,
                'strategy': 'Predict next action at each timestep'
            }
        except Exception as e:
            print(f"Single-step inference failed: {e}")
        
        # 2. Receding horizon inference  
        try:
            receding_preds = self.receding_horizon_inference(model, video_embeddings, horizon=5)
            results['receding_horizon'] = {
                'predictions': receding_preds,
                'shape': receding_preds.shape,
                'strategy': 'Plan 5 steps ahead, execute first'
            }
        except Exception as e:
            print(f"Receding horizon inference failed: {e}")
        
        # 3. Full trajectory generation
        try:
            full_traj_preds = self.full_trajectory_inference(
                model, video_embeddings[0], len(video_embeddings)-1
            )
            results['full_trajectory'] = {
                'predictions': full_traj_preds,
                'shape': full_traj_preds.shape,
                'strategy': 'Generate entire trajectory at once'
            }
        except Exception as e:
            print(f"Full trajectory inference failed: {e}")
        
        # 4. Autoregressive generation
        try:
            autoregr_preds = self.autoregressive_inference(
                model, video_embeddings[0], len(video_embeddings)-1  
            )
            results['autoregressive'] = {
                'predictions': autoregr_preds,
                'shape': autoregr_preds.shape,
                'strategy': 'Step-by-step using own predictions'
            }
        except Exception as e:
            print(f"Autoregressive inference failed: {e}")
        
        # Compare results
        print(f"\n📊 STRATEGY COMPARISON:")
        print(f"   Ground truth shape: {ground_truth_actions.shape}")
        
        for strategy_name, result in results.items():
            print(f"\n   {strategy_name.upper()}:")
            print(f"      Shape: {result['shape']}")
            print(f"      Strategy: {result['strategy']}")
            
            # Compute simple accuracy if shapes match
            if result['predictions'].shape == ground_truth_actions[1:].shape:
                accuracy = np.mean(result['predictions'] == ground_truth_actions[1:])
                print(f"      Accuracy: {accuracy:.3f}")
        
        return results
    
    def create_evaluation_matrix_explanation(self):
        """
        Explain what the evaluation matrices look like for mAP trajectory analysis
        """
        
        print("\n📋 EVALUATION MATRIX SHAPES FOR mAP TRAJECTORY ANALYSIS")
        print("=" * 70)
        
        print("For mAP trajectory analysis, we need to understand what we're evaluating:")
        
        scenarios = {
            "Cumulative Evaluation": {
                "description": "At each timestep t, evaluate predictions from start to t",
                "prediction_matrix": "(video_length, num_classes)",
                "evaluation_points": "video_length",
                "mAP_computation": "For timestep t, compute mAP using predictions[0:t] vs ground_truth[0:t]"
            },
            
            "Sliding Window Evaluation": {
                "description": "At each timestep t, evaluate predictions for next horizon steps",
                "prediction_matrix": "(video_length, horizon, num_classes)",
                "evaluation_points": "video_length",
                "mAP_computation": "For timestep t, compute mAP using predictions[t, :] vs ground_truth[t:t+horizon]"
            },
            
            "Fixed Horizon Evaluation": {
                "description": "From each timestep t, predict fixed horizon, evaluate degradation",
                "prediction_matrix": "(video_length, horizon, num_classes)",
                "evaluation_points": "horizon",
                "mAP_computation": "For horizon h, compute mAP using predictions[:, h] vs ground_truth at h steps ahead"
            }
        }
        
        for scenario_name, details in scenarios.items():
            print(f"\n🎯 {scenario_name.upper()}")
            print(f"   Description: {details['description']}")
            print(f"   Prediction Matrix Shape: {details['prediction_matrix']}")
            print(f"   Evaluation Points: {details['evaluation_points']}")
            print(f"   mAP Computation: {details['mAP_computation']}")
        
        print(f"\n💡 RECOMMENDATION FOR YOUR PAPER:")
        print(f"   For 'mAP degradation over time', use CUMULATIVE EVALUATION:")
        print(f"   - At each timestep t, you have predictions from timestep 0 to t")
        print(f"   - Compute mAP using all predictions up to timestep t")
        print(f"   - This shows how prediction quality changes as you get further from start")
        print(f"   - Matrix shape: (video_length, num_classes)")
        print(f"   - mAP trajectory shape: (video_length,) - one mAP per timestep")

def demonstrate_inference_strategies():
    """
    Demonstrate different inference strategies with example
    """
    
    # Create example data
    video_length = 50
    num_classes = 100
    embedding_dim = 768
    
    # Simulate video embeddings and ground truth
    video_embeddings = np.random.randn(video_length, embedding_dim)
    ground_truth_actions = (np.random.rand(video_length, num_classes) > 0.9).astype(float)
    
    # Create inference explainer
    explainer = ActionPredictionInference(video_length, num_classes)
    
    # Explain strategies
    explainer.explain_inference_strategies()
    
    # Explain evaluation matrices
    explainer.create_evaluation_matrix_explanation()
    
    print(f"\n" + "="*70)
    print("🎯 SUMMARY FOR YOUR RESEARCH:")
    print("="*70)
    
    print(f"""
📊 RECOMMENDED APPROACH FOR YOUR PAPER:

1. **Inference Strategy**: Use SINGLE-STEP inference for fair comparison
   - At each timestep t, predict action for timestep t+1
   - This is most realistic for surgical applications
   - Avoids error accumulation bias between methods

2. **Evaluation Strategy**: Use CUMULATIVE evaluation for mAP trajectory
   - At timestep t, evaluate predictions from start to timestep t
   - Shows how mAP degrades as you move further from initial context
   - Perfect for your "mAP degradation over trajectory length" analysis

3. **Matrix Shapes**:
   - Predictions: (video_length-1, num_classes) - one prediction per timestep
   - Ground Truth: (video_length-1, num_classes) - corresponding ground truth
   - mAP Trajectory: (video_length-1,) - cumulative mAP at each timestep

4. **Comparison Fairness**:
   - All methods (IL, PPO, SAC) use same inference strategy
   - Same evaluation protocol for all methods
   - Statistical tests on same prediction matrices
""")

if __name__ == "__main__":
    demonstrate_inference_strategies()
