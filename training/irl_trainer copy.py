#!/usr/bin/env python3
"""
Specific IRL Implementation: Maximum Entropy IRL + Lightweight GAIL
No stable-baselines dependency - custom implementation for surgical next action prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random
from tqdm import tqdm

class MaxEntIRL(nn.Module):
    """
    Maximum Entropy Inverse Reinforcement Learning
    
    Key idea: Learn reward function that makes expert demonstrations 
    have maximum entropy while being optimal under that reward
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4):
        super(MaxEntIRL, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Reward network: (state, action) -> scalar reward
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)
        
    def compute_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute learned reward for state-action pairs"""
        # Concatenate state and action
        sa_pairs = torch.cat([states, actions], dim=-1)
        rewards = self.reward_net(sa_pairs)
        return rewards.squeeze(-1)
    
    def learn_reward_from_trajectories(self, expert_trajectories: List[Dict], 
                                     num_iterations: int = 100):
        """
        Learn reward function using Maximum Entropy IRL
        
        Algorithm:
        1. Sample random policy trajectories
        2. Compute feature expectations for expert vs random
        3. Update reward to make expert trajectories more likely
        """
        
        print(f"🎯 Learning rewards using MaxEnt IRL ({num_iterations} iterations)")
        
        # Extract expert state-action pairs
        expert_states = []
        expert_actions = []
        
        for traj in expert_trajectories:
            states = torch.tensor(traj['states'], dtype=torch.float32)
            actions = torch.tensor(traj['actions'], dtype=torch.float32)
            
            expert_states.append(states)
            expert_actions.append(actions)
        
        expert_states = torch.cat(expert_states, dim=0)
        expert_actions = torch.cat(expert_actions, dim=0)
        
        # Training loop
        for iteration in range(num_iterations):
            # Sample random trajectories (negative examples)
            random_actions = self._sample_random_actions(expert_states.shape[0])
            
            # Compute rewards
            expert_rewards = self.compute_reward(expert_states, expert_actions)
            random_rewards = self.compute_reward(expert_states, random_actions)
            
            # MaxEnt IRL loss: expert actions should have higher reward
            # with maximum entropy regularization
            loss = -torch.mean(expert_rewards) + torch.mean(torch.exp(random_rewards))
            
            # Regularization to prevent reward explosion
            l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.reward_net.parameters())
            total_loss = loss + l2_reg
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
            self.optimizer.step()
            
            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: Loss = {total_loss.item():.4f}, "
                      f"Expert Reward = {expert_rewards.mean().item():.3f}, "
                      f"Random Reward = {random_rewards.mean().item():.3f}")
        
        print("✅ MaxEnt IRL training completed")
    
    def _sample_random_actions(self, num_samples: int) -> torch.Tensor:
        """Sample random actions for negative examples"""
        # Sample sparse binary actions (similar to surgical action structure)
        random_actions = torch.rand(num_samples, self.action_dim)
        # Make actions sparse (only ~10% of actions active)
        random_actions = (random_actions > 0.9).float()
        return random_actions

class LightweightGAIL:
    """
    Lightweight GAIL implementation for surgical action prediction
    
    Key difference from full GAIL: 
    - Uses existing IL model as generator base
    - Only learns small adjustments to IL predictions
    - Focuses on specific scenarios where IL struggles
    """
    
    def __init__(self, il_model, state_dim: int, action_dim: int, lr: float = 1e-4):
        self.il_model = il_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Discriminator: tries to distinguish expert from policy actions
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability that action is from expert
        )
        
        # Policy adjustment network: small corrections to IL predictions
        self.policy_adjustment = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Bounded adjustments [-1, 1]
        )
        
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_adjustment.parameters(), lr=lr)
        
    def train_gail(self, expert_trajectories: List[Dict], num_epochs: int = 50):
        """
        Train GAIL discriminator and policy adjustment
        """
        
        print(f"🎮 Training Lightweight GAIL ({num_epochs} epochs)")
        
        # Prepare expert data
        expert_states = []
        expert_actions = []
        
        for traj in expert_trajectories:
            states = torch.tensor(traj['states'], dtype=torch.float32)
            actions = torch.tensor(traj['actions'], dtype=torch.float32)
            expert_states.append(states)
            expert_actions.append(actions)
        
        expert_states = torch.cat(expert_states, dim=0)
        expert_actions = torch.cat(expert_actions, dim=0)
        
        for epoch in range(num_epochs):
            epoch_disc_loss = 0
            epoch_policy_loss = 0
            num_batches = 0
            
            # Sample batches
            batch_size = 32
            num_samples = len(expert_states)
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_states = expert_states[i:end_idx]
                batch_expert_actions = expert_actions[i:end_idx]
                
                # Generate policy actions using IL + adjustment
                with torch.no_grad():
                    il_predictions = self.il_model.predict_next_action(
                        batch_states.unsqueeze(1)  # Add sequence dimension
                    ).squeeze(1)
                
                # Apply policy adjustment
                adjustments = self.policy_adjustment(il_predictions)
                policy_actions = torch.sigmoid(il_predictions + 0.1 * adjustments)
                
                # Train discriminator
                self._train_discriminator_step(
                    batch_states, batch_expert_actions, policy_actions
                )
                
                # Train policy (every few discriminator updates)
                if num_batches % 3 == 0:
                    policy_loss = self._train_policy_step(batch_states, policy_actions)
                    epoch_policy_loss += policy_loss
                
                num_batches += 1
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Policy Loss = {epoch_policy_loss/max(1, num_batches//3):.4f}")
        
        print("✅ Lightweight GAIL training completed")
    
    def _train_discriminator_step(self, states: torch.Tensor, 
                                expert_actions: torch.Tensor, 
                                policy_actions: torch.Tensor):
        """Single discriminator training step"""
        
        # Expert samples (label = 1)
        expert_input = torch.cat([states, expert_actions], dim=-1)
        expert_pred = self.discriminator(expert_input)
        expert_loss = F.binary_cross_entropy(expert_pred, torch.ones_like(expert_pred))
        
        # Policy samples (label = 0)
        policy_input = torch.cat([states, policy_actions.detach()], dim=-1)
        policy_pred = self.discriminator(policy_input)
        policy_loss = F.binary_cross_entropy(policy_pred, torch.zeros_like(policy_pred))
        
        # Total discriminator loss
        disc_loss = expert_loss + policy_loss
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return disc_loss.item()
    
    def _train_policy_step(self, states: torch.Tensor, policy_actions: torch.Tensor):
        """Single policy training step"""
        
        # Policy tries to fool discriminator
        policy_input = torch.cat([states, policy_actions], dim=-1)
        disc_pred = self.discriminator(policy_input)
        
        # Policy loss: maximize discriminator's belief that actions are expert
        policy_loss = F.binary_cross_entropy(disc_pred, torch.ones_like(disc_pred))
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def get_adjusted_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """Get IL prediction with GAIL adjustment"""
        
        with torch.no_grad():
            # Get IL baseline prediction
            il_pred = self.il_model.predict_next_action(state.unsqueeze(0).unsqueeze(0))
            il_pred = il_pred.squeeze()
            
            # Apply learned adjustment
            adjustment = self.policy_adjustment(il_pred)
            adjusted_pred = torch.sigmoid(il_pred + 0.1 * adjustment)
            
            return adjusted_pred

class ScenarioSpecificIRL:
    """
    Main class combining MaxEnt IRL + Lightweight GAIL for different surgical scenarios
    """
    
    def __init__(self, il_model, config, logger, device='cuda'):
        self.il_model = il_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # IRL components for each scenario
        self.maxent_irl_models = {}
        self.gail_models = {}
        
        # Scenario classification (reuse from previous implementation)
        from models.irl_model import SurgicalScenarioClassifier
        self.scenario_classifier = SurgicalScenarioClassifier(config)
        
    def train_scenario_specific_irl(self, train_data: List[Dict], scenarios_to_train: List[str] = None):
        """Train IRL models for specific scenarios"""
        
        if scenarios_to_train is None:
            scenarios_to_train = ['high_complexity', 'rare_actions', 'critical_moments', 'phase_transitions']
        
        self.logger.info(f"🎯 Training Scenario-Specific IRL for: {scenarios_to_train}")
        
        # Classify all training data by scenario
        all_scenario_data = defaultdict(list)
        
        for video in tqdm(train_data, desc="Classifying training videos"):
            video_scenarios = self.scenario_classifier.classify_video_scenarios(video)
            
            for scenario_type, frame_indices in video_scenarios.items():
                if scenario_type in scenarios_to_train and frame_indices:
                    # Extract relevant frames for this scenario
                    scenario_trajectories = {
                        'video_id': video['video_id'],
                        'states': video['frame_embeddings'][frame_indices],
                        'actions': video['actions_binaries'][frame_indices]
                    }
                    all_scenario_data[scenario_type].append(scenario_trajectories)
        
        # Train IRL for each scenario
        for scenario_type in scenarios_to_train:
            if scenario_type not in all_scenario_data:
                self.logger.warning(f"No data found for scenario: {scenario_type}")
                continue
            
            self.logger.info(f"🔧 Training IRL for scenario: {scenario_type}")
            scenario_data = all_scenario_data[scenario_type]
            
            # Combine all video data for this scenario
            combined_trajectories = []
            for video_data in scenario_data:
                combined_trajectories.append({
                    'states': video_data['states'],
                    'actions': video_data['actions']
                })
            
            if not combined_trajectories:
                continue
            
            # Train MaxEnt IRL for reward learning
            maxent_irl = MaxEntIRL(
                state_dim=1024,  # Your embedding dimension
                action_dim=100,   # Your action classes
                lr=1e-4
            ).to(self.device)
            
            maxent_irl.learn_reward_from_trajectories(
                combined_trajectories, num_iterations=100
            )
            
            self.maxent_irl_models[scenario_type] = maxent_irl
            
            # Train Lightweight GAIL for policy adjustment
            gail_model = LightweightGAIL(
                il_model=self.il_model,
                state_dim=1024,
                action_dim=100,
                lr=1e-4
            ).to(self.device)
            
            gail_model.train_gail(combined_trajectories, num_epochs=50)
            
            self.gail_models[scenario_type] = gail_model
            
            self.logger.info(f"✅ Completed IRL training for: {scenario_type}")
    
    def predict_with_irl(self, state: torch.Tensor, scenario_type: str = 'standard') -> torch.Tensor:
        """Predict next action using appropriate IRL model"""
        
        # For standard scenarios, use IL baseline
        if scenario_type == 'standard' or scenario_type not in self.gail_models:
            with torch.no_grad():
                return self.il_model.predict_next_action(state.unsqueeze(0).unsqueeze(0)).squeeze()
        
        # For trained scenarios, use GAIL-adjusted prediction
        gail_model = self.gail_models[scenario_type]
        return gail_model.get_adjusted_prediction(state)
    
    def evaluate_irl_vs_il(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Comprehensive evaluation of IRL vs IL"""
        
        self.logger.info("📊 Evaluating IRL vs IL across scenarios")
        
        results = {
            'by_scenario': defaultdict(lambda: {'il_scores': [], 'irl_scores': []}),
            'overall': {},
            'video_level': {}
        }
        
        for video in test_data:
            video_id = video['video_id']
            
            # Classify scenarios in test video
            scenarios = self.scenario_classifier.classify_video_scenarios(video)
            
            states = torch.tensor(video['frame_embeddings'], dtype=torch.float32).to(self.device)
            true_actions = torch.tensor(video['actions_binaries'], dtype=torch.float32)
            
            il_predictions = []
            irl_predictions = []
            
            # Get predictions for each frame
            for i in range(len(states)):
                state = states[i]
                
                # Determine scenario for this frame
                frame_scenario = 'standard'
                for scenario, frames in scenarios.items():
                    if i in frames and scenario != 'standard':
                        frame_scenario = scenario
                        break
                
                # IL prediction
                with torch.no_grad():
                    il_pred = self.il_model.predict_next_action(state.unsqueeze(0).unsqueeze(0))
                    il_predictions.append(il_pred.squeeze().cpu())
                
                # IRL prediction
                irl_pred = self.predict_with_irl(state, frame_scenario)
                irl_predictions.append(irl_pred.cpu())
            
            # Calculate performance by scenario
            il_preds_tensor = torch.stack(il_predictions)
            irl_preds_tensor = torch.stack(irl_predictions)
            
            for scenario, frame_indices in scenarios.items():
                if not frame_indices:
                    continue
                
                # Get predictions for this scenario
                scenario_il_preds = il_preds_tensor[frame_indices]
                scenario_irl_preds = irl_preds_tensor[frame_indices]
                scenario_true = true_actions[frame_indices]
                
                # Calculate mAP for this scenario
                il_score = self._calculate_map(scenario_il_preds, scenario_true)
                irl_score = self._calculate_map(scenario_irl_preds, scenario_true)
                
                results['by_scenario'][scenario]['il_scores'].append(il_score)
                results['by_scenario'][scenario]['irl_scores'].append(irl_score)
            
            # Overall video performance
            il_score_overall = self._calculate_map(il_preds_tensor, true_actions)
            irl_score_overall = self._calculate_map(irl_preds_tensor, true_actions)
            
            results['video_level'][video_id] = {
                'il_score': il_score_overall,
                'irl_score': irl_score_overall,
                'improvement': irl_score_overall - il_score_overall,
                'scenarios': scenarios
            }
        
        # Aggregate results
        for scenario, scores in results['by_scenario'].items():
            if scores['il_scores'] and scores['irl_scores']:
                scores['il_mean'] = np.mean(scores['il_scores'])
                scores['irl_mean'] = np.mean(scores['irl_scores'])
                scores['improvement'] = scores['irl_mean'] - scores['il_mean']
                scores['improvement_percentage'] = (scores['improvement'] / scores['il_mean']) * 100 if scores['il_mean'] > 0 else 0
        
        return results
    
    def _calculate_map(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean Average Precision"""
        try:
            from sklearn.metrics import average_precision_score
            
            pred_np = predictions.detach().numpy()
            target_np = targets.numpy()
            
            aps = []
            for i in range(target_np.shape[1]):
                if target_np[:, i].sum() > 0:
                    ap = average_precision_score(target_np[:, i], pred_np[:, i])
                    aps.append(ap)
            
            return np.mean(aps) if aps else 0.0
            
        except Exception:
            pred_binary = (predictions > 0.5).float()
            accuracy = (pred_binary == targets).float().mean()
            return accuracy.item()

# Integration function - NO STABLE-BASELINES NEEDED
def train_surgical_irl(config, train_data, test_data, logger, il_model):
    """
    Complete IRL training pipeline without stable-baselines dependencies
    
    Why we don't use stable-baselines:
    1. We have a working IL baseline - don't need full RL algorithms
    2. Our IRL is focused on specific scenarios, not general RL
    3. Easier to debug and control custom implementation
    4. Faster iteration for MICCAI paper timeline
    5. No dependency conflicts with your existing setup
    """
    
    logger.info("🎯 Training Surgical IRL (Custom Implementation - No Stable-Baselines)")
    
    # Initialize scenario-specific IRL
    irl_system = ScenarioSpecificIRL(
        il_model=il_model,
        config=config,
        logger=logger,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train IRL for challenging scenarios only
    challenging_scenarios = ['high_complexity', 'rare_actions', 'critical_moments', 'phase_transitions']
    
    irl_system.train_scenario_specific_irl(
        train_data=train_data,
        scenarios_to_train=challenging_scenarios
    )
    
    # Evaluate IRL vs IL
    evaluation_results = irl_system.evaluate_irl_vs_il(test_data)
    
    # Log results
    logger.info("🏆 IRL TRAINING RESULTS:")
    logger.info(f"Trained IRL for scenarios: {challenging_scenarios}")
    
    for scenario, results in evaluation_results['by_scenario'].items():
        if 'improvement' in results:
            logger.info(f"  {scenario}: IL={results['il_mean']:.4f}, IRL={results['irl_mean']:.4f}, "
                       f"Improvement={results['improvement']:.4f} ({results['improvement_percentage']:.1f}%)")
    
    return {
        'irl_system': irl_system,
        'evaluation_results': evaluation_results,
        'trained_scenarios': challenging_scenarios,
        'technique': 'MaxEnt IRL + Lightweight GAIL (Custom Implementation)'
    }

if __name__ == "__main__":
    print("🎯 SCENARIO-SPECIFIC IRL FOR SURGICAL ACTION PREDICTION")
    print("=" * 60)
    print("Technique: Maximum Entropy IRL + Lightweight GAIL")
    print("Implementation: Custom (No Stable-Baselines)")
    print("Focus: Scenario-specific improvements over IL baseline")
    print("✅ Ready for integration with existing codebase")