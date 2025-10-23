#!/usr/bin/env python3
"""
Surgical Inverse Reinforcement Learning for Next Action Prediction
Building on existing IL infrastructure to show RL advantages in specific scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random
from tqdm import tqdm

class SurgicalScenarioClassifier:
    """Classify surgical scenarios where RL should outperform IL"""
    
    def __init__(self, config):
        self.config = config
        
    def classify_video_scenarios(self, video_data: Dict) -> Dict[str, List[int]]:
        """Classify frames in a video by scenario type"""
        
        embeddings = video_data['frame_embeddings']
        actions = video_data['actions_binaries']
        phases = video_data.get('phase_binaries', [])
        
        scenarios = {
            'high_complexity': [],
            'phase_transitions': [],
            'rare_actions': [],
            'high_uncertainty': [],
            'critical_moments': [],
            'standard': []
        }
        
        num_frames = len(embeddings)
        
        for i in tqdm(range(num_frames), desc=f"Classifying scenarios for video {video_data['video_id']}"):
            frame_scenarios = []
            
            # 1. High Complexity: Multiple simultaneous actions
            active_actions = np.sum(actions[i] > 0.5)
            if active_actions >= 3:
                frame_scenarios.append('high_complexity')
            
            # 2. Phase Transitions: Change in surgical phase
            if len(phases) > i + 1:
                current_phase = np.argmax(phases[i])
                next_phase = np.argmax(phases[i + 1])
                if current_phase != next_phase:
                    frame_scenarios.append('phase_transitions')
            
            # 3. Rare Actions: Uncommon action combinations
            action_pattern = tuple(np.where(actions[i] > 0.5)[0])
            if self._is_rare_action_pattern(action_pattern):
                frame_scenarios.append('rare_actions')
            
            # 4. High Uncertainty: Based on embedding variance
            if i > 0:
                frame_diff = np.linalg.norm(embeddings[i] - embeddings[i-1])
                if frame_diff > np.percentile([np.linalg.norm(embeddings[j] - embeddings[j-1]) 
                                             for j in range(1, num_frames)], 85):
                    frame_scenarios.append('high_uncertainty')
            
            # 5. Critical Moments: Near important structures (heuristic)
            if self._is_critical_moment(embeddings[i], actions[i]):
                frame_scenarios.append('critical_moments')
            
            # Add to appropriate scenario lists
            if frame_scenarios:
                for scenario in frame_scenarios:
                    scenarios[scenario].append(i)
            else:
                scenarios['standard'].append(i)
        
        return scenarios
    
    def _is_rare_action_pattern(self, action_pattern: Tuple) -> bool:
        """Heuristic to identify rare action combinations"""
        # Simple heuristic: patterns with >2 actions or specific combinations
        if len(action_pattern) > 2:
            return True
        # Add domain-specific rare patterns here
        return False
    
    def _is_critical_moment(self, embedding: np.ndarray, actions: np.ndarray) -> bool:
        """Heuristic to identify critical surgical moments"""
        # Simple heuristic: high action intensity + cutting actions
        cutting_actions = [2, 3, 5]  # Example: scissors, cautery, hook
        has_cutting = any(actions[i] > 0.5 for i in cutting_actions if i < len(actions))
        high_intensity = np.sum(actions > 0.5) >= 2
        return has_cutting and high_intensity

class SurgicalRewardLearner:
    """Learn reward functions from expert demonstrations"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Reward networks for different scenarios
        self.reward_networks = nn.ModuleDict({
            'standard': self._build_reward_network(),
            'high_complexity': self._build_reward_network(),
            'phase_transitions': self._build_reward_network(),
            'rare_actions': self._build_reward_network(),
            'critical_moments': self._build_reward_network()
        })
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
    def _build_reward_network(self) -> nn.Module:
        """Build a reward network for a specific scenario"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim // 2 + 100, self.hidden_dim),  # state + action features
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def extract_features(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Extract features from state-action pairs"""
        state_features = self.feature_extractor(state)
        combined_features = torch.cat([state_features, action], dim=-1)
        return combined_features
    
    def compute_reward(self, state: torch.Tensor, action: torch.Tensor, 
                      scenario: str = 'standard') -> torch.Tensor:
        """Compute reward for state-action pair given scenario"""
        features = self.extract_features(state, action)
        reward_net = self.reward_networks.get(scenario, self.reward_networks['standard'])
        return reward_net(features)
    
    def learn_from_expert_trajectories(self, expert_trajectories: List[Dict], 
                                     scenario_classifications: Dict[str, List]):
        """Learn reward functions from expert demonstrations using IRL"""
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        for epoch in range(100):  # IRL training epochs
            total_loss = 0.0
            
            for scenario_type, frame_indices in scenario_classifications.items():
                if not frame_indices:
                    continue
                
                scenario_loss = 0.0
                num_samples = 0
                
                for traj in expert_trajectories:
                    states = torch.tensor(traj['states'], dtype=torch.float32)
                    actions = torch.tensor(traj['actions'], dtype=torch.float32)
                    
                    # Sample from this scenario
                    scenario_frames = [i for i in frame_indices if i < len(states)]
                    if not scenario_frames:
                        continue
                    
                    # Expert demonstrations (positive examples)
                    expert_indices = random.sample(scenario_frames, 
                                                 min(10, len(scenario_frames)))
                    expert_rewards = []
                    
                    for idx in expert_indices:
                        reward = self.compute_reward(states[idx], actions[idx], scenario_type)
                        expert_rewards.append(reward)
                    
                    # Generate negative examples (random actions)
                    negative_rewards = []
                    for idx in expert_indices:
                        random_action = torch.rand_like(actions[idx])
                        random_action = (random_action > 0.7).float()  # Sparse random actions
                        reward = self.compute_reward(states[idx], random_action, scenario_type)
                        negative_rewards.append(reward)
                    
                    # IRL loss: expert actions should have higher reward
                    if expert_rewards and negative_rewards:
                        expert_reward_mean = torch.stack(expert_rewards).mean()
                        negative_reward_mean = torch.stack(negative_rewards).mean()
                        
                        # Margin loss: expert rewards should be higher by margin
                        margin_loss = F.relu(1.0 - (expert_reward_mean - negative_reward_mean))
                        scenario_loss += margin_loss
                        num_samples += 1
                
                if num_samples > 0:
                    scenario_loss /= num_samples
                    total_loss += scenario_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"IRL Epoch {epoch}: Loss = {total_loss.item():.4f}")

class ScenarioAwarePolicy:
    """Policy that combines IL and RL based on scenario"""
    
    def __init__(self, il_model, reward_learner: SurgicalRewardLearner, 
                 scenario_classifier: SurgicalScenarioClassifier):
        self.il_model = il_model
        self.reward_learner = reward_learner
        self.scenario_classifier = scenario_classifier
        
        # Policy networks for different scenarios (small networks that modify IL output)
        self.policy_adjustments = nn.ModuleDict({
            'high_complexity': self._build_adjustment_network(),
            'phase_transitions': self._build_adjustment_network(),
            'rare_actions': self._build_adjustment_network(),
            'critical_moments': self._build_adjustment_network()
        })
        
    def _build_adjustment_network(self) -> nn.Module:
        """Build small network to adjust IL predictions"""
        return nn.Sequential(
            nn.Linear(100, 64),  # num_action_classes -> hidden
            nn.ReLU(),
            nn.Linear(64, 100),  # hidden -> num_action_classes
            nn.Tanh()  # Output adjustments in [-1, 1]
        )
    
    def predict_next_action(self, state: torch.Tensor, 
                          scenario_type: str = 'standard') -> torch.Tensor:
        """Predict next action using IL + RL adjustment"""
        
        # Get IL baseline prediction
        with torch.no_grad():
            il_prediction = self.il_model.predict_next_action(state.unsqueeze(0))
            il_prediction = il_prediction.squeeze(0)
        
        # For standard scenarios, use IL as-is
        if scenario_type == 'standard':
            return il_prediction
        
        # For complex scenarios, apply RL adjustment
        if scenario_type in self.policy_adjustments:
            adjustment = self.policy_adjustments[scenario_type](il_prediction)
            # Apply small adjustment to IL prediction
            adjusted_prediction = il_prediction + 0.1 * adjustment
            adjusted_prediction = torch.sigmoid(adjusted_prediction)
            return adjusted_prediction
        
        return il_prediction
    
    def train_rl_adjustments(self, training_data: List[Dict], 
                           scenario_classifications: Dict[str, List]):
        """Train RL adjustments using learned rewards"""
        
        optimizers = {
            scenario: torch.optim.Adam(network.parameters(), lr=1e-4)
            for scenario, network in self.policy_adjustments.items()
        }
        # RL training epochs
        for epoch in tqdm(range(50), desc="Training RL Adjustments"): 
            total_loss = 0.0
            
            for scenario_type, frame_indices in scenario_classifications.items():
                if scenario_type not in self.policy_adjustments or not frame_indices:
                    continue
                
                optimizer = optimizers[scenario_type]
                scenario_loss = 0.0
                num_samples = 0
                
                for traj in training_data:
                    states = torch.tensor(traj['states'], dtype=torch.float32)
                    actions = torch.tensor(traj['actions'], dtype=torch.float32)
                    
                    scenario_frames = [i for i in frame_indices if i < len(states)]
                    if not scenario_frames:
                        continue
                    
                    # Sample frames from this scenario
                    sample_indices = random.sample(scenario_frames, 
                                                 min(5, len(scenario_frames)))
                    
                    for idx in sample_indices:
                        state = states[idx]
                        expert_action = actions[idx]
                        
                        # Get policy prediction
                        predicted_action = self.predict_next_action(state, scenario_type)
                        
                        # Compute reward for predicted action
                        predicted_reward = self.reward_learner.compute_reward(
                            state, predicted_action, scenario_type
                        )
                        
                        # Compute reward for expert action  
                        expert_reward = self.reward_learner.compute_reward(
                            state, expert_action, scenario_type
                        )
                        
                        # RL loss: maximize reward
                        rl_loss = -predicted_reward.mean()
                        
                        # Behavioral cloning loss: stay close to expert
                        bc_loss = F.binary_cross_entropy(predicted_action, expert_action)
                        
                        # Combined loss
                        combined_loss = rl_loss + 0.5 * bc_loss
                        scenario_loss += combined_loss
                        num_samples += 1
                
                if num_samples > 0:
                    scenario_loss /= num_samples
                    
                    optimizer.zero_grad()
                    scenario_loss.backward()
                    optimizer.step()
                    
                    total_loss += scenario_loss.item()
            
            if epoch % 10 == 0:
                print(f"RL Training Epoch {epoch}: Loss = {total_loss:.4f}")

class SurgicalIRLTrainer:
    """Main trainer for surgical IRL approach"""
    
    def __init__(self, il_model, config, logger, device='cuda'):
        self.il_model = il_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Initialize components
        self.scenario_classifier = SurgicalScenarioClassifier(config)
        self.reward_learner = SurgicalRewardLearner(
            feature_dim=1024,  # Your embedding dimension
            hidden_dim=256
        ).to(device)
        
        self.scenario_policy = None  # Will be initialized after reward learning
        
    def train_irl_system(self, train_data: List[Dict], test_data: List[Dict]):
        """Complete IRL training pipeline"""
        
        self.logger.info("🎯 Starting Surgical IRL Training")
        
        # Step 1: Classify scenarios in all videos
        self.logger.info("📊 Step 1: Classifying scenarios across videos")
        all_scenario_classifications = {}
        
        for video in tqdm(train_data, desc="Classifying scenarios"):
            video_id = video['video_id']
            self.logger.info(f"Classifying scenarios for video {video_id}")
            scenarios = self.scenario_classifier.classify_video_scenarios(video)
            all_scenario_classifications[video_id] = scenarios
            
            # Log scenario distribution
            total_frames = sum(len(frames) for frames in scenarios.values())
            self.logger.info(f"Video {video_id}: {total_frames} frames classified")
            for scenario, frames in scenarios.items():
                if frames:
                    self.logger.info(f"  {scenario}: {len(frames)} frames ({len(frames)/total_frames*100:.1f}%)")
        
        # Step 2: Learn rewards from expert demonstrations
        self.logger.info("🏆 Step 2: Learning rewards from expert demonstrations")
        
        # Prepare expert trajectories
        expert_trajectories = []
        combined_scenario_classifications = defaultdict(list)
        
        for video in train_data:
            trajectory = {
                'states': video['frame_embeddings'],
                'actions': video['actions_binaries'],
                'video_id': video['video_id']
            }
            expert_trajectories.append(trajectory)
            
            # Combine scenario classifications
            video_scenarios = all_scenario_classifications[video['video_id']]
            for scenario, frames in video_scenarios.items():
                combined_scenario_classifications[scenario].extend(frames)
        
        # Train reward functions
        self.reward_learner.learn_from_expert_trajectories(
            expert_trajectories, combined_scenario_classifications
        )
        
        # Step 3: Train scenario-aware policy
        self.logger.info("🎮 Step 3: Training scenario-aware policy")
        
        self.scenario_policy = ScenarioAwarePolicy(
            self.il_model, self.reward_learner, self.scenario_classifier
        ).to(self.device)
        
        self.scenario_policy.train_rl_adjustments(
            expert_trajectories, combined_scenario_classifications
        )
        
        self.logger.info("✅ IRL training completed")
        
        return {
            'scenario_classifications': all_scenario_classifications,
            'reward_learner': self.reward_learner,
            'scenario_policy': self.scenario_policy
        }
    
    def evaluate_irl_vs_il(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate IRL vs IL performance by scenario"""
        
        results = {
            'by_scenario': defaultdict(lambda: {'il_scores': [], 'irl_scores': []}),
            'overall': {'il_score': 0, 'irl_score': 0},
            'video_level': {}
        }
        
        for video in tqdm(test_data, desc="Evaluating videos"):
            video_id = video['video_id']
            self.logger.info(f"Evaluating video {video_id}")
            
            # Classify scenarios in test video
            scenarios = self.scenario_classifier.classify_video_scenarios(video)
            
            video_results = {'scenarios': scenarios}
            
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
                irl_pred = self.scenario_policy.predict_next_action(state, frame_scenario)
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
            
            video_results.update({
                'il_score': il_score_overall,
                'irl_score': irl_score_overall,
                'improvement': irl_score_overall - il_score_overall
            })
            
            results['video_level'][video_id] = video_results
        
        # Aggregate results
        for scenario, scores in results['by_scenario'].items():
            if scores['il_scores'] and scores['irl_scores']:
                scores['il_mean'] = np.mean(scores['il_scores'])
                scores['irl_mean'] = np.mean(scores['irl_scores'])
                scores['improvement'] = scores['irl_mean'] - scores['il_mean']
                scores['improvement_percentage'] = (scores['improvement'] / scores['il_mean']) * 100
        
        # Overall results
        all_il_scores = [v['il_score'] for v in results['video_level'].values()]
        all_irl_scores = [v['irl_score'] for v in results['video_level'].values()]
        
        results['overall'] = {
            'il_mean': np.mean(all_il_scores),
            'irl_mean': np.mean(all_irl_scores),
            'improvement': np.mean(all_irl_scores) - np.mean(all_il_scores),
            'improvement_percentage': ((np.mean(all_irl_scores) - np.mean(all_il_scores)) / np.mean(all_il_scores)) * 100
        }
        
        return results
    
    def _calculate_map(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean Average Precision"""
        try:
            from sklearn.metrics import average_precision_score
            
            # Convert to numpy
            pred_np = predictions.detach().numpy()
            target_np = targets.numpy()
            
            # Calculate mAP across all classes
            aps = []
            for i in range(target_np.shape[1]):
                if target_np[:, i].sum() > 0:  # Only calculate for classes that appear
                    ap = average_precision_score(target_np[:, i], pred_np[:, i])
                    aps.append(ap)
            
            return np.mean(aps) if aps else 0.0
            
        except Exception:
            # Fallback to simple accuracy
            pred_binary = (predictions > 0.5).float()
            accuracy = (pred_binary == targets).float().mean()
            return accuracy.item()

# Integration with existing codebase
def integrate_irl_with_existing_trainer(autoregressive_trainer, config, logger):
    """Integrate IRL with your existing autoregressive trainer"""
    
    # Create IRL trainer
    irl_trainer = SurgicalIRLTrainer(
        il_model=autoregressive_trainer.model,
        config=config,
        logger=logger,
        device=autoregressive_trainer.device
    )
    
    return irl_trainer

# Example usage in your main training loop
def enhanced_training_with_irl(config, train_data, test_data, logger):
    """Enhanced training that combines IL + IRL"""
    
    # Step 1: Train IL baseline (your existing code)
    from training.autoregressive_il_trainer import AutoregressiveILTrainer
    from models.autoregressive_il_model import AutoregressiveILModel
    
    # Initialize IL model
    il_model = AutoregressiveILModel(**config['models']['autoregressive_il'])
    
    # Train IL baseline
    il_trainer = AutoregressiveILTrainer(il_model, config, logger)
    
    # Create your dataloaders
    from datasets.autoregressive_dataset import create_autoregressive_dataloaders
    train_loader, test_loaders = create_autoregressive_dataloaders(
        config['data'], train_data, test_data, 
        batch_size=config['training']['batch_size']
    )
    
    # Train IL baseline
    logger.info("🎓 Phase 1: Training IL baseline")
    best_il_path = il_trainer.train(train_loader, test_loaders)
    
    # Load best IL model
    il_model = AutoregressiveILModel.load_model(best_il_path)
    
    # Step 2: Train IRL enhancement
    logger.info("🎯 Phase 2: Training IRL enhancement")
    irl_trainer = SurgicalIRLTrainer(il_model, config, logger)
    
    irl_results = irl_trainer.train_irl_system(train_data, test_data)
    
    # Step 3: Evaluate IL vs IRL
    logger.info("📊 Phase 3: Evaluating IL vs IRL")
    evaluation_results = irl_trainer.evaluate_irl_vs_il(test_data)
    
    # Log results
    logger.info("🏆 RESULTS SUMMARY:")
    logger.info(f"Overall IL mAP: {evaluation_results['overall']['il_mean']:.4f}")
    logger.info(f"Overall IRL mAP: {evaluation_results['overall']['irl_mean']:.4f}")
    logger.info(f"Overall Improvement: {evaluation_results['overall']['improvement']:.4f} ({evaluation_results['overall']['improvement_percentage']:.1f}%)")
    
    logger.info("📊 By Scenario:")
    for scenario, results in evaluation_results['by_scenario'].items():
        if 'improvement' in results:
            logger.info(f"  {scenario}: IL={results['il_mean']:.4f}, IRL={results['irl_mean']:.4f}, "
                       f"Improvement={results['improvement']:.4f} ({results['improvement_percentage']:.1f}%)")
    
    return {
        'il_model': il_model,
        'irl_trainer': irl_trainer,
        'evaluation_results': evaluation_results,
        'irl_components': irl_results
    }