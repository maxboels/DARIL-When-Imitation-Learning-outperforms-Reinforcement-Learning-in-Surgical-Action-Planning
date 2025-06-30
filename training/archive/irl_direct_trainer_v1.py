#!/usr/bin/env python3
"""
Direct IRL Trainer - Using Existing IVT Labels
Simple approach: Learn rewards directly from your 100 IVT action labels
No scenario classification needed - analyze results post-hoc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random
from tqdm import tqdm

class DirectIRL(nn.Module):
    """
    Direct IRL: Learn rewards from your existing IVT labels
    No scenarios needed - single reward function for all contexts
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, device: str = 'cuda'):
        super(DirectIRL, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Single reward network for all surgical contexts
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(device)
        
        # Small policy adjustment network
        self.policy_adjustment = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Bounded adjustments [-1, 1]
        ).to(device)
        
        self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_adjustment.parameters(), lr=lr)
        
    def compute_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute learned reward for state-action pairs"""
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        # Concatenate state and action
        sa_pairs = torch.cat([states, actions], dim=-1)
        rewards = self.reward_net(sa_pairs)
        return rewards.squeeze(-1)
    
    def predict_with_irl(self, il_model, state: torch.Tensor) -> torch.Tensor:
        """Get IL prediction with small IRL adjustment"""
        with torch.no_grad():
            # Get IL baseline prediction
            il_input = state.unsqueeze(0).unsqueeze(0)
            if hasattr(il_model, 'device'):
                il_input = il_input.to(il_model.device)
            
            il_pred = il_model.predict_next_action(il_input).squeeze()
            il_pred = il_pred.to(self.device)
            
            # Apply small learned adjustment
            adjustment = self.policy_adjustment(il_pred)
            adjusted_pred = torch.sigmoid(il_pred + 0.05 * adjustment)  # Very small adjustment
            
            return adjusted_pred


class DirectIRLTrainer:
    """
    Direct IRL Trainer for your CholecT50 data
    Uses existing IVT labels directly - no scenario classification
    """
    
    def __init__(self, il_model, config, logger, device='cuda'):
        self.il_model = il_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Ensure IL model is on correct device
        if hasattr(il_model, 'to'):
            self.il_model = self.il_model.to(device)
        
        # Initialize direct IRL system
        self.irl_system = DirectIRL(
            state_dim=1024,  # Your embedding dimension
            action_dim=100,   # Your action classes
            lr=1e-4,
            device=device
        )
        
    def train_direct_irl(self, train_data: List[Dict], num_iterations: int = 100):
        """Train IRL directly on all expert demonstrations"""
        
        self.logger.info(f"🎯 Training Direct IRL on existing IVT labels ({num_iterations} iterations)")
        
        # Extract all expert state-action pairs from your data
        expert_states = []
        expert_actions = []
        
        for video in tqdm(train_data, desc="Extracting expert demonstrations"):
            frame_embeddings = video['frame_embeddings']
            actions_binaries = video['actions_binaries']
            
            # Convert to tensors if needed
            if isinstance(frame_embeddings, np.ndarray):
                states = torch.tensor(frame_embeddings, dtype=torch.float32)
            else:
                states = frame_embeddings
            
            if isinstance(actions_binaries, np.ndarray):
                actions = torch.tensor(actions_binaries, dtype=torch.float32)
            else:
                actions = actions_binaries
            
            # Only include frames with actions (skip empty frames)
            for i in range(len(states)):
                if torch.sum(actions[i]) > 0:  # Frame has at least one action
                    expert_states.append(states[i])
                    expert_actions.append(actions[i])
        
        if not expert_states:
            self.logger.error("❌ No expert demonstrations found!")
            return False
        
        expert_states = torch.stack(expert_states).to(self.device)
        expert_actions = torch.stack(expert_actions).to(self.device)
        
        self.logger.info(f"📊 Training on {len(expert_states)} expert state-action pairs")
        
        # Train reward function using MaxEnt IRL
        for iteration in range(num_iterations):
            # Generate negative examples (realistic but suboptimal)
            negative_actions = self._generate_realistic_negatives(expert_actions)
            
            # Compute rewards
            expert_rewards = self.irl_system.compute_reward(expert_states, expert_actions)
            negative_rewards = self.irl_system.compute_reward(expert_states, negative_actions)
            
            # MaxEnt IRL loss: expert actions should have higher reward
            reward_loss = -torch.mean(expert_rewards) + torch.mean(torch.exp(negative_rewards))
            
            # Regularization
            l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.irl_system.reward_net.parameters())
            total_reward_loss = reward_loss + l2_reg
            
            # Update reward network
            self.irl_system.reward_optimizer.zero_grad()
            total_reward_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.irl_system.reward_net.parameters(), 1.0)
            self.irl_system.reward_optimizer.step()
            
            if iteration % 20 == 0:
                self.logger.info(f"  Iteration {iteration}: Reward Loss = {total_reward_loss.item():.4f}, "
                               f"Expert Reward = {expert_rewards.mean().item():.3f}, "
                               f"Negative Reward = {negative_rewards.mean().item():.3f}")
        
        # Train policy adjustment (lightweight GAIL-style)
        self.logger.info("🎮 Training policy adjustment...")
        self._train_policy_adjustment(expert_states, expert_actions)
        
        self.logger.info("✅ Direct IRL training completed")
        return True
    
    def _generate_realistic_negatives(self, expert_actions: torch.Tensor) -> torch.Tensor:
        """Generate realistic but suboptimal action sequences"""
        batch_size = expert_actions.shape[0]
        
        # Strategy 1: Random sparse actions (similar to your 0-3 actions per frame)
        negative_actions = torch.zeros_like(expert_actions)
        
        for i in range(batch_size):
            # Random number of actions (0-3, matching your data distribution)
            num_actions = np.random.randint(0, 4)
            if num_actions > 0:
                # Random action indices
                action_indices = np.random.choice(100, num_actions, replace=False)
                negative_actions[i, action_indices] = 1.0
        
        return negative_actions.to(self.device)
    
    def _train_policy_adjustment(self, expert_states: torch.Tensor, expert_actions: torch.Tensor, num_epochs: int = 30):
        """Train small policy adjustments to IL predictions"""
        
        batch_size = 32
        num_samples = len(expert_states)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_states = expert_states[i:end_idx]
                batch_expert_actions = expert_actions[i:end_idx]
                
                # Get IL predictions
                with torch.no_grad():
                    il_predictions = []
                    for state in batch_states:
                        il_input = state.unsqueeze(0).unsqueeze(0)
                        if hasattr(self.il_model, 'device'):
                            il_input = il_input.to(self.il_model.device)
                        
                        il_pred = self.il_model.predict_next_action(il_input).squeeze()
                        il_predictions.append(il_pred.to(self.device))
                    
                    il_predictions = torch.stack(il_predictions)
                
                # Apply policy adjustment
                adjustments = self.irl_system.policy_adjustment(il_predictions)
                adjusted_predictions = torch.sigmoid(il_predictions + 0.05 * adjustments)
                
                # Loss: adjusted predictions should have higher reward than IL alone
                adjusted_rewards = self.irl_system.compute_reward(batch_states, adjusted_predictions)
                il_rewards = self.irl_system.compute_reward(batch_states, il_predictions)
                
                # Policy loss: maximize adjusted rewards
                policy_loss = -torch.mean(adjusted_rewards) + 0.5 * F.mse_loss(adjusted_predictions, batch_expert_actions)
                
                self.irl_system.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.irl_system.policy_optimizer.step()
                
                epoch_loss += policy_loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"  Policy Epoch {epoch}: Loss = {epoch_loss/num_batches:.4f}")
    
    def evaluate_direct_irl(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate IL vs IRL and analyze by existing labels"""
        
        self.logger.info("📊 Evaluating Direct IRL vs IL")
        
        results = {
            'overall': {},
            'by_action_type': defaultdict(lambda: {'il_scores': [], 'irl_scores': []}),
            'by_phase': defaultdict(lambda: {'il_scores': [], 'irl_scores': []}),
            'by_complexity': defaultdict(lambda: {'il_scores': [], 'irl_scores': []}),
            'video_level': {}
        }
        
        for video in test_data:
            video_id = video['video_id']
            
            frame_embeddings = video['frame_embeddings']
            actions_binaries = video['actions_binaries']
            phases = video.get('phase_binaries', [])
            
            # Convert to tensors
            if isinstance(frame_embeddings, np.ndarray):
                states = torch.tensor(frame_embeddings, dtype=torch.float32)
            else:
                states = frame_embeddings
            
            if isinstance(actions_binaries, np.ndarray):
                true_actions = torch.tensor(actions_binaries, dtype=torch.float32)
            else:
                true_actions = actions_binaries
            
            il_predictions = []
            irl_predictions = []
            
            # Get predictions for each frame
            for i in range(len(states)):
                state = states[i]
                
                # IL prediction
                with torch.no_grad():
                    il_pred = self.il_model.predict_next_action(state.unsqueeze(0).unsqueeze(0))
                    il_predictions.append(il_pred.squeeze().cpu())
                
                # IRL prediction  
                irl_pred = self.irl_system.predict_with_irl(self.il_model, state)
                irl_predictions.append(irl_pred.cpu())
            
            # Calculate overall performance
            il_preds_tensor = torch.stack(il_predictions)
            irl_preds_tensor = torch.stack(irl_predictions)
            
            il_score_overall = self._calculate_map(il_preds_tensor, true_actions.cpu())
            irl_score_overall = self._calculate_map(irl_preds_tensor, true_actions.cpu())
            
            results['video_level'][video_id] = {
                'il_score': il_score_overall,
                'irl_score': irl_score_overall,
                'improvement': irl_score_overall - il_score_overall
            }
            
            # Analyze by existing labels (post-hoc analysis)
            for i in range(len(true_actions)):
                frame_actions = true_actions[i]
                active_actions = torch.where(frame_actions > 0.5)[0]
                
                # By action type (using your existing IVT labels)
                for action_id in active_actions:
                    action_id = action_id.item()
                    results['by_action_type'][action_id]['il_scores'].append(
                        self._calculate_map(il_preds_tensor[i:i+1], frame_actions.unsqueeze(0))
                    )
                    results['by_action_type'][action_id]['irl_scores'].append(
                        self._calculate_map(irl_preds_tensor[i:i+1], frame_actions.unsqueeze(0))
                    )
                
                # By phase (using your existing phase labels)
                if len(phases) > i:
                    current_phase = torch.argmax(torch.tensor(phases[i])).item()
                    results['by_phase'][current_phase]['il_scores'].append(
                        self._calculate_map(il_preds_tensor[i:i+1], frame_actions.unsqueeze(0))
                    )
                    results['by_phase'][current_phase]['irl_scores'].append(
                        self._calculate_map(irl_preds_tensor[i:i+1], frame_actions.unsqueeze(0))
                    )
                
                # By complexity (using natural sparsity)
                num_actions = len(active_actions)
                complexity = 'simple' if num_actions <= 1 else 'complex'
                results['by_complexity'][complexity]['il_scores'].append(
                    self._calculate_map(il_preds_tensor[i:i+1], frame_actions.unsqueeze(0))
                )
                results['by_complexity'][complexity]['irl_scores'].append(
                    self._calculate_map(irl_preds_tensor[i:i+1], frame_actions.unsqueeze(0))
                )
        
        # Aggregate results
        all_il_scores = [v['il_score'] for v in results['video_level'].values()]
        all_irl_scores = [v['irl_score'] for v in results['video_level'].values()]
        
        results['overall'] = {
            'il_mean': np.mean(all_il_scores),
            'irl_mean': np.mean(all_irl_scores),
            'improvement': np.mean(all_irl_scores) - np.mean(all_il_scores),
            'improvement_percentage': ((np.mean(all_irl_scores) - np.mean(all_il_scores)) / np.mean(all_il_scores)) * 100 if np.mean(all_il_scores) > 0 else 0
        }
        
        # Analyze which action types/phases benefited most
        improvements = {}
        
        # Action type improvements
        for action_id, scores in results['by_action_type'].items():
            if len(scores['il_scores']) > 5:  # Sufficient data
                il_mean = np.mean(scores['il_scores'])
                irl_mean = np.mean(scores['irl_scores'])
                improvements[f'action_{action_id}'] = irl_mean - il_mean
        
        # Phase improvements
        for phase_id, scores in results['by_phase'].items():
            if len(scores['il_scores']) > 5:
                il_mean = np.mean(scores['il_scores'])
                irl_mean = np.mean(scores['irl_scores'])
                improvements[f'phase_{phase_id}'] = irl_mean - il_mean
        
        # Find top improvements
        top_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:10]
        
        self.logger.info("🏆 TOP 10 IMPROVEMENTS (Direct IRL vs IL):")
        for label, improvement in top_improvements:
            self.logger.info(f"  {label}: +{improvement:.4f} mAP improvement")
        
        results['top_improvements'] = top_improvements
        
        return results
    
    def _calculate_map(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean Average Precision"""
        try:
            from sklearn.metrics import average_precision_score
            
            pred_np = predictions.detach().cpu().numpy()
            target_np = targets.cpu().numpy() if hasattr(targets, 'cpu') else targets.numpy()
            
            # Handle single frame case
            if pred_np.ndim == 1:
                pred_np = pred_np.reshape(1, -1)
            if target_np.ndim == 1:
                target_np = target_np.reshape(1, -1)
            
            aps = []
            for i in range(target_np.shape[1]):
                if target_np[:, i].sum() > 0:
                    ap = average_precision_score(target_np[:, i], pred_np[:, i])
                    aps.append(ap)
            
            return np.mean(aps) if aps else 0.0
            
        except Exception as e:
            self.logger.warning(f"mAP calculation failed ({e}), using accuracy")
            # Fallback to simple accuracy
            pred_binary = (predictions > 0.5).float()
            accuracy = (pred_binary == targets).float().mean()
            return accuracy.item()


# Integration function for your existing codebase
def train_direct_surgical_irl(config, train_data, test_data, logger, il_model):
    """
    Main function to integrate with your existing experiment runner
    Replaces the scenario-based IRL approach
    """
    
    logger.info("🎯 Training Direct Surgical IRL (Simplified Approach)")
    
    # Get device from config
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Initialize direct IRL trainer
    irl_trainer = DirectIRLTrainer(
        il_model=il_model,
        config=config,
        logger=logger,
        device=device
    )
    
    # Train direct IRL
    irl_config = config.get('experiment', {}).get('irl_enhancement', {})
    num_iterations = irl_config.get('maxent_irl', {}).get('num_iterations', 100)
    
    success = irl_trainer.train_direct_irl(train_data, num_iterations=num_iterations)
    
    if not success:
        return {
            'status': 'failed',
            'error': 'Direct IRL training failed'
        }
    
    # Evaluate IL vs IRL
    evaluation_results = irl_trainer.evaluate_direct_irl(test_data)
    
    logger.info("🏆 DIRECT IRL TRAINING RESULTS:")
    logger.info(f"Overall IL mAP: {evaluation_results['overall']['il_mean']:.4f}")
    logger.info(f"Overall IRL mAP: {evaluation_results['overall']['irl_mean']:.4f}")
    logger.info(f"Overall Improvement: {evaluation_results['overall']['improvement']:.4f} ({evaluation_results['overall']['improvement_percentage']:.1f}%)")
    
    return {
        'irl_trainer': irl_trainer,
        'evaluation_results': evaluation_results,
        'technique': 'Direct IRL (No Scenarios)',
        'device': device,
        'approach': 'Single reward function learned from all IVT demonstrations'
    }


if __name__ == "__main__":
    print("🎯 DIRECT IRL FOR SURGICAL ACTION PREDICTION")
    print("=" * 60)
    print("✅ Simple approach: Learn from existing IVT labels directly")
    print("✅ No scenario classification needed")
    print("✅ Post-hoc analysis using existing action/phase categories")
    print("✅ Ready for integration with existing codebase")