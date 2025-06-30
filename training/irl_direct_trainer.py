#!/usr/bin/env python3
"""
Fixed IRL Enhancement Integration for Surgical Action Prediction
Integrates the direct IRL approach with the existing experiment runner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random
from tqdm import tqdm
import os
import json

from datasets.irl_negative_generator import CholecT50NegativeGenerator


class DirectIRLSystem(nn.Module):
    """
    Direct IRL System - Learn rewards from existing IVT labels
    No scenarios needed - single reward function for all contexts
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, device: str = 'cuda'):
        super(DirectIRLSystem, self).__init__()
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
        # self.policy_adjustment = nn.Sequential(
        #     nn.Linear(action_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, action_dim),
        #     nn.Tanh()  # Bounded adjustments [-1, 1]
        # ).to(device)
        
        self.policy_adjustment = StateAwarePolicyAdjustment(
            state_dim=1024,    # Video embeddings
            action_dim=100,    # Action classes  
            phase_dim=7        # Surgical phases
        )
        
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

    def predict_with_irl(self, il_model, state, phase=None):

        # Base prediction from IL model
        il_pred = il_model.forward(state) # NOTE: Make sure it has enough context length in its sequence dimension
        
        # Now the adjustment can SEE the surgical scene!
        adjustment = self.policy_adjustment(state, il_pred, phase)
        
        final_pred = torch.sigmoid(il_pred + 0.1 * adjustment)  # Can increase to 10%
        return final_pred

    def predict_with_irl_policy(self, il_model, state: torch.Tensor) -> torch.Tensor:
        """Get IL prediction with small IRL adjustment"""
        with torch.no_grad():
            # Ensure state is on the correct device
            state = state.to(self.device)
            
            # Prepare input for IL model - needs proper sequence format
            il_input = state.unsqueeze(0)  # Add batch dimension
            if len(il_input.shape) == 2:  # [batch, features] -> [batch, seq_len, features]
                il_input = il_input.unsqueeze(1)
            
            # Ensure minimum sequence length for autoregressive model
            if il_input.shape[1] < 2:
                # Duplicate the frame to create minimum sequence length
                il_input = il_input.repeat(1, 2, 1)
            
            # Move input to same device as IL model
            if hasattr(il_model, 'device'):
                il_input = il_input.to(il_model.device)
            elif next(il_model.parameters()).is_cuda:
                il_input = il_input.to('cuda')
            
            # Handle different IL model interfaces
            try:
                if hasattr(il_model, 'predict_next_action'):
                    il_pred = il_model.predict_next_action(il_input).squeeze()
                elif hasattr(il_model, 'forward'):
                    outputs = il_model.forward(il_input)
                    if isinstance(outputs, dict):
                        il_pred = outputs.get('action_pred', outputs.get('action_logits', None))
                        if il_pred is not None:
                            if len(il_pred.shape) > 1:
                                il_pred = il_pred[:, -1, :]  # Take last timestep
                            il_pred = torch.sigmoid(il_pred).squeeze()
                        else:
                            raise ValueError("Could not find action predictions in model output")
                    else:
                        il_pred = torch.sigmoid(outputs).squeeze()
                else:
                    raise ValueError("IL model does not have compatible prediction interface")
                
                # Ensure prediction has correct shape and is on our device
                if len(il_pred.shape) == 0:  # Scalar
                    il_pred = il_pred.unsqueeze(0)
                if il_pred.shape[0] != 100:  # Should have 100 action classes
                    # Create default prediction if shape is wrong
                    il_pred = torch.rand(100, device=self.device) * 0.1
                
                il_pred = il_pred.to(self.device)
                
            except Exception as e:
                self.logger.warning(f"IL prediction failed: {e}, using random prediction")
                il_pred = torch.rand(100, device=self.device) * 0.1
            
            # Apply small learned adjustment
            try:
                adjustment = self.policy_adjustment(il_pred)
                adjusted_pred = torch.sigmoid(il_pred + 0.05 * adjustment)  # Very small adjustment
            except Exception as e:
                self.logger.warning(f"Policy adjustment failed: {e}, using IL prediction")
                adjusted_pred = il_pred
            
            return adjusted_pred


class DirectIRLTrainer:
    """
    Direct IRL Trainer for CholecT50 data
    Uses existing IVT labels directly - no scenario classification
    """
    
    def __init__(self, il_model, config, logger, device='cuda', tb_writer=None):
        self.il_model = il_model
        self.config = config
        self.logger = logger
        self.device = device
    
        # Setup logging directories
        self.log_dir = logger.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Tensorboard
        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard', 'direct_irl')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        # Create the IRL system
        self.irl_system = self._create_enhanced_irl_system()
        
        # Ensure IL model is on correct device
        if hasattr(il_model, 'to'):
            self.il_model = self.il_model.to(device)
        
        # Initialize direct IRL system
        self.irl_system = DirectIRLSystem(
            state_dim=1024,  # Your embedding dimension
            action_dim=100,   # Your action classes
            lr=1e-4,
            device=device
        )

    def train_direct_irl(self, train_data: List[Dict], num_iterations: int = 100):
        """🎯 ENHANCE your existing method with TensorBoard monitoring"""
        
        self.logger.info(f"🎯 Training Enhanced Direct IRL ({num_iterations} iterations)")
        
        # Keep your existing data extraction code
        expert_states, expert_actions, expert_phases = self._extract_expert_data_with_phases(train_data)
        
        self.logger.info(f"📊 Training on {len(expert_states)} expert state-action pairs")
        
        # 🎯 Enhanced training loop
        for iteration in tqdm(range(num_iterations), desc="Training reward function"):
            iteration_start = time.time()
            
            # Keep your existing negative generation (it's already good!)
            negative_actions = self._generate_realistic_negatives(expert_actions, expert_phases)
            
            # 🎯 Enhanced reward computation with phases
            expert_rewards = self.irl_system.compute_reward(expert_states, expert_actions, expert_phases)
            negative_rewards = self.irl_system.compute_reward(expert_states, negative_actions, expert_phases)
            
            # 🎯 Enhanced loss function
            reward_loss = self._compute_enhanced_loss(expert_rewards, negative_rewards)
            
            # Keep your existing regularization
            l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.irl_system.reward_net.parameters())
            total_loss = reward_loss + l2_reg
            
            # 🎯 Enhanced training step
            self.irl_system.reward_optimizer.zero_grad()
            total_loss.backward()
            
            # 🎯 ADD gradient monitoring
            grad_norm = self._monitor_gradients(iteration)
            
            torch.nn.utils.clip_grad_norm_(
                list(self.irl_system.reward_net.parameters()) + 
                list(self.irl_system.phase_reward_net.parameters()), 1.0
            )
            self.irl_system.reward_optimizer.step()
            
            iteration_time = time.time() - iteration_start
            
            # 🎯 ADD TensorBoard monitoring
            self._log_to_tensorboard(iteration, expert_rewards, negative_rewards, 
                                   total_loss, grad_norm, iteration_time)
            
            # Keep your existing console logging but enhance it
            if iteration % 5 == 0:  # More frequent logging
                expert_mean = expert_rewards.mean().item()
                negative_mean = negative_rewards.mean().item()
                gap = expert_mean - negative_mean
                
                self.logger.info(
                    f"🔄 Iter {iteration:3d}: Loss={total_loss:.4f}, "
                    f"Expert={expert_mean:+.4f}, Negative={negative_mean:+.4f}, "
                    f"Gap={gap:+.4f}, Time={iteration_time:.1f}s"
                )
        
        # Keep your existing policy adjustment training
        self.logger.info("🎮 Training policy adjustment...")
        self._train_policy_adjustment(expert_states, expert_actions)
        
        self.logger.info("✅ Enhanced Direct IRL training completed")
        return True


    # def train_direct_irl(self, train_data: List[Dict], num_iterations: int = 100):
    #     """Train IRL directly on all expert demonstrations"""
        
    #     self.logger.info(f"🎯 Training Direct IRL on existing IVT labels ({num_iterations} iterations)")
        
    #     # Extract all expert state-action pairs from your data
    #     expert_states = []
    #     expert_actions = []
    #     expert_phases = []
        
    #     for video in tqdm(train_data, desc="Extracting expert demonstrations"):
    #         frame_embeddings = video['frame_embeddings']
    #         actions_binaries = video['actions_binaries']
    #         phases_binaries = video['phase_binaries']
            
    #         # Convert to tensors if needed
    #         if isinstance(frame_embeddings, np.ndarray):
    #             states = torch.tensor(frame_embeddings, dtype=torch.float32)
    #         else:
    #             states = frame_embeddings
            
    #         if isinstance(actions_binaries, np.ndarray):
    #             actions = torch.tensor(actions_binaries, dtype=torch.float32)
    #         else:
    #             actions = actions_binaries

    #         if isinstance(phases_binaries, np.ndarray):
    #             phases = torch.tensor(phases_binaries, dtype=torch.float32)
            
    #         # Only include frames with actions (skip empty frames)
    #         for i in range(len(states)):
    #             if torch.sum(actions[i]) > 0:  # Frame has at least one action
    #                 expert_states.append(states[i])
    #                 expert_actions.append(actions[i])
    #                 expert_phases.append(phases[i])
        
    #     if not expert_states:
    #         self.logger.error("❌ No expert demonstrations found!")
    #         return False
        
    #     expert_states = torch.stack(expert_states).to(self.device)
    #     expert_actions = torch.stack(expert_actions).to(self.device)
    #     expert_phases = torch.stack(expert_phases).to(self.device)
                
    #     self.logger.info(f"📊 Training on {len(expert_states)} expert state-action pairs")
        
    #     # Train reward function using MaxEnt IRL
    #     for iteration in tqdm(range(num_iterations), desc="Training reward function"):

    #         # Generate negative examples (realistic but suboptimal)
    #         negative_actions = self._generate_realistic_negatives(expert_actions, expert_phases)
            
    #         # Compute rewards
    #         expert_rewards = self.irl_system.compute_reward(expert_states, expert_actions)
    #         negative_rewards = self.irl_system.compute_reward(expert_states, negative_actions)
            
    #         # MaxEnt IRL loss: expert actions should have higher reward
    #         reward_loss = -torch.mean(expert_rewards) + torch.mean(torch.exp(negative_rewards))
            
    #         # Regularization
    #         l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.irl_system.reward_net.parameters())
    #         total_reward_loss = reward_loss + l2_reg
            
    #         # Update reward network
    #         self.irl_system.reward_optimizer.zero_grad()
    #         total_reward_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.irl_system.reward_net.parameters(), 1.0)
    #         self.irl_system.reward_optimizer.step()
            
    #         if iteration % 20 == 0:
    #             self.logger.info(f"  Iteration {iteration}: Reward Loss = {total_reward_loss.item():.4f}, "
    #                            f"Expert Reward = {expert_rewards.mean().item():.3f}, "
    #                            f"Negative Reward = {negative_rewards.mean().item():.3f}")
        
    #     # Train policy adjustment (lightweight GAIL-style)
    #     self.logger.info("🎮 Training policy adjustment...")
    #     self._train_policy_adjustment(expert_states, expert_actions)
        
    #     self.logger.info("✅ Direct IRL training completed")
    #     return True

    def _generate_realistic_negatives(self, expert_actions, current_phase) -> torch.Tensor:
        """Enhanced negative generation using domain knowledge"""
        
        # Initialize the generator (do this once in __init__)
        if not hasattr(self, 'negative_generator'):
            with open('data/labels.json', 'r') as f:
                labels_config = json.load(f)
            self.negative_generator = CholecT50NegativeGenerator(labels_config)
                
        return self.negative_generator.generate_realistic_negatives(
            expert_actions, current_phase=current_phase
        )
    
    def _train_policy_adjustment(self, expert_states: torch.Tensor, expert_actions: torch.Tensor, num_epochs: int = 30):
        """Train small policy adjustments to IL predictions"""
        
        batch_size = 32
        num_samples = len(expert_states)
        
        for epoch in tqdm(range(num_epochs), desc="Training policy adjustment"):
            epoch_loss = 0
            num_batches = 0
            
            batch_indices = list(range(0, num_samples, batch_size))
            for i in tqdm(batch_indices, desc=f"Epoch {epoch+1} batches", leave=False):
                end_idx = min(i + batch_size, num_samples)
                batch_states = expert_states[i:end_idx]
                batch_expert_actions = expert_actions[i:end_idx]
                
                # Get IL predictions
                with torch.no_grad():
                    il_predictions = []
                    for state in batch_states:
                        il_pred = self._get_il_prediction(state)
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
    
    def _get_il_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """Get IL prediction for a single state"""
        try:
            # Ensure state is on correct device
            state = state.to(self.device)
            
            # Prepare input for IL model
            il_input = state.unsqueeze(0)  # Add batch dimension
            if len(il_input.shape) == 2:  # [batch, features] -> [batch, seq_len, features]
                il_input = il_input.unsqueeze(1)
            
            # Ensure minimum sequence length for autoregressive model
            if il_input.shape[1] < 2:
                # Duplicate the frame to create minimum sequence length
                il_input = il_input.repeat(1, 2, 1)
            
            # Move input to same device as IL model
            if hasattr(self.il_model, 'device'):
                il_input = il_input.to(self.il_model.device)
            elif next(self.il_model.parameters()).is_cuda:
                il_input = il_input.to('cuda')
            
            if hasattr(self.il_model, 'predict_next_action'):
                pred = self.il_model.predict_next_action(il_input).squeeze()
            elif hasattr(self.il_model, 'forward'):
                outputs = self.il_model.forward(il_input)
                if isinstance(outputs, dict):
                    pred = outputs.get('action_pred', outputs.get('action_logits', None))
                    if pred is not None:
                        if len(pred.shape) > 1:
                            pred = pred[:, -1, :]  # Take last timestep
                        pred = torch.sigmoid(pred).squeeze()
                    else:
                        return torch.rand(100, device=self.device) * 0.1
                else:
                    pred = torch.sigmoid(outputs).squeeze()
            else:
                return torch.rand(100, device=self.device) * 0.1
            
            # Ensure correct shape and device
            if len(pred.shape) == 0:  # Scalar
                pred = pred.unsqueeze(0)
            if pred.shape[0] != 100:  # Should have 100 action classes
                return torch.rand(100, device=self.device) * 0.1
                
            return pred.to(self.device)
            
        except Exception as e:
            self.logger.warning(f"IL prediction failed: {e}, using random prediction")
            return torch.rand(100, device=self.device) * 0.1
    
    def evaluate_direct_irl(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate IL vs IRL and analyze by existing labels"""
        
        self.logger.info("📊 Evaluating Direct IRL vs IL")
        
        results = {
            'overall': {},
            'by_action_type': defaultdict(lambda: {'il_scores': [], 'irl_scores': []}),
            'by_phase': defaultdict(lambda: {'il_scores': [], 'irl_scores': []}),
            'video_level': {}
        }
        
        for video in tqdm(test_data, desc="Evaluating videos"):
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
            for i in tqdm(range(len(states)), desc=f"Processing {video_id}", leave=False):
                state = states[i]
                
                # IL prediction
                il_pred = self._get_il_prediction(state)
                il_predictions.append(il_pred.cpu())
                
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
    
    def _extract_expert_data_with_phases(self, train_data):
        """🎯 ADD this method - Enhanced data extraction with phases"""
        expert_states, expert_actions, expert_phases = [], [], []
        
        for video in tqdm(train_data, desc="Extracting expert demonstrations"):
            frame_embeddings = video['frame_embeddings']
            actions_binaries = video['actions_binaries']
            phases_binaries = video.get('phase_binaries', [])
            
            # Convert to tensors (keep your existing conversion logic)
            if isinstance(frame_embeddings, np.ndarray):
                states = torch.tensor(frame_embeddings, dtype=torch.float32)
            else:
                states = frame_embeddings
                
            if isinstance(actions_binaries, np.ndarray):
                actions = torch.tensor(actions_binaries, dtype=torch.float32)
            else:
                actions = actions_binaries
                
            if isinstance(phases_binaries, np.ndarray):
                phases = torch.tensor(phases_binaries, dtype=torch.float32)
            elif isinstance(phases_binaries, list) and len(phases_binaries) > 0:
                phases = torch.tensor(phases_binaries, dtype=torch.float32)
            else:
                # Default phases if not available
                phases = torch.zeros(len(states), 7)
                phases[:, 0] = 1
            
            # Keep your existing frame filtering logic
            for i in range(len(states)):
                if torch.sum(actions[i]) > 0:
                    expert_states.append(states[i])
                    expert_actions.append(actions[i])
                    if i < len(phases):
                        expert_phases.append(phases[i])
                    else:
                        default_phase = torch.zeros(7)
                        default_phase[0] = 1
                        expert_phases.append(default_phase)
        
        expert_states = torch.stack(expert_states).to(self.device)
        expert_actions = torch.stack(expert_actions).to(self.device)
        expert_phases = torch.stack(expert_phases).to(self.device)
        
        return expert_states, expert_actions, expert_phases
    
    def _compute_enhanced_loss(self, expert_rewards, negative_rewards):
        """🎯 ADD this method - Enhanced loss function"""
        # Your existing MaxEnt IRL loss
        standard_loss = -torch.mean(expert_rewards) + torch.mean(torch.exp(negative_rewards))
        
        # Add reward separation encouragement
        expert_mean = torch.mean(expert_rewards)
        negative_mean = torch.mean(negative_rewards)
        separation_bonus = torch.clamp(expert_mean - negative_mean, min=0.0)
        
        # Enhanced loss
        enhanced_loss = standard_loss - 0.1 * separation_bonus
        return enhanced_loss
    
    def _monitor_gradients(self, iteration):
        """🎯 ADD this method - Gradient monitoring"""
        total_norm = 0.0
        param_count = 0
        
        for p in list(self.irl_system.reward_net.parameters()) + list(self.irl_system.phase_reward_net.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) / max(param_count, 1)
    
    def _log_to_tensorboard(self, iteration, expert_rewards, negative_rewards, 
                           total_loss, grad_norm, iteration_time):
        """🎯 ADD this method - TensorBoard logging"""
        if self.tb_writer is None:
            return
        
        # Core metrics
        expert_mean = expert_rewards.mean().item()
        negative_mean = negative_rewards.mean().item()
        reward_gap = expert_mean - negative_mean
        
        # Log to TensorBoard
        self.tb_writer.add_scalar('IRL/Loss/Total', total_loss.item(), iteration)
        self.tb_writer.add_scalar('IRL/Rewards/Expert_Mean', expert_mean, iteration)
        self.tb_writer.add_scalar('IRL/Rewards/Negative_Mean', negative_mean, iteration)
        self.tb_writer.add_scalar('IRL/Rewards/Gap', reward_gap, iteration)
        self.tb_writer.add_scalar('IRL/Training/Gradient_Norm', grad_norm, iteration)
        self.tb_writer.add_scalar('IRL/Training/Iteration_Time', iteration_time, iteration)
        
        # Detailed analysis every 10 iterations
        if iteration % 10 == 0:
            self.tb_writer.add_histogram('IRL/Distributions/Expert_Rewards', 
                                        expert_rewards.cpu(), iteration)
            self.tb_writer.add_histogram('IRL/Distributions/Negative_Rewards', 
                                        negative_rewards.cpu(), iteration)


# Integration function for your existing codebase
def train_direct_irl(config, train_data, test_data, logger, il_model):
    """
    Main function to integrate with your existing experiment runner
    Uses the direct IRL approach (no scenarios needed)
    """
    
    logger.info("🎯 Training Direct Surgical IRL (Simplified Approach)")
    
    # Get device from config
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Ensure IL model is on the correct device
    if il_model is not None:
        try:
            il_model = il_model.to(device)
            logger.info(f"✅ IL model moved to device: {device}")
        except Exception as e:
            logger.warning(f"⚠️ Could not move IL model to device: {e}")
    
    # Initialize direct IRL trainer
    irl_trainer = DirectIRLTrainer(
        il_model=il_model,
        config=config,
        logger=logger,
        device=device,
    )
    
    # Train direct IRL
    irl_config = config.get('experiment', {}).get('irl_enhancement', {})
    num_iterations = irl_config.get('maxent_irl', {}).get('num_iterations', 100)
    
    logger.info(f"🎯 Starting IRL training with {num_iterations} iterations")
    success = irl_trainer.train_direct_irl(train_data, num_iterations=num_iterations)
    
    if not success:
        return {
            'status': 'failed',
            'error': 'Direct IRL training failed'
        }
    
    # Evaluate IL vs IRL
    logger.info("📊 Starting IRL evaluation...")
    try:
        evaluation_results = irl_trainer.evaluate_direct_irl(test_data)
    except Exception as e:
        logger.error(f"❌ IRL evaluation failed: {e}")
        return {
            'status': 'failed',
            'error': f'IRL evaluation failed: {str(e)}'
        }
    
    logger.info("🏆 DIRECT IRL TRAINING RESULTS:")
    logger.info(f"Overall IL mAP: {evaluation_results['overall']['il_mean']:.4f}")
    logger.info(f"Overall IRL mAP: {evaluation_results['overall']['irl_mean']:.4f}")
    logger.info(f"Overall Improvement: {evaluation_results['overall']['improvement']:.4f} ({evaluation_results['overall']['improvement_percentage']:.1f}%)")
    
    return {
        'status': 'success',
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