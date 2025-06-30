#!/usr/bin/env python3
"""
IRL Dataset for Next Action Prediction
Matches the temporal structure of AutoregressiveDataset for consistent training
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

class IRLNextActionDataset(Dataset):
    """
    IRL Dataset that matches AutoregressiveDataset structure
    Predicts NEXT actions (t+1) from current states (t)
    
    Key alignment with IL approach:
    - Same temporal structure as AutoregressiveDataset
    - Same context_length handling
    - Same target preparation for next action prediction
    - Compatible with sophisticated negative generation
    """
    
    def __init__(self, video_data: List[Dict], config: Dict):
        """
        Args:
            video_data: List of video dictionaries from load_cholect50_data
            config: Data configuration (same as AutoregressiveDataset)
        """
        self.samples = []
        
        # Use same parameters as AutoregressiveDataset
        context_length = config.get('context_length', 10)
        padding_value = config.get('padding_value', 0.0)
        
        print(f"🎯 Creating IRL Dataset with context_length={context_length} (matching IL approach)")
        
        for video in video_data:
            video_id = video['video_id']
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']  # Use correct key
            phases = video.get('phase_binaries', [])
            
            num_frames = len(embeddings)
            embedding_dim = embeddings.shape[1]
            num_actions = actions.shape[1]
            
            # Same iteration strategy as AutoregressiveDataset
            for i in range(num_frames - 1):  # -1 because we need t+1 for next prediction
                
                # Build current context sequence (for prediction at time t)
                current_context_frames = []
                current_context_actions = []
                current_context_phases = []
                
                # Get sequence indices (same as AutoregressiveDataset)
                sequence_indices = list(range(max(0, i - context_length + 1), i + 1))
                
                # Build current context sequences
                for idx, j in enumerate(sequence_indices):
                    if j < 0:
                        # Padding (same as AutoregressiveDataset)
                        current_context_frames.append([padding_value] * embedding_dim)
                        current_context_actions.append([0] * num_actions)
                        current_context_phases.append(0)
                    else:
                        # Current frame, action, phase at time j
                        current_context_frames.append(embeddings[j])
                        current_context_actions.append(actions[j])
                        
                        if len(phases) > j:
                            current_context_phases.append(np.argmax(phases[j]))
                        else:
                            current_context_phases.append(0)
                
                # Target next action (what we want to predict at t+1)
                if i + 1 < num_frames:
                    target_next_action = actions[i + 1]
                    target_next_phase = np.argmax(phases[i + 1]) if len(phases) > i + 1 else 0
                else:
                    # Edge case: use current action as fallback
                    target_next_action = actions[i]
                    target_next_phase = np.argmax(phases[i]) if len(phases) > i else 0
                
                # Only include samples where target has actions (like AutoregressiveDataset logic)
                if np.sum(target_next_action) > 0:
                    
                    # Ensure all sequences have the same length (like AutoregressiveDataset)
                    while len(current_context_frames) < context_length:
                        current_context_frames.insert(0, [padding_value] * embedding_dim)
                        current_context_actions.insert(0, [0] * num_actions)
                        current_context_phases.insert(0, 0)
                    
                    self.samples.append({
                        'video_id': video_id,
                        'frame_idx': i,
                        'target_frame_idx': i + 1,
                        
                        # Current context (input for prediction)
                        'current_context_frames': current_context_frames,  # [context_length, embedding_dim]
                        'current_context_actions': current_context_actions,  # [context_length, num_actions]
                        'current_context_phases': current_context_phases,   # [context_length]
                        
                        # Current state (for reward computation)
                        'current_state': embeddings[i],                    # [embedding_dim]
                        'current_action': actions[i],                      # [num_actions]
                        'current_phase': phases[i] if len(phases) > i else np.zeros(7),  # [7]
                        
                        # Next targets (what we want to predict)
                        'target_next_action': target_next_action,          # [num_actions]
                        'target_next_phase': target_next_phase,            # scalar
                    })
        
        print(f"✅ IRL Next Action Dataset created: {len(self.samples)} samples")
        print(f"   Temporal structure: current_context → predict next_action")
        print(f"   Compatible with AutoregressiveDataset approach")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx'],
            'target_frame_idx': sample['target_frame_idx'],
            
            # Current context (for IL model input)
            'current_context_frames': torch.tensor(np.array(sample['current_context_frames']), dtype=torch.float32),
            'current_context_actions': torch.tensor(np.array(sample['current_context_actions']), dtype=torch.float32),
            'current_context_phases': torch.tensor(np.array(sample['current_context_phases']), dtype=torch.long),
            
            # Current state (for IRL reward computation)
            'current_state': torch.tensor(sample['current_state'], dtype=torch.float32),
            'current_action': torch.tensor(sample['current_action'], dtype=torch.float32),
            'current_phase': torch.tensor(sample['current_phase'], dtype=torch.float32),
            
            # Target next action (what we want to predict)
            'target_next_action': torch.tensor(sample['target_next_action'], dtype=torch.float32),
            'target_next_phase': torch.tensor(sample['target_next_phase'], dtype=torch.long),
        }


def create_irl_next_action_dataloaders(train_data: List[Dict], 
                                       test_data: List[Dict],
                                       config: Dict,
                                       batch_size: int = 32,
                                       num_workers: int = 4) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """
    Create DataLoaders for IRL Next Action Prediction
    Matches create_autoregressive_dataloaders interface
    
    Args:
        train_data: Training video data
        test_data: Test video data
        config: Data configuration (same as AutoregressiveDataset)
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, test_loaders_dict)
    """
    
    print("🎯 Creating IRL DataLoaders for Next Action Prediction")
    
    # Training dataset
    train_dataset = IRLNextActionDataset(train_data, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Important for IRL training
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    # Test datasets (one per video for detailed analysis)
    test_loaders = {}
    for test_video in test_data:
        video_id = test_video['video_id']
        video_dataset = IRLNextActionDataset([test_video], config)
        
        if len(video_dataset) > 0:
            test_loaders[video_id] = DataLoader(
                video_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    print(f"✅ IRL Next Action DataLoaders created:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test videos: {len(test_loaders)}")
    print(f"   Task: Predict next_action(t+1) from current_context(t)")
    
    return train_loader, test_loaders


class EnhancedIRLNextActionTrainer:
    """
    Enhanced IRL Trainer for Next Action Prediction
    Matches the temporal structure and training approach of AutoregressiveILTrainer
    """
    
    def __init__(self, il_model, config, logger, device='cuda'):
        self.il_model = il_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Initialize IRL system (same as before)
        from training.irl_direct_trainer import DirectIRLSystem
        self.irl_system = DirectIRLSystem(
            state_dim=1024,
            action_dim=100,
            lr=1e-4,
            device=device
        )
        
        # Initialize sophisticated negative generator
        from datasets.irl_negative_generator import CholecT50NegativeGenerator
        import json
        with open('data/labels.json', 'r') as f:
            labels_config = json.load(f)
        self.negative_generator = CholecT50NegativeGenerator(labels_config)
    
    def train_next_action_irl(self, train_loader: DataLoader, 
                              test_loaders: Dict[str, DataLoader],
                              num_epochs: int = 20) -> bool:
        """
        Train IRL for Next Action Prediction task
        Matches the training approach of AutoregressiveILTrainer
        
        Args:
            train_loader: Training DataLoader with next action structure
            test_loaders: Test DataLoaders per video  
            num_epochs: Number of training epochs
            
        Returns:
            Success status
        """
        
        self.logger.info(f"🎯 Training IRL for Next Action Prediction ({num_epochs} epochs)")
        self.logger.info(f"   Training batches per epoch: {len(train_loader)}")
        self.logger.info(f"   Task: Learn rewards for next_action(t+1) predictions")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop with next action focus
            self.irl_system.train()
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                
                # Extract batch data
                current_states = batch['current_state'].to(self.device)           # [batch_size, embedding_dim]
                target_next_actions = batch['target_next_action'].to(self.device) # [batch_size, num_actions]
                current_phases = batch['current_phase'].to(self.device)           # [batch_size, 7]
                
                # EXPERT demonstrations: target_next_actions are the EXPERT choices for next actions
                expert_next_actions = target_next_actions
                
                # Generate sophisticated negatives for NEXT actions
                # This is key: we're learning what makes a good NEXT action vs bad NEXT action
                negative_next_actions = self.negative_generator.generate_realistic_negatives(
                    expert_next_actions, current_phase=current_phases
                ).to(self.device)
                
                # Compute rewards for expert vs negative NEXT actions
                # Reward function learns: "What makes a good next surgical action?"
                expert_rewards = self.irl_system.compute_reward(
                    current_states, expert_next_actions, current_phases
                )
                negative_rewards = self.irl_system.compute_reward(
                    current_states, negative_next_actions, current_phases
                )
                
                # Enhanced loss computation
                reward_loss = self._compute_enhanced_loss(expert_rewards, negative_rewards)
                
                # L2 regularization
                l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.irl_system.reward_net.parameters())
                total_loss = reward_loss + l2_reg
                
                # Optimization step
                self.irl_system.reward_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.irl_system.reward_net.parameters(), 1.0)
                self.irl_system.reward_optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Log batch progress
                if batch_idx % 10 == 0:
                    expert_mean = expert_rewards.mean().item()
                    negative_mean = negative_rewards.mean().item()
                    gap = expert_mean - negative_mean
                    
                    self.logger.info(
                        f"  Batch {batch_idx}/{len(train_loader)}: "
                        f"Loss={total_loss:.4f}, Expert_Next={expert_mean:+.4f}, "
                        f"Negative_Next={negative_mean:+.4f}, Gap={gap:+.4f}"
                    )
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch+1} completed: Avg Loss = {avg_epoch_loss:.4f}")
            
            # Evaluate on test data every few epochs
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"Running next action evaluation after epoch {epoch+1}...")
                eval_results = self.evaluate_next_action_prediction(test_loaders)
                improvement = eval_results['overall']['improvement']
                self.logger.info(f"Current next action improvement: {improvement:.4f} mAP")
        
        # Train policy adjustment for NEXT action prediction
        self.logger.info("🎮 Training policy adjustment for next action prediction...")
        self._train_next_action_policy_adjustment(train_loader)
        
        self.logger.info("✅ IRL Next Action Prediction training completed")
        return True
    
    def _train_next_action_policy_adjustment(self, train_loader: DataLoader, num_epochs: int = 10):
        """
        Train policy adjustment specifically for next action prediction
        Matches the IL approach where we adjust IL's next action predictions
        """
        
        self.logger.info("🎮 Training Policy Adjustment for Next Action Prediction")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Policy epoch {epoch+1}"):
                current_context_frames = batch['current_context_frames'].to(self.device)  # [batch, context_length, embedding_dim]
                target_next_actions = batch['target_next_action'].to(self.device)         # [batch, num_actions]
                current_states = batch['current_state'].to(self.device)                  # [batch, embedding_dim]
                current_phases = batch['current_phase'].to(self.device)                  # [batch, 7]
                
                # Get IL predictions for NEXT actions (this is the key alignment!)
                il_next_predictions = []
                for i in range(current_context_frames.shape[0]):
                    context = current_context_frames[i:i+1]  # [1, context_length, embedding_dim]
                    
                    with torch.no_grad():
                        # Use IL model to predict NEXT action from current context
                        il_next_pred = self.il_model.predict_next_action(context)
                        il_next_predictions.append(il_next_pred)
                
                il_next_predictions = torch.stack(il_next_predictions).to(self.device)
                
                # Apply policy adjustment to IL's NEXT action predictions
                # Policy adjustment learns: "How to improve IL's next action predictions"
                adjustments = self.irl_system.policy_adjustment(
                    current_states, il_next_predictions, current_phases
                )
                adjusted_next_predictions = torch.sigmoid(il_next_predictions + 0.05 * adjustments)
                
                # Compute policy loss: adjusted predictions should have higher reward than IL alone
                adjusted_rewards = self.irl_system.compute_reward(
                    current_states, adjusted_next_predictions, current_phases
                )
                il_rewards = self.irl_system.compute_reward(
                    current_states, il_next_predictions, current_phases
                )
                
                # Policy loss: maximize adjusted rewards + match expert next actions
                policy_loss = -torch.mean(adjusted_rewards) + 0.5 * torch.nn.functional.mse_loss(
                    adjusted_next_predictions, target_next_actions
                )
                
                # Optimization
                self.irl_system.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.irl_system.policy_optimizer.step()
                
                epoch_loss += policy_loss.item()
                num_batches += 1
            
            self.logger.info(f"Policy epoch {epoch+1}: Next Action Loss = {epoch_loss/num_batches:.4f}")
    
    def evaluate_next_action_prediction(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Evaluate IRL vs IL on Next Action Prediction task
        Matches AutoregressiveILTrainer evaluation approach
        """
        
        self.irl_system.eval()
        results = {
            'overall': {},
            'video_level': {}
        }
        
        all_il_scores = []
        all_irl_scores = []
        
        for video_id, test_loader in test_loaders.items():
            video_il_scores = []
            video_irl_scores = []
            
            for batch in test_loader:
                current_context_frames = batch['current_context_frames'].to(self.device)
                target_next_actions = batch['target_next_action'].to(self.device)
                current_states = batch['current_state'].to(self.device)
                current_phases = batch['current_phase'].to(self.device)
                
                # Get IL and IRL predictions for NEXT actions
                il_next_predictions = []
                irl_next_predictions = []
                
                for i in range(current_context_frames.shape[0]):
                    context = current_context_frames[i:i+1]
                    current_state = current_states[i]
                    current_phase = current_phases[i]
                    
                    with torch.no_grad():
                        # IL prediction for next action
                        il_next_pred = self.il_model.predict_next_action(context)
                        il_next_predictions.append(il_next_pred)
                        
                        # IRL prediction for next action (IL + policy adjustment)
                        irl_next_pred = self.irl_system.predict_with_irl(
                            self.il_model, current_state, current_phase
                        )
                        irl_next_predictions.append(irl_next_pred)
                
                il_batch = torch.stack(il_next_predictions)
                irl_batch = torch.stack(irl_next_predictions)
                
                # Calculate mAP for next action prediction
                il_score = self._calculate_map(il_batch, target_next_actions)
                irl_score = self._calculate_map(irl_batch, target_next_actions)
                
                video_il_scores.append(il_score)
                video_irl_scores.append(irl_score)
            
            # Video-level results
            video_il_mean = np.mean(video_il_scores)
            video_irl_mean = np.mean(video_irl_scores)
            
            results['video_level'][video_id] = {
                'il_next_action_score': video_il_mean,
                'irl_next_action_score': video_irl_mean,
                'next_action_improvement': video_irl_mean - video_il_mean
            }
            
            all_il_scores.extend(video_il_scores)
            all_irl_scores.extend(video_irl_scores)
        
        # Overall results
        results['overall'] = {
            'il_next_action_mean': np.mean(all_il_scores),
            'irl_next_action_mean': np.mean(all_irl_scores),
            'improvement': np.mean(all_irl_scores) - np.mean(all_il_scores),
            'improvement_percentage': ((np.mean(all_irl_scores) - np.mean(all_il_scores)) / np.mean(all_il_scores)) * 100
        }
        
        return results
    
    def _compute_enhanced_loss(self, expert_rewards, negative_rewards):
        """Enhanced loss function (same as before)"""
        standard_loss = -torch.mean(expert_rewards) + torch.mean(torch.exp(negative_rewards))
        expert_mean = torch.mean(expert_rewards)
        negative_mean = torch.mean(negative_rewards)
        separation_bonus = torch.clamp(expert_mean - negative_mean, min=0.0)
        enhanced_loss = standard_loss - 0.1 * separation_bonus
        return enhanced_loss
    
    def _calculate_map(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """mAP calculation (same as before)"""
        try:
            from sklearn.metrics import average_precision_score
            
            pred_np = predictions.detach().cpu().numpy()
            target_np = targets.cpu().numpy()
            
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
            pred_binary = (predictions > 0.5).float()
            accuracy = (pred_binary == targets).float().mean()
            return accuracy.item()


# Integration function for your experiment runner
def train_irl_next_action_prediction(config, train_data, test_data, logger, il_model):
    """
    IRL training function that matches AutoregressiveIL approach
    Trains on next action prediction task with sophisticated negatives
    """
    
    logger.info("🎯 Training IRL for Next Action Prediction (Matching IL Approach)")
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DataLoaders with same config as AutoregressiveDataset
    data_config = config['data']
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    train_loader, test_loaders = create_irl_next_action_dataloaders(
        train_data=train_data,
        test_data=test_data,
        config=data_config,  # Same config as AutoregressiveDataset
        batch_size=batch_size,
        num_workers=config.get('training', {}).get('num_workers', 4)
    )
    
    # Create enhanced trainer
    trainer = EnhancedIRLNextActionTrainer(
        il_model=il_model,
        config=config,
        logger=logger,
        device=device
    )
    
    # Train with next action prediction focus
    num_epochs = config.get('experiment', {}).get('irl_enhancement', {}).get('num_epochs', 20)
    success = trainer.train_next_action_irl(train_loader, test_loaders, num_epochs)
    
    if not success:
        return {'status': 'failed', 'error': 'IRL Next Action training failed'}
    
    # Evaluate next action prediction performance
    evaluation_results = trainer.evaluate_next_action_prediction(test_loaders)
    
    logger.info("🏆 IRL NEXT ACTION PREDICTION RESULTS:")
    logger.info(f"IL Next Action mAP: {evaluation_results['overall']['il_next_action_mean']:.4f}")
    logger.info(f"IRL Next Action mAP: {evaluation_results['overall']['irl_next_action_mean']:.4f}")
    logger.info(f"Next Action Improvement: {evaluation_results['overall']['improvement']:.4f} ({evaluation_results['overall']['improvement_percentage']:.1f}%)")
    
    return {
        'status': 'success',
        'irl_trainer': trainer,
        'evaluation_results': evaluation_results,
        'technique': 'IRL for Next Action Prediction',
        'device': device,
        'approach': 'Next action prediction with sophisticated negatives (matching IL approach)',
        'task_alignment': 'next_action_prediction',
        'temporal_structure': 'current_context(t) → predict_next_action(t+1)',
        'improvements': [
            'Matches AutoregressiveIL temporal structure',
            'Next action prediction focus',
            'Sophisticated negative generation for next actions',
            'Policy adjustment on IL next action predictions',
            'Consistent task definition across IL and IRL'
        ]
    }


if __name__ == "__main__":
    print("🎯 IRL NEXT ACTION PREDICTION DATASET")
    print("=" * 50)
    print("✅ Matches AutoregressiveDataset temporal structure")
    print("✅ Predicts next_action(t+1) from current_context(t)")
    print("✅ Compatible with sophisticated negative generation")
    print("✅ Policy adjustment on IL next action predictions")
    print("✅ Consistent task definition with IL approach")
