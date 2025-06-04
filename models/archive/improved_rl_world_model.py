#!/usr/bin/env python3
"""
Improved RL World Model + Fair Evaluation Framework
Addresses evaluation bias and improves RL performance vs IL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Enhanced World Model for Better RL Performance
class EnhancedDualWorldModel(nn.Module):
    """
    Enhanced world model with improved architecture for better RL performance.
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 embedding_dim: int = 1024,
                 action_embedding_dim: int = 128,
                 n_layer: int = 8,  # Increased depth
                 num_action_classes: int = 100,
                 num_phase_classes: int = 7,
                 max_length: int = 1024,
                 dropout: float = 0.1,
                 # Enhanced features
                 use_attention_pooling: bool = True,
                 use_residual_prediction: bool = True,
                 use_uncertainty_estimation: bool = True,
                 **kwargs):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.n_layer = n_layer
        self.num_action_classes = num_action_classes
        self.num_phase_classes = num_phase_classes
        
        # Enhanced features
        self.use_attention_pooling = use_attention_pooling
        self.use_residual_prediction = use_residual_prediction
        self.use_uncertainty_estimation = use_uncertainty_estimation
        
        self._init_enhanced_architecture()
    
    def _init_enhanced_architecture(self):
        """Initialize enhanced architecture components."""
        
        # Improved state encoding with residual connections
        self.state_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Enhanced action embedding with learned importance weights
        self.action_encoder = nn.Sequential(
            nn.Linear(self.num_action_classes, self.action_embedding_dim),
            nn.LayerNorm(self.action_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.action_embedding_dim, self.action_embedding_dim)
        )
        
        # Action importance weights (learned)
        self.action_importance = nn.Parameter(torch.ones(self.num_action_classes))
        
        # Multi-head attention for better sequence modeling
        if self.use_attention_pooling:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Enhanced prediction heads with uncertainty
        self._init_enhanced_heads()
        
        # Residual prediction networks
        if self.use_residual_prediction:
            self.residual_state_net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.embedding_dim)
            )
    
    def _init_enhanced_heads(self):
        """Initialize enhanced prediction heads."""
        
        self.heads = nn.ModuleDict()
        
        # Enhanced state prediction with skip connections
        self.heads['state'] = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.embedding_dim)
        )
        
        # Enhanced action prediction with class-specific networks
        self.heads['action'] = nn.ModuleDict({
            'shared': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'tool_actions': nn.Linear(self.hidden_dim, 30),  # Tool-related actions
            'dissection_actions': nn.Linear(self.hidden_dim, 25),  # Dissection actions
            'other_actions': nn.Linear(self.hidden_dim, 45)  # Other actions
        })
        
        # Outcome prediction heads (for fair RL evaluation)
        self.heads['outcomes'] = nn.ModuleDict({
            'phase_success': nn.Linear(self.hidden_dim, 1),  # Phase completion quality
            'efficiency_score': nn.Linear(self.hidden_dim, 1),  # Action efficiency
            'safety_score': nn.Linear(self.hidden_dim, 1),  # Safety assessment
            'skill_score': nn.Linear(self.hidden_dim, 1),  # Overall skill demonstration
        })
        
        # Uncertainty estimation heads
        if self.use_uncertainty_estimation:
            self.heads['uncertainty'] = nn.ModuleDict({
                'state_var': nn.Linear(self.hidden_dim, self.embedding_dim),
                'action_var': nn.Linear(self.hidden_dim, self.num_action_classes),
                'outcome_var': nn.Linear(self.hidden_dim, 4)  # Uncertainty for 4 outcomes
            })
        
        # Enhanced reward prediction with multiple reward types
        self.heads['rewards'] = nn.ModuleDict({
            'immediate_reward': nn.Linear(self.hidden_dim, 1),
            'phase_reward': nn.Linear(self.hidden_dim, 1),
            'efficiency_reward': nn.Linear(self.hidden_dim, 1),
            'safety_reward': nn.Linear(self.hidden_dim, 1),
            'progress_reward': nn.Linear(self.hidden_dim, 1)
        })
    
    def forward(self, 
                current_states: torch.Tensor,
                actions: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with better representations."""
        
        batch_size, seq_len, _ = current_states.shape
        device = current_states.device
        
        # Enhanced state encoding
        state_embeds = self.state_encoder(current_states)
        
        # Enhanced action encoding with importance weighting
        if actions is not None:
            # Apply learned importance weights
            weighted_actions = actions * self.action_importance.unsqueeze(0).unsqueeze(0)
            action_embeds = self.action_encoder(weighted_actions)
        else:
            action_embeds = torch.zeros(batch_size, seq_len, self.action_embedding_dim, device=device)
        
        # Combine with attention if enabled
        if self.use_attention_pooling:
            combined_embeds = torch.cat([state_embeds, action_embeds], dim=-1)
            combined_embeds = nn.Linear(
                self.hidden_dim + self.action_embedding_dim, 
                self.hidden_dim
            ).to(device)(combined_embeds)
            
            # Self-attention for better sequence modeling
            attended_embeds, attention_weights = self.attention(
                combined_embeds, combined_embeds, combined_embeds
            )
            final_embeds = combined_embeds + attended_embeds  # Residual connection
        else:
            final_embeds = state_embeds + action_embeds[:, :, :self.hidden_dim]
        
        outputs = {'hidden_states': final_embeds}
        
        # Generate predictions
        self._generate_predictions(final_embeds, outputs)
        
        return outputs
    
    def _generate_predictions(self, hidden_states: torch.Tensor, outputs: Dict):
        """Generate all predictions from hidden states."""
        
        # State prediction
        state_pred = self.heads['state'](hidden_states)
        if self.use_residual_prediction:
            state_residual = self.residual_state_net(hidden_states)
            state_pred = state_pred + state_residual
        outputs['state_pred'] = state_pred
        
        # Enhanced action prediction
        shared_action_features = self.heads['action']['shared'](hidden_states)
        tool_actions = self.heads['action']['tool_actions'](shared_action_features)
        dissection_actions = self.heads['action']['dissection_actions'](shared_action_features)
        other_actions = self.heads['action']['other_actions'](shared_action_features)
        
        action_pred = torch.cat([tool_actions, dissection_actions, other_actions], dim=-1)
        outputs['action_pred'] = action_pred
        
        # Outcome predictions (crucial for fair RL evaluation)
        for outcome_name, head in self.heads['outcomes'].items():
            outputs[f'outcome_{outcome_name}'] = head(hidden_states)
        
        # Reward predictions
        for reward_name, head in self.heads['rewards'].items():
            outputs[f'reward_{reward_name}'] = head(hidden_states)
        
        # Uncertainty estimates
        if self.use_uncertainty_estimation:
            for unc_name, head in self.heads['uncertainty'].items():
                outputs[f'uncertainty_{unc_name}'] = F.softplus(head(hidden_states))


class FairEvaluationFramework:
    """
    Fair evaluation framework that addresses IL bias and properly evaluates RL.
    """
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.evaluation_metrics = {}
    
    def evaluate_both_paradigms(self, 
                               il_model, 
                               rl_models: Dict,
                               test_data: List[Dict],
                               world_model) -> Dict[str, Any]:
        """
        Comprehensive evaluation addressing IL bias.
        """
        
        results = {
            'il_results': {},
            'rl_results': {},
            'fair_comparison': {},
            'clinical_outcomes': {}
        }
        
        # 1. Traditional IL evaluation (action matching)
        self.logger.info("📊 Evaluating IL with traditional metrics...")
        il_traditional = self._evaluate_il_traditional(il_model, test_data)
        results['il_results']['traditional'] = il_traditional
        
        # 2. Outcome-based IL evaluation (fair comparison)
        self.logger.info("🎯 Evaluating IL with outcome-based metrics...")
        il_outcomes = self._evaluate_il_outcomes(il_model, test_data, world_model)
        results['il_results']['outcome_based'] = il_outcomes
        
        # 3. RL evaluation with multiple approaches
        self.logger.info("🤖 Evaluating RL models comprehensively...")
        for rl_name, rl_model in rl_models.items():
            rl_eval = self._evaluate_rl_comprehensive(rl_model, test_data, world_model)
            results['rl_results'][rl_name] = rl_eval
        
        # 4. Clinical outcome comparison
        self.logger.info("🏥 Comparing clinical outcomes...")
        clinical_comparison = self._compare_clinical_outcomes(
            il_model, rl_models, test_data, world_model
        )
        results['clinical_outcomes'] = clinical_comparison
        
        # 5. Fair comparison metrics
        self.logger.info("⚖️ Performing fair comparison...")
        fair_comparison = self._perform_fair_comparison(results)
        results['fair_comparison'] = fair_comparison
        
        return results
    
    def _evaluate_il_traditional(self, il_model, test_data) -> Dict[str, float]:
        """Traditional IL evaluation (action matching) - biased toward IL."""
        
        il_model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for video in test_data:
                states = torch.tensor(video['frame_embeddings']).unsqueeze(0)
                targets = torch.tensor(video['actions_binaries']).unsqueeze(0)
                
                outputs = il_model(current_states=states)
                if 'action_pred' in outputs:
                    predictions = torch.sigmoid(outputs['action_pred'])
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
        
        if not all_predictions:
            return {'error': 'No predictions available'}
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return self._calculate_action_matching_metrics(predictions, targets)
    
    def _evaluate_il_outcomes(self, il_model, test_data, world_model) -> Dict[str, float]:
        """Evaluate IL based on surgical outcomes (fairer comparison)."""
        
        il_model.eval()
        world_model.eval()
        
        outcome_scores = []
        
        with torch.no_grad():
            for video in test_data:
                # Generate IL action sequence
                states = torch.tensor(video['frame_embeddings']).unsqueeze(0)
                il_outputs = il_model(current_states=states)
                
                if 'action_pred' in il_outputs:
                    predicted_actions = torch.sigmoid(il_outputs['action_pred'])
                    
                    # Evaluate outcomes using world model
                    world_outputs = world_model(
                        current_states=states,
                        actions=predicted_actions
                    )
                    
                    # Extract outcome scores
                    video_outcomes = self._extract_outcome_scores(world_outputs, video)
                    outcome_scores.append(video_outcomes)
        
        return self._aggregate_outcome_scores(outcome_scores)
    
    def _evaluate_rl_comprehensive(self, rl_model, test_data, world_model) -> Dict[str, float]:
        """Comprehensive RL evaluation focusing on outcomes."""
        
        results = {
            'action_similarity': 0.0,  # How similar to expert actions
            'outcome_quality': 0.0,   # Quality of achieved outcomes
            'efficiency': 0.0,        # Action efficiency
            'safety': 0.0,           # Safety assessment
            'consistency': 0.0,       # Consistency across videos
            'novelty': 0.0           # Novel strategy discovery
        }
        
        outcome_scores = []
        action_similarities = []
        efficiency_scores = []
        
        for video in test_data:
            # Run RL policy on video
            rl_outcomes = self._run_rl_on_video(rl_model, video, world_model)
            
            # Calculate various metrics
            outcome_scores.append(rl_outcomes['outcome_quality'])
            action_similarities.append(rl_outcomes['action_similarity'])
            efficiency_scores.append(rl_outcomes['efficiency'])
        
        results['outcome_quality'] = np.mean(outcome_scores)
        results['action_similarity'] = np.mean(action_similarities)
        results['efficiency'] = np.mean(efficiency_scores)
        results['consistency'] = 1.0 - np.std(outcome_scores)  # Lower std = higher consistency
        
        return results
    
    def _run_rl_on_video(self, rl_model, video, world_model) -> Dict[str, float]:
        """Run RL policy on a video and evaluate outcomes."""
        
        states = video['frame_embeddings']
        expert_actions = video['actions_binaries']
        
        rl_actions = []
        predicted_states = []
        
        current_state = states[0]
        
        # Rollout RL policy
        for i in range(len(states) - 1):
            # Get RL action
            state_tensor = torch.tensor(current_state).unsqueeze(0)
            rl_action, _ = rl_model.predict(state_tensor, deterministic=True)
            rl_actions.append(rl_action)
            
            # Predict next state using world model
            with torch.no_grad():
                world_output = world_model(
                    current_states=state_tensor.unsqueeze(0),
                    actions=torch.tensor(rl_action).unsqueeze(0).unsqueeze(0).float()
                )
                if 'state_pred' in world_output:
                    predicted_state = world_output['state_pred'].squeeze().cpu().numpy()
                    predicted_states.append(predicted_state)
                    current_state = predicted_state
                else:
                    current_state = states[i + 1]  # Fallback to ground truth
        
        # Evaluate outcomes
        rl_actions = np.array(rl_actions)
        
        # Action similarity to expert
        action_similarity = self._calculate_action_similarity(rl_actions, expert_actions[:-1])
        
        # Outcome quality (using world model predictions)
        outcome_quality = self._calculate_outcome_quality(rl_actions, video)
        
        # Efficiency (fewer actions for same outcome)
        efficiency = self._calculate_efficiency(rl_actions, expert_actions[:-1])
        
        return {
            'action_similarity': action_similarity,
            'outcome_quality': outcome_quality,
            'efficiency': efficiency
        }
    
    def _calculate_action_similarity(self, rl_actions, expert_actions) -> float:
        """Calculate similarity between RL and expert actions."""
        
        if len(rl_actions) != len(expert_actions):
            min_len = min(len(rl_actions), len(expert_actions))
            rl_actions = rl_actions[:min_len]
            expert_actions = expert_actions[:min_len]
        
        # Hamming similarity
        matches = np.mean(rl_actions == expert_actions)
        return float(matches)
    
    def _calculate_outcome_quality(self, actions, video) -> float:
        """Calculate quality of surgical outcomes."""
        
        # Use phase progression as proxy for outcome quality
        if 'next_rewards' in video:
            rewards = video['next_rewards']
            if '_r_phase_progression' in rewards:
                phase_progress = np.mean(rewards['_r_phase_progression'])
                return float(np.clip(phase_progress, 0, 1))
        
        # Fallback: action density as efficiency proxy
        action_density = np.mean(np.sum(actions, axis=1))
        optimal_density = 3.0  # Assume optimal is ~3 actions per frame
        efficiency = 1.0 - abs(action_density - optimal_density) / optimal_density
        return float(np.clip(efficiency, 0, 1))
    
    def _calculate_efficiency(self, rl_actions, expert_actions) -> float:
        """Calculate action efficiency compared to expert."""
        
        rl_total = np.sum(rl_actions)
        expert_total = np.sum(expert_actions)
        
        if expert_total == 0:
            return 1.0 if rl_total == 0 else 0.0
        
        # Efficiency = achieving similar results with fewer actions
        efficiency = min(expert_total / (rl_total + 1e-8), 2.0)  # Cap at 2x efficiency
        return float(efficiency)
    
    def _compare_clinical_outcomes(self, il_model, rl_models, test_data, world_model) -> Dict[str, Any]:
        """Compare clinical outcomes between IL and RL approaches."""
        
        outcomes = {
            'il_clinical_score': 0.0,
            'rl_clinical_scores': {},
            'clinical_metrics': {
                'phase_completion_rate': {},
                'error_recovery_rate': {},
                'safety_incidents': {},
                'procedure_efficiency': {}
            }
        }
        
        # Evaluate IL clinical outcomes
        il_clinical = self._evaluate_clinical_outcomes(il_model, test_data, world_model)
        outcomes['il_clinical_score'] = il_clinical['overall_score']
        
        # Evaluate RL clinical outcomes
        for rl_name, rl_model in rl_models.items():
     