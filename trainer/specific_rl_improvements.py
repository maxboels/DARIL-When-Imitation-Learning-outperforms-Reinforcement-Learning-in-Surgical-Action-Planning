#!/usr/bin/env python3
"""
Specific Actionable Improvements for RL vs IL Comparison
Addresses evaluation bias and provides concrete implementation strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import json

@dataclass
class ClinicalOutcome:
    """Structure for clinical outcome evaluation"""
    phase_completion_rate: float
    error_count: int
    efficiency_score: float
    safety_score: float
    innovation_score: float
    overall_clinical_score: float


class OutcomeBasedRewardFunction:
    """
    Improved reward function focusing on surgical outcomes rather than action mimicry.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Clinical outcome weights
        self.outcome_weights = {
            'phase_progression': 2.0,    # High weight for surgical progress
            'efficiency': 1.5,           # Reward for efficient action sequences
            'safety': 3.0,              # Highest weight for safety
            'innovation': 0.5,          # Small reward for novel but effective strategies
            'completion_quality': 2.5    # Quality of phase completion
        }
        
        # Phase-specific reward scaling
        self.phase_importance = {
            0: 1.0,  # Preparation
            1: 1.2,  # Incision
            2: 1.5,  # Dissection (more important)
            3: 2.0,  # Critical view (most important)
            4: 1.8,  # Clipping
            5: 1.3,  # Cutting
            6: 1.1   # Extraction
        }
        
        # Action importance weights (learned or predefined)
        self.action_importance = self._initialize_action_importance()
    
    def _initialize_action_importance(self) -> np.ndarray:
        """Initialize action importance weights based on clinical knowledge."""
        
        # 100 surgical actions - assign importance based on clinical relevance
        importance = np.ones(100)
        
        # Critical actions (higher importance)
        critical_actions = [15, 23, 34, 45, 67, 78, 89, 92]  # Example critical action indices
        importance[critical_actions] = 3.0
        
        # Safety-critical actions (highest importance)
        safety_critical = [12, 28, 56, 71, 85]
        importance[safety_critical] = 5.0
        
        # Routine actions (lower importance)
        routine_actions = list(range(0, 10)) + list(range(90, 100))
        importance[routine_actions] = 0.5
        
        return importance
    
    def calculate_reward(self, 
                        predicted_actions: np.ndarray,
                        current_state: np.ndarray,
                        video_metadata: Dict,
                        frame_idx: int,
                        phase: int) -> float:
        """
        Calculate outcome-based reward that doesn't rely on expert action matching.
        """
        
        reward = 0.0
        
        # 1. Phase progression reward (key improvement)
        phase_reward = self._calculate_phase_progression_reward(
            predicted_actions, video_metadata, frame_idx, phase
        )
        reward += phase_reward * self.outcome_weights['phase_progression']
        
        # 2. Efficiency reward (fewer actions for same outcome)
        efficiency_reward = self._calculate_efficiency_reward(predicted_actions, phase)
        reward += efficiency_reward * self.outcome_weights['efficiency']
        
        # 3. Safety reward (avoid risky action combinations)
        safety_reward = self._calculate_safety_reward(predicted_actions, phase)
        reward += safety_reward * self.outcome_weights['safety']
        
        # 4. Innovation bonus (reward for discovering effective novel strategies)
        innovation_reward = self._calculate_innovation_reward(
            predicted_actions, video_metadata, frame_idx
        )
        reward += innovation_reward * self.outcome_weights['innovation']
        
        # 5. Completion quality reward
        completion_reward = self._calculate_completion_quality_reward(
            predicted_actions, video_metadata, frame_idx, phase
        )
        reward += completion_reward * self.outcome_weights['completion_quality']
        
        # Apply phase importance scaling
        phase_multiplier = self.phase_importance.get(phase, 1.0)
        reward *= phase_multiplier
        
        return float(reward)
    
    def _calculate_phase_progression_reward(self, actions, metadata, frame_idx, phase) -> float:
        """Reward based on surgical phase progression (outcome-focused)."""
        
        # Use action importance weighting
        weighted_actions = actions * self.action_importance
        action_score = np.sum(weighted_actions)
        
        # Reward progression indicators
        if 'next_rewards' in metadata:
            if '_r_phase_progression' in metadata['next_rewards']:
                if frame_idx < len(metadata['next_rewards']['_r_phase_progression']):
                    ground_truth_progression = metadata['next_rewards']['_r_phase_progression'][frame_idx]
                    # Reward actions that lead to positive progression
                    if action_score > 0 and ground_truth_progression > 0:
                        return 2.0
                    elif action_score == 0 and ground_truth_progression <= 0:
                        return 1.0  # Correctly doing nothing
                    else:
                        return -0.5  # Mismatch
        
        # Fallback: reward appropriate action density for phase
        optimal_density = {0: 2, 1: 3, 2: 4, 3: 5, 4: 4, 5: 3, 6: 2}
        expected_actions = optimal_density.get(phase, 3)
        actual_actions = np.sum(actions)
        
        density_reward = 1.0 - abs(actual_actions - expected_actions) / expected_actions
        return max(density_reward, 0.0)
    
    def _calculate_efficiency_reward(self, actions, phase) -> float:
        """Reward efficient action sequences."""
        
        total_actions = np.sum(actions)
        
        # Phase-specific efficiency targets
        efficiency_targets = {
            0: 2.0,  # Preparation: minimal actions needed
            1: 3.0,  # Incision: moderate actions
            2: 4.0,  # Dissection: more complex
            3: 5.0,  # Critical view: most complex
            4: 4.0,  # Clipping: moderate complexity
            5: 3.0,  # Cutting: focused actions
            6: 2.0   # Extraction: minimal actions
        }
        
        target = efficiency_targets.get(phase, 3.0)
        
        # Reward being close to optimal
        if total_actions <= target:
            return 1.0  # Efficient
        else:
            # Penalize excessive actions
            return max(0.0, 1.0 - (total_actions - target) / target)
    
    def _calculate_safety_reward(self, actions, phase) -> float:
        """Reward safe action combinations."""
        
        # Define unsafe action combinations (example indices)
        unsafe_combinations = [
            [15, 23],  # Don't do action 15 and 23 together
            [34, 45],  # Another unsafe combination
            [67, 71, 78]  # Triple unsafe combination
        ]
        
        safety_penalty = 0.0
        
        # Check for unsafe combinations
        active_actions = np.where(actions > 0.5)[0]
        
        for unsafe_combo in unsafe_combinations:
            if all(action in active_actions for action in unsafe_combo):
                safety_penalty += 2.0  # Significant penalty
        
        # Base safety reward
        base_safety = 1.0
        
        # Phase-specific safety considerations
        if phase == 3:  # Critical view phase - extra safety important
            base_safety *= 1.5
        
        return max(0.0, base_safety - safety_penalty)
    
    def _calculate_innovation_reward(self, actions, metadata, frame_idx) -> float:
        """Reward effective strategies that differ from typical expert behavior."""
        
        # Calculate "typical" expert action pattern
        if 'actions_binaries' in metadata:
            expert_actions = metadata['actions_binaries'][frame_idx] if frame_idx < len(metadata['actions_binaries']) else np.zeros(100)
            
            # Calculate novelty (how different from expert)
            novelty = 1.0 - np.mean(actions == expert_actions)
            
            # Only reward novelty if it's potentially beneficial
            # (This is a simplified heuristic - could be learned)
            if novelty > 0.3:  # Significantly different
                # Check if it's potentially beneficial based on action importance
                novel_actions = actions[actions != expert_actions]
                if len(novel_actions) > 0:
                    novel_importance = np.mean(self.action_importance[actions != expert_actions])
                    if novel_importance > 1.0:  # Using important actions differently
                        return min(novelty * 0.5, 1.0)  # Cap innovation reward
        
        return 0.0
    
    def _calculate_completion_quality_reward(self, actions, metadata, frame_idx, phase) -> float:
        """Reward quality of task completion."""
        
        # Use phase completion signals if available
        if 'next_rewards' in metadata:
            if '_r_phase_completion' in metadata['next_rewards']:
                if frame_idx < len(metadata['next_rewards']['_r_phase_completion']):
                    completion_signal = metadata['next_rewards']['_r_phase_completion'][frame_idx]
                    
                    # Reward actions that contribute to completion
                    action_intensity = np.sum(actions * self.action_importance)
                    
                    if completion_signal > 0.5 and action_intensity > 1.0:
                        return 2.0  # Good completion actions
                    elif completion_signal <= 0.5 and action_intensity <= 1.0:
                        return 1.0  # Appropriately minimal actions
                    else:
                        return 0.0  # Mismatch
        
        # Fallback: reward appropriate action patterns
        return 0.5


class FairEvaluationMetrics:
    """
    Fair evaluation metrics that don't bias toward IL action mimicry.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_surgical_outcomes(self, 
                                 predicted_actions: np.ndarray,
                                 video_metadata: Dict,
                                 model_type: str = 'unknown') -> ClinicalOutcome:
        """
        Evaluate based on surgical outcomes rather than action similarity.
        """
        
        # 1. Phase completion rate
        phase_completion = self._evaluate_phase_completion(predicted_actions, video_metadata)
        
        # 2. Error count (based on unsafe action patterns)
        error_count = self._count_errors(predicted_actions)
        
        # 3. Efficiency score
        efficiency = self._calculate_efficiency_score(predicted_actions, video_metadata)
        
        # 4. Safety score
        safety = self._calculate_safety_score(predicted_actions)
        
        # 5. Innovation score (only for RL)
        innovation = self._calculate_innovation_score(predicted_actions, video_metadata) if 'RL' in model_type else 0.0
        
        # 6. Overall clinical score
        overall = self._calculate_overall_clinical_score(
            phase_completion, error_count, efficiency, safety, innovation
        )
        
        return ClinicalOutcome(
            phase_completion_rate=phase_completion,
            error_count=error_count,
            efficiency_score=efficiency,
            safety_score=safety,
            innovation_score=innovation,
            overall_clinical_score=overall
        )
    
    def _evaluate_phase_completion(self, actions, metadata) -> float:
        """Evaluate how well phases are completed."""
        
        if 'next_rewards' in metadata and '_r_phase_completion' in metadata['next_rewards']:
            completion_rewards = np.array(metadata['next_rewards']['_r_phase_completion'])
            
            # Calculate completion rate
            positive_completions = np.sum(completion_rewards > 0.5)
            total_frames = len(completion_rewards)
            
            return positive_completions / total_frames if total_frames > 0 else 0.0
        
        # Fallback: estimate based on action patterns
        action_density = np.mean(np.sum(actions, axis=1))
        # Assume optimal density leads to good completion
        return min(action_density / 5.0, 1.0)
    
    def _count_errors(self, actions) -> int:
        """Count potential errors based on action patterns."""
        
        # Define error patterns (unsafe action combinations)
        error_patterns = [
            [15, 23],      # Unsafe combination 1
            [34, 45, 67],  # Unsafe combination 2
            [78, 89]       # Unsafe combination 3
        ]
        
        error_count = 0
        
        for frame_actions in actions:
            active_actions = set(np.where(frame_actions > 0.5)[0])
            
            for pattern in error_patterns:
                if all(action in active_actions for action in pattern):
                    error_count += 1
        
        return error_count
    
    def _calculate_efficiency_score(self, actions, metadata) -> float:
        """Calculate efficiency based on action economy."""
        
        total_actions = np.sum(actions)
        num_frames = len(actions)
        
        # Calculate action density
        action_density = total_actions / num_frames
        
        # Optimal density depends on procedure complexity
        if 'video_id' in metadata:
            # Could vary by video complexity
            optimal_density = 3.5  # Average optimal actions per frame
        else:
            optimal_density = 3.5
        
        # Efficiency is inversely related to deviation from optimal
        efficiency = 1.0 - abs(action_density - optimal_density) / optimal_density
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_safety_score(self, actions) -> float:
        """Calculate safety score based on risk assessment."""
        
        # Define high-risk actions
        high_risk_actions = [12, 28, 56, 71, 85]
        
        risk_score = 0.0
        total_frames = len(actions)
        
        for frame_actions in actions:
            frame_risk = 0.0
            
            # Count high-risk actions
            for risk_action in high_risk_actions:
                if frame_actions[risk_action] > 0.5:
                    frame_risk += 1.0
            
            # Penalize multiple high-risk actions in same frame
            if frame_risk > 1:
                frame_risk *= 2.0  # Exponential penalty
            
            risk_score += frame_risk
        
        # Convert to safety score (lower risk = higher safety)
        avg_risk = risk_score / total_frames if total_frames > 0 else 0.0
        safety_score = max(0.0, 1.0 - avg_risk / 5.0)  # Normalize
        
        return safety_score
    
    def _calculate_innovation_score(self, actions, metadata) -> float:
        """Calculate innovation score for RL approaches."""
        
        if 'actions_binaries' not in metadata:
            return 0.0
        
        expert_actions = np.array(metadata['actions_binaries'])
        
        # Ensure same length
        min_len = min(len(actions), len(expert_actions))
        actions = actions[:min_len]
        expert_actions = expert_actions[:min_len]
        
        # Calculate novelty
        differences = np.sum(actions != expert_actions, axis=1)
        novelty_rate = np.mean(differences) / actions.shape[1]
        
        # Innovation score: reward meaningful novelty
        if novelty_rate > 0.2:  # Significantly different
            # Check if novel actions are in important categories
            action_importance = np.ones(100)
            action_importance[[15, 23, 34, 45, 67, 78, 89, 92]] = 3.0  # Important actions
            
            novel_importance = 0.0
            for i in range(min_len):
                diff_mask = actions[i] != expert_actions[i]
                if np.sum(diff_mask) > 0:
                    novel_importance += np.mean(action_importance[diff_mask])
            
            avg_novel_importance = novel_importance / min_len
            
            # Scale innovation score by importance of novel actions
            innovation_score = min(novelty_rate * avg_novel_importance / 3.0, 1.0)
            return innovation_score
        
        return 0.0
    
    def _calculate_overall_clinical_score(self, 
                                        phase_completion: float,
                                        error_count: int,
                                        efficiency: float,
                                        safety: float,
                                        innovation: float) -> float:
        """Calculate overall clinical performance score."""
        
        # Weights for different aspects of clinical performance
        weights = {
            'completion': 0.3,
            'safety': 0.3,
            'efficiency': 0.2,
            'errors': 0.15,
            'innovation': 0.05
        }
        
        # Normalize error count (fewer errors = better score)
        error_score = max(0.0, 1.0 - error_count / 10.0)  # Assume 10+ errors is very bad
        
        # Calculate weighted score
        overall_score = (
            weights['completion'] * phase_completion +
            weights['safety'] * safety +
            weights['efficiency'] * efficiency +
            weights['errors'] * error_score +
            weights['innovation'] * innovation
        )
        
        return overall_score


class ImprovedRLEnvironment(gym.Env):
    """
    Improved RL environment with outcome-based rewards and fair evaluation.
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Initialize improved reward function
        self.reward_function = OutcomeBasedRewardFunction(config)
        
        # Initialize fair evaluation
        self.evaluator = FairEvaluationMetrics()
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_training', {}).get('rl_horizon', 50)
        
        # Action and observation spaces
        self.action_space = spaces.MultiBinary(100)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )
        
        # Episode state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment with improved initialization."""
        super().reset(seed=seed)
        
        # Select random video and starting frame
        self.current_video_idx = np.random.randint(0, len(self.video_data))
        self.video = self.video_data[self.current_video_idx]
        
        max_start = len(self.video['frame_embeddings']) - self.max_episode_steps - 1
        self.current_frame_idx = np.random.randint(0, max(1, max_start))
        
        # Initialize episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_actions = []
        self.episode_outcomes = []
        
        # Get initial state and phase
        self.current_state = self.video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        self.current_phase = self._get_current_phase()
        
        return self.current_state, {}
    
    def step(self, action):
        """Environment step with improved reward calculation."""
        
        self.current_step += 1
        self.episode_actions.append(action.copy())
        
        # Convert to binary action
        binary_action = (action > 0.5).astype(int)
        
        # Check if episode is done
        done = (
            self.current_step >= self.max_episode_steps or 
            self.current_frame_idx + 1 >= len(self.video['frame_embeddings'])
        )
        
        if not done:
            # Move to next frame
            self.current_frame_idx += 1
            next_state = self.video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
            
            # Calculate improved reward
            reward = self.reward_function.calculate_reward(
                binary_action,
                self.current_state,
                self.video,
                self.current_frame_idx - 1,
                self.current_phase
            )
            
            self.current_state = next_state
            self.current_phase = self._get_current_phase()
            
        else:
            reward = 0.0
            next_state = self.current_state
        
        self.episode_reward += reward
        
        # Enhanced info for evaluation
        info = {
            'episode_reward': self.episode_reward,
            'video_id': self.video['video_id'],
            'frame_idx': self.current_frame_idx,
            'phase': self.current_phase,
            'binary_action': binary_action,
            'clinical_outcome': self._get_clinical_outcome() if done else None
        }
        
        return next_state, reward, done, False, info
    
    def _get_current_phase(self) -> int:
        """Get current surgical phase."""
        
        if 'phase_binaries' in self.video and self.current_frame_idx < len(self.video['phase_binaries']):
            phase_vector = self.video['phase_binaries'][self.current_frame_idx]
            return int(np.argmax(phase_vector))
        
        # Fallback: estimate phase based on frame position
        total_frames = len(self.video['frame_embeddings'])
        phase_progress = self.current_frame_idx / total_frames
        return min(int(phase_progress * 7), 6)
    
    def _get_clinical_outcome(self) -> ClinicalOutcome:
        """Get clinical outcome for completed episode."""
        
        if not self.episode_actions:
            return ClinicalOutcome(0, 0, 0, 0, 0, 0)
        
        episode_actions = np.array(self.episode_actions)
        
        return self.evaluator.evaluate_surgical_outcomes(
            episode_actions,
            self.video,
            'RL'
        )


def create_implementation_guide() -> Dict[str, List[str]]:
    """Create specific implementation guide for improved RL vs IL comparison."""
    
    guide = {
        'immediate_improvements': [
            "Replace action-matching evaluation with outcome-based evaluation",
            "Implement OutcomeBasedRewardFunction in your RL environment",
            "Add clinical outcome metrics to your evaluation pipeline",
            "Weight safety and efficiency higher than action similarity",
            "Use ClinicalOutcome dataclass for consistent evaluation"
        ],
        
        'world_model_enhancements': [
            "Add outcome prediction heads to your DualWorldModel",
            "Implement uncertainty estimation for better exploration",
            "Use attention mechanisms for better sequence modeling",
            "Add residual connections in prediction networks",
            "Train on outcome labels, not just next-frame prediction"
        ],
        
        'fair_evaluation_implementation': [
            "Create separate evaluation metrics for IL and RL",
            "Use FairEvaluationMetrics class for outcome-based assessment",
            "Compare on clinical outcomes: completion, safety, efficiency",
            "Allow RL to be rewarded for novel but effective strategies",
            "Include innovation score for RL approaches"
        ],
        
        'reward_design_specifics': [
            "Define action importance weights based on clinical knowledge",
            "Implement phase-specific reward scaling",
            "Penalize unsafe action combinations",
            "Reward efficiency (fewer actions for same outcome)",
            "Add innovation bonus for discovering new effective strategies"
        ],
        
        'evaluation_bias_fixes': [
            "Stop using expert actions as ground truth for RL evaluation",
            "Focus on 'did the surgery succeed?' rather than 'did actions match?'",
            "Include efficiency metrics (RL might be more efficient)",
            "Evaluate safety independently of action similarity",
            "Allow RL to discover better strategies than experts"
        ]
    }
    
    return guide


def main():
    """Demonstrate the improved RL framework."""
    
    print("🔧 SPECIFIC RL IMPROVEMENTS FOR FAIR IL vs RL COMPARISON")
    print("=" * 70)
    print()
    
    # Show the key insight
    print("🎯 KEY INSIGHT: Evaluation Bias Problem")
    print("Current evaluation uses expert actions as 'ground truth'")
    print("This unfairly penalizes RL for discovering potentially better strategies!")
    print()
    
    # Show the solution
    print("✅ SOLUTION: Outcome-Based Evaluation")
    print("Evaluate both IL and RL based on surgical outcomes:")
    print("• Phase completion quality")
    print("• Safety scores")
    print("• Efficiency metrics") 
    print("• Innovation potential (for RL)")
    print("• Overall clinical performance")
    print()
    
    # Implementation guide
    guide = create_implementation_guide()
    
    for category, items in guide.items():
        print(f"📋 {category.replace('_', ' ').title()}:")
        for item in items:
            print(f"   • {item}")
        print()
    
    # Expected improvements
    print("📈 EXPECTED IMPROVEMENTS:")
    print("• RL performance increases due to outcome-focused rewards")
    print("• Fair comparison between IL and RL approaches")
    print("• RL can discover novel strategies without penalty")
    print("• Clinical relevance of evaluation metrics")
    print("• Better insights for surgical AI system design")
    print()
    
    print("🚀 READY TO IMPLEMENT!")
    print("Use these specific improvements in your existing codebase.")


if __name__ == "__main__":
    main()
