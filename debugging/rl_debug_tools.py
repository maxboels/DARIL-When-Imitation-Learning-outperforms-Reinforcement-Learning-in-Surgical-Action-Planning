#!/usr/bin/env python3
"""
RL Debugging and Monitoring Tools
Comprehensive debugging for surgical RL training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

class RLDebugger:
    """Comprehensive RL debugging and monitoring system."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Metrics tracking
        self.episode_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        self.action_distributions = []
        self.reward_components = defaultdict(list)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'rl_debug.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_episode(self, episode_info: Dict[str, Any]):
        """Log episode-level metrics."""
        for key, value in episode_info.items():
            if isinstance(value, (int, float)):
                self.episode_metrics[key].append(value)
        
        # Log every 10 episodes
        if len(self.episode_metrics.get('episode_reward', [])) % 10 == 0:
            self._print_episode_summary()
    
    def log_step(self, step_info: Dict[str, Any]):
        """Log step-level metrics."""
        for key, value in step_info.items():
            if isinstance(value, (int, float)):
                self.step_metrics[key].append(value)
    
    def log_action_distribution(self, actions: np.ndarray, expert_actions: Optional[np.ndarray] = None):
        """Log action distribution analysis."""
        action_stats = {
            'mean_actions_per_step': np.mean(np.sum(actions > 0.5, axis=-1)),
            'action_sparsity': 1 - np.mean(actions > 0.5),
            'max_prob': np.max(actions),
            'min_prob': np.min(actions),
            'std_prob': np.std(actions)
        }
        
        if expert_actions is not None:
            # Compare with expert actions
            action_stats.update({
                'expert_action_count': np.mean(np.sum(expert_actions, axis=-1)),
                'expert_sparsity': 1 - np.mean(expert_actions),
                'action_overlap': np.mean((actions > 0.5) & expert_actions)
            })
        
        self.action_distributions.append(action_stats)
    
    def log_reward_components(self, reward_components: Dict[str, float]):
        """Log individual reward components."""
        for component, value in reward_components.items():
            self.reward_components[component].append(value)
    
    def _print_episode_summary(self):
        """Print episode summary."""
        if not self.episode_metrics['episode_reward']:
            return
        
        recent_rewards = self.episode_metrics['episode_reward'][-10:]
        recent_lengths = self.episode_metrics.get('episode_length', [])[-10:]
        
        self.logger.info(f"=== Episode Summary (Last 10 episodes) ===")
        self.logger.info(f"Mean Reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}")
        if recent_lengths:
            self.logger.info(f"Mean Length: {np.mean(recent_lengths):.1f}")
        
        # Check for learning progress
        if len(self.episode_metrics['episode_reward']) >= 20:
            early_rewards = self.episode_metrics['episode_reward'][-20:-10]
            recent_rewards = self.episode_metrics['episode_reward'][-10:]
            improvement = np.mean(recent_rewards) - np.mean(early_rewards)
            self.logger.info(f"Improvement: {improvement:+.3f}")
            
            if improvement < 0.01:
                self.logger.warning("⚠️  Learning may have plateaued!")
    
    def plot_training_curves(self):
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Episode rewards
        if 'episode_reward' in self.episode_metrics:
            rewards = self.episode_metrics['episode_reward']
            axes[0, 0].plot(rewards, alpha=0.6, label='Episode Reward')
            axes[0, 0].plot(pd.Series(rewards).rolling(10).mean(), label='10-episode MA')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if 'episode_length' in self.episode_metrics:
            lengths = self.episode_metrics['episode_length']
            axes[0, 1].plot(lengths, alpha=0.6, label='Episode Length')
            axes[0, 1].plot(pd.Series(lengths).rolling(10).mean(), label='10-episode MA')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Action distribution over time
        if self.action_distributions:
            action_counts = [ad['mean_actions_per_step'] for ad in self.action_distributions]
            expert_counts = [ad.get('expert_action_count', 0) for ad in self.action_distributions]
            
            axes[0, 2].plot(action_counts, label='Predicted Actions', alpha=0.7)
            if any(expert_counts):
                axes[0, 2].plot(expert_counts, label='Expert Actions', alpha=0.7)
            axes[0, 2].set_title('Actions per Step')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Average Actions')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Reward components
        if self.reward_components:
            for component, values in list(self.reward_components.items())[:3]:  # Plot top 3
                axes[1, 0].plot(values, label=component, alpha=0.7)
            axes[1, 0].set_title('Reward Components')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Reward Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Action overlap with expert (if available)
        if self.action_distributions and any('action_overlap' in ad for ad in self.action_distributions):
            overlaps = [ad.get('action_overlap', 0) for ad in self.action_distributions]
            axes[1, 1].plot(overlaps, alpha=0.7)
            axes[1, 1].set_title('Action Overlap with Expert')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Overlap Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate / exploration (if tracked)
        if 'learning_rate' in self.step_metrics:
            lr_values = self.step_metrics['learning_rate']
            axes[1, 2].plot(lr_values, alpha=0.7)
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('LR')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'rl_training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Training curves saved to {plot_path}")
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze if RL is converging properly."""
        analysis = {}
        
        if 'episode_reward' in self.episode_metrics:
            rewards = np.array(self.episode_metrics['episode_reward'])
            
            if len(rewards) >= 50:
                # Check if learning is happening
                early_rewards = rewards[:len(rewards)//2]
                late_rewards = rewards[len(rewards)//2:]
                
                analysis['reward_improvement'] = np.mean(late_rewards) - np.mean(early_rewards)
                analysis['reward_trend'] = 'improving' if analysis['reward_improvement'] > 0.1 else 'plateaued'
                
                # Check variance (high variance might indicate instability)
                analysis['reward_std_early'] = np.std(early_rewards)
                analysis['reward_std_late'] = np.std(late_rewards)
                analysis['stability'] = 'stable' if analysis['reward_std_late'] < analysis['reward_std_early'] else 'unstable'
                
                # Check for oscillations
                recent_rewards = rewards[-20:] if len(rewards) >= 20 else rewards
                analysis['oscillation_score'] = np.mean(np.abs(np.diff(recent_rewards)))
        
        # Action distribution analysis
        if self.action_distributions:
            recent_distributions = self.action_distributions[-10:]
            
            action_counts = [ad['mean_actions_per_step'] for ad in recent_distributions]
            analysis['action_consistency'] = 1.0 / (1.0 + np.std(action_counts))
            
            if all('action_overlap' in ad for ad in recent_distributions):
                overlaps = [ad['action_overlap'] for ad in recent_distributions]
                analysis['expert_alignment'] = np.mean(overlaps)
        
        return analysis
    
    def generate_debug_report(self) -> str:
        """Generate comprehensive debug report."""
        analysis = self.analyze_convergence()
        
        report = f"""
# RL Training Debug Report

## Training Progress
- Episodes completed: {len(self.episode_metrics.get('episode_reward', []))}
- Steps completed: {len(self.step_metrics.get('step_reward', []))}

## Convergence Analysis
"""
        
        if 'reward_improvement' in analysis:
            report += f"""
### Reward Learning
- Reward improvement: {analysis['reward_improvement']:.3f}
- Trend: {analysis['reward_trend']}
- Stability: {analysis['stability']}
- Recent oscillation: {analysis.get('oscillation_score', 0):.3f}
"""
        
        if 'action_consistency' in analysis:
            report += f"""
### Action Learning
- Action consistency: {analysis['action_consistency']:.3f}
- Expert alignment: {analysis.get('expert_alignment', 0):.3f}
"""
        
        # Recommendations
        report += f"""
## Recommendations

"""
        
        if analysis.get('reward_trend') == 'plateaued':
            report += "- ⚠️ Learning has plateaued. Consider:\n"
            report += "  - Adjusting learning rate\n"
            report += "  - Improving reward function\n"
            report += "  - Adding exploration noise\n"
        
        if analysis.get('stability') == 'unstable':
            report += "- ⚠️ Training is unstable. Consider:\n"
            report += "  - Reducing learning rate\n"
            report += "  - Using gradient clipping\n"
            report += "  - Increasing batch size\n"
        
        if analysis.get('expert_alignment', 1.0) < 0.3:
            report += "- ⚠️ Poor alignment with expert. Consider:\n"
            report += "  - Improving imitation rewards\n"
            report += "  - Adding behavioral cloning pre-training\n"
            report += "  - Checking action space alignment\n"
        
        # Save report
        report_path = self.save_dir / 'debug_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📋 Debug report saved to {report_path}")
        return report


class FixedRewardWorldModelEnv:
    """
    Fixed version of World Model Environment with proper reward design.
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Better reward weights
        self.reward_weights = {
            'action_accuracy': 5.0,      # High weight for matching expert actions
            'action_sparsity': 1.0,      # Reward appropriate action density
            'state_consistency': 2.0,    # Reward consistent state evolution
            'completion_bonus': 10.0,    # Episode completion
            'safety_penalty': -3.0       # Penalty for dangerous combinations
        }
        
        # Track expert actions for comparison
        self.current_expert_actions = None
    
    def _calculate_improved_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
        """Calculate improved reward that actually drives learning."""
        reward = 0.0
        
        # 1. Action accuracy reward (most important)
        if self.current_expert_actions is not None:
            binary_action = (action > 0.5).astype(int)
            expert_binary = self.current_expert_actions
            
            # Accuracy reward
            accuracy = np.mean(binary_action == expert_binary)
            reward += self.reward_weights['action_accuracy'] * accuracy
            
            # F1-based reward for sparse actions
            if np.sum(expert_binary) > 0:
                tp = np.sum((binary_action == 1) & (expert_binary == 1))
                fp = np.sum((binary_action == 1) & (expert_binary == 0))
                fn = np.sum((binary_action == 0) & (expert_binary == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                reward += self.reward_weights['action_accuracy'] * f1
        
        # 2. Action sparsity reward (surgical actions should be sparse)
        action_count = np.sum(action > 0.5)
        if 1 <= action_count <= 5:  # Reasonable range
            reward += self.reward_weights['action_sparsity']
        elif action_count == 0:
            reward += self.reward_weights['safety_penalty'] * 0.5  # Mild penalty for no action
        elif action_count > 10:
            reward += self.reward_weights['safety_penalty'] * 0.3  # Penalty for too many actions
        
        # 3. Use world model predicted rewards (but with lower weight)
        world_model_reward = sum(predicted_rewards.values()) * 0.1
        reward += world_model_reward
        
        # 4. Exploration bonus
        action_entropy = -np.sum(action * np.log(action + 1e-8) + (1-action) * np.log(1-action + 1e-8))
        reward += 0.1 * action_entropy / 100  # Small exploration bonus
        
        return np.clip(reward, -5.0, 15.0)


class EnhancedRLTrainer:
    """
    Enhanced RL trainer with proper debugging and monitoring.
    """
    
    def __init__(self, model, config: Dict, logger, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Initialize debugger
        self.debugger = RLDebugger(str(Path(logger.log_dir) / 'rl_debug'))
        
    def train_with_debugging(self, train_data: List[Dict], timesteps: int = 10000):
        """Train RL with comprehensive debugging."""
        
        self.logger.info("🚀 Starting Enhanced RL Training with Debugging")
        
        # Create environment with debugger integration
        env = self._create_monitored_env(train_data)
        
        # Create RL algorithm with better hyperparameters
        algorithm = self._create_optimized_algorithm(env)
        
        # Custom callback for debugging
        debug_callback = RLDebugCallback(self.debugger)
        
        # Train with monitoring
        algorithm.learn(
            total_timesteps=timesteps,
            callback=debug_callback,
            progress_bar=True
        )
        
        # Generate final debug report
        self.debugger.plot_training_curves()
        debug_report = self.debugger.generate_debug_report()
        
        return algorithm, debug_report
    
    def _create_optimized_algorithm(self, env):
        """Create RL algorithm with optimized hyperparameters for surgical tasks."""
        from stable_baselines3 import PPO
        
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # Lower learning rate for stability
            n_steps=512,         # More steps for better gradient estimates
            batch_size=64,       # Larger batch size
            n_epochs=10,         # More epochs per update
            gamma=0.95,          # Slightly lower gamma for immediate rewards
            gae_lambda=0.9,
            clip_range=0.1,      # Lower clip range for stability
            ent_coef=0.05,       # Higher entropy for exploration
            vf_coef=0.5,
            verbose=1,
            device='cpu',
            policy_kwargs={
                'net_arch': [256, 256, 128],  # Larger network
                'activation_fn': torch.nn.ReLU
            }
        )


class RLDebugCallback:
    """Callback for monitoring RL training."""
    
    def __init__(self, debugger: RLDebugger):
        self.debugger = debugger
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # This would be called by the RL algorithm
        # Log step-level metrics here
        return True
    
    def _on_episode_end(self, episode_info: Dict):
        self.episode_count += 1
        self.debugger.log_episode(episode_info)
        
        # Plot curves every 50 episodes
        if self.episode_count % 50 == 0:
            self.debugger.plot_training_curves()


# Example usage
if __name__ == "__main__":
    print("🔧 RL DEBUGGING TOOLS")
    print("=" * 50)
    print("✅ Comprehensive RL monitoring and debugging")
    print("✅ Convergence analysis")
    print("✅ Action distribution tracking")
    print("✅ Reward component analysis")
    print("✅ Training curve visualization")
    print("✅ Automated debug reports")
