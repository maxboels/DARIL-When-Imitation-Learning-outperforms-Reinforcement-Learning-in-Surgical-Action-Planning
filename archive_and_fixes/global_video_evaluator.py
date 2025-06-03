# ===================================================================
# File: global_video_evaluator.py
# Full video evaluation system for RL policies
# ===================================================================

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import json

class GlobalVideoEvaluator:
    """
    Evaluates RL policies across entire video sequences to get global action predictions
    """
    
    def __init__(self, world_model, env_config: Dict, device: str = 'cuda'):
        self.world_model = world_model
        self.env_config = env_config
        self.device = device
        
        # Import environment class
        from models import SurgicalWorldModelEnv
        self.env = SurgicalWorldModelEnv(world_model, env_config, device)
        
    def evaluate_policy_on_full_video(self, policy_model, video_data: Dict, 
                                     method_name: str, max_steps: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Evaluate RL policy on entire video sequence
        
        Args:
            policy_model: Trained RL policy (PPO/SAC)
            video_data: Video data from cholect50
            method_name: 'ppo' or 'sac'
            max_steps: Maximum steps to evaluate (None = full video)
            
        Returns:
            action_predictions: Array of predicted actions [num_frames, num_actions]
            evaluation_info: Dictionary with evaluation metadata
        """
        
        video_id = video_data['video_id']
        gt_embeddings = video_data['frame_embeddings']
        gt_actions = video_data['actions_binaries']
        
        if max_steps is None:
            max_steps = len(gt_embeddings)
        else:
            max_steps = min(max_steps, len(gt_embeddings))
        
        print(f"  🎬 Evaluating {method_name.upper()} on {video_id} ({max_steps} frames)")
        
        # Initialize environment with this specific video
        obs, _ = self.env.reset(options={'video_id': video_id})
        
        # Storage for predictions
        action_predictions = []
        episode_rewards = []
        episode_info = []
        
        # Evaluation loop
        for step in tqdm(range(max_steps), desc=f"{method_name} evaluation", leave=False):
            
            # Get action from policy
            try:
                action, _ = policy_model.predict(obs, deterministic=True)
                
                # Handle action space conversion for SAC
                if method_name.lower() == 'sac':
                    # SAC outputs continuous [0,1], convert to binary
                    action = (action > 0.5).astype(np.float32)
                
                # Ensure correct shape
                if len(action) != self.world_model.num_action_classes:
                    action = np.resize(action, self.world_model.num_action_classes)
                
                # Store predicted action
                action_predictions.append(action.copy())
                
                # Take step in environment
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_rewards.append(reward)
                episode_info.append({
                    'step': step,
                    'reward': reward,
                    'current_phase': info.get('current_phase', -1),
                    'reward_breakdown': info.get('reward_breakdown', {})
                })
                
                # If episode terminates early, pad with zeros
                if done or truncated:
                    remaining_steps = max_steps - step - 1
                    if remaining_steps > 0:
                        zero_actions = np.zeros((remaining_steps, self.world_model.num_action_classes))
                        action_predictions.extend(zero_actions)
                    break
                    
            except Exception as e:
                print(f"    ❌ Error at step {step}: {e}")
                # Fill with zeros for remaining steps
                remaining_steps = max_steps - step
                zero_actions = np.zeros((remaining_steps, self.world_model.num_action_classes))
                action_predictions.extend(zero_actions)
                break
        
        # Convert to numpy array
        action_predictions = np.array(action_predictions)
        
        # Ensure correct shape
        if len(action_predictions) < max_steps:
            padding = np.zeros((max_steps - len(action_predictions), self.world_model.num_action_classes))
            action_predictions = np.vstack([action_predictions, padding])
        
        evaluation_info = {
            'video_id': video_id,
            'method': method_name,
            'total_steps': len(episode_rewards),
            'total_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'final_phase': episode_info[-1]['current_phase'] if episode_info else -1,
            'episode_info': episode_info
        }
        
        return action_predictions, evaluation_info
    
    def evaluate_imitation_learning_on_full_video(self, video_data: Dict, 
                                                 stride: int = 1) -> Tuple[np.ndarray, Dict]:
        """
        Evaluate imitation learning (world model) on full video with proper stride
        
        Args:
            video_data: Video data from cholect50
            stride: Evaluate every nth frame (1 = every frame)
            
        Returns:
            action_predictions: Array of predicted actions
            evaluation_info: Dictionary with evaluation metadata
        """
        
        video_id = video_data['video_id']
        embeddings = video_data['frame_embeddings']
        
        print(f"  🧠 Evaluating Imitation Learning on {video_id} ({len(embeddings)} frames, stride={stride})")
        
        action_predictions = []
        
        # Evaluate frame by frame
        for i in tqdm(range(0, len(embeddings), stride), desc="IL evaluation", leave=False):
            try:
                frame_embedding = torch.tensor(
                    embeddings[i], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs = self.world_model.predict_next_action(frame_embedding)
                    
                    # Handle dimensions
                    if action_probs.dim() == 3:
                        action_probs = action_probs.squeeze(0).squeeze(0)
                    elif action_probs.dim() == 2:
                        action_probs = action_probs.squeeze(0)
                    
                    # Convert to binary
                    action_pred = (action_probs.cpu().numpy() > 0.5).astype(int)
                    
                    # If using stride > 1, duplicate prediction for intermediate frames
                    if stride > 1:
                        for _ in range(min(stride, len(embeddings) - i)):
                            action_predictions.append(action_pred)
                    else:
                        action_predictions.append(action_pred)
                    
            except Exception as e:
                print(f"    ❌ Error at frame {i}: {e}")
                # Fill with zeros
                zero_action = np.zeros(self.world_model.num_action_classes)
                if stride > 1:
                    for _ in range(min(stride, len(embeddings) - i)):
                        action_predictions.append(zero_action)
                else:
                    action_predictions.append(zero_action)
        
        # Ensure correct length
        while len(action_predictions) < len(embeddings):
            action_predictions.append(np.zeros(self.world_model.num_action_classes))
        
        # Trim if too long
        action_predictions = action_predictions[:len(embeddings)]
        
        action_predictions = np.array(action_predictions)
        
        evaluation_info = {
            'video_id': video_id,
            'method': 'imitation_learning',
            'total_frames': len(embeddings),
            'evaluation_stride': stride,
            'prediction_shape': action_predictions.shape
        }
        
        return action_predictions, evaluation_info

class EnhancedActionAnalyzer:
    """
    Enhanced action analyzer with global video evaluation
    """
    
    def __init__(self, save_dir: str = 'enhanced_action_analysis'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.global_predictions = {}
        self.ground_truth = {}
        self.evaluation_info = {}
        
    def collect_global_predictions(self, models: Dict, test_data: List[Dict], 
                                 world_model, env_config: Dict, device: str = 'cuda',
                                 max_frames_per_video: int = 1000) -> Dict:
        """
        Collect predictions from all models across entire videos
        
        Args:
            models: Dictionary of trained models
            test_data: Test video data
            world_model: World model for environment
            env_config: Environment configuration
            max_frames_per_video: Maximum frames to evaluate per video
            
        Returns:
            Global predictions for all methods
        """
        
        print("🌍 Collecting Global Action Predictions...")
        
        # Initialize global evaluator
        evaluator = GlobalVideoEvaluator(world_model, env_config, device)
        
        all_predictions = {}
        all_evaluation_info = {}
        
        for video in test_data:
            video_id = video['video_id']
            print(f"\n📹 Analyzing video: {video_id}")
            
            # Store ground truth
            gt_actions = video['actions_binaries']
            max_frames = min(len(gt_actions), max_frames_per_video)
            self.ground_truth[video_id] = gt_actions[:max_frames]
            
            video_predictions = {}
            video_info = {}
            
            # Evaluate each model
            for method_name, model in models.items():
                try:
                    if method_name.lower() == 'imitation_learning':
                        # Full video evaluation for imitation learning
                        predictions, info = evaluator.evaluate_imitation_learning_on_full_video(
                            video, stride=max(1, len(gt_actions) // max_frames)
                        )
                        predictions = predictions[:max_frames]
                        
                    elif method_name.lower() in ['ppo', 'sac']:
                        # Full video evaluation for RL policies
                        predictions, info = evaluator.evaluate_policy_on_full_video(
                            model, video, method_name, max_frames
                        )
                        
                    else:
                        print(f"    ⚠️  Unknown method: {method_name}")
                        continue
                    
                    video_predictions[method_name] = predictions
                    video_info[method_name] = info
                    
                    print(f"    ✅ {method_name}: {predictions.shape} predictions collected")
                    
                except Exception as e:
                    print(f"    ❌ Error with {method_name}: {e}")
                    # Fill with random predictions as fallback
                    fallback_predictions = np.random.rand(max_frames, world_model.num_action_classes) > 0.8
                    video_predictions[method_name] = fallback_predictions
                    video_info[method_name] = {'error': str(e)}
            
            all_predictions[video_id] = video_predictions
            all_evaluation_info[video_id] = video_info
        
        self.global_predictions = all_predictions
        self.evaluation_info = all_evaluation_info
        
        print("\n✅ Global prediction collection complete!")
        
        # Save results
        self.save_global_results()
        
        return all_predictions
    
    def create_enhanced_timeline_visualization(self, video_id: str, 
                                             save: bool = True) -> go.Figure:
        """
        Create enhanced timeline visualization with full video coverage
        """
        
        if video_id not in self.global_predictions:
            print(f"No global predictions found for video {video_id}")
            return None
        
        gt_actions = self.ground_truth[video_id]
        pred_actions = self.global_predictions[video_id]
        
        # Focus on most active actions (top 15)
        action_activity = np.sum(gt_actions, axis=0)
        top_actions = np.argsort(action_activity)[-15:]
        
        fig = make_subplots(
            rows=len(pred_actions) + 1, cols=1,
            subplot_titles=['Ground Truth'] + [m.replace('_', ' ').title() for m in pred_actions.keys()],
            vertical_spacing=0.02,
            shared_xaxes=True
        )
        
        colors = {
            'ground_truth': 'blue',
            'imitation_learning': 'green', 
            'ppo': 'red',
            'sac': 'orange'
        }
        
        # Ground truth timeline
        self._add_action_traces(fig, gt_actions, top_actions, 'blue', 1)
        
        # Prediction timelines
        for row_idx, (method, predictions) in enumerate(pred_actions.items(), 2):
            color = colors.get(method, 'gray')
            self._add_action_traces(fig, predictions, top_actions, color, row_idx)
        
        fig.update_layout(
            title=f'Enhanced Action Timeline - {video_id} (Full Video Coverage)',
            height=150 * (len(pred_actions) + 1),
            showlegend=False,
            xaxis_title="Frame Number"
        )
        
        # Add evaluation info as annotations
        if video_id in self.evaluation_info:
            annotations = []
            y_pos = 0.95
            for method, info in self.evaluation_info[video_id].items():
                if 'total_reward' in info:
                    text = f"{method}: Reward={info['total_reward']:.2f}, Steps={info['total_steps']}"
                    annotations.append(dict(
                        xref="paper", yref="paper",
                        x=0.02, y=y_pos,
                        text=text,
                        showarrow=False,
                        font=dict(size=10)
                    ))
                    y_pos -= 0.05
            
            fig.update_layout(annotations=annotations)
        
        if save:
            fig.write_html(self.save_dir / f'enhanced_timeline_{video_id}.html')
            fig.write_image(self.save_dir / f'enhanced_timeline_{video_id}.png', 
                           width=1400, height=800)
        
        return fig
    
    def _add_action_traces(self, fig, actions, top_actions, color, row):
        """Helper function to add action traces to timeline plot"""
        for action_idx in top_actions:
            frames_with_action = np.where(actions[:, action_idx] == 1)[0]
            
            if len(frames_with_action) > 0:
                # Create segments for continuous actions
                segments = self._get_action_segments(frames_with_action)
                
                for start, end in segments:
                    fig.add_trace(
                        go.Scatter(
                            x=[start, end],
                            y=[action_idx, action_idx],
                            mode='lines',
                            line=dict(color=color, width=3),
                            showlegend=False,
                            hovertemplate=f'Action {action_idx}<br>Frames {start}-{end}<extra></extra>'
                        ),
                        row=row, col=1
                    )
    
    def _get_action_segments(self, frames):
        """Convert frame indices to continuous segments"""
        if len(frames) == 0:
            return []
        
        segments = []
        start = frames[0]
        
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] > 1:  # Gap found
                segments.append((start, frames[i-1]))
                start = frames[i]
        
        segments.append((start, frames[-1]))
        return segments
    
    def create_coverage_analysis(self, save: bool = True) -> plt.Figure:
        """
        Analyze temporal coverage of predictions across videos
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect coverage data
        coverage_data = {}
        
        for video_id in self.global_predictions:
            gt_length = len(self.ground_truth[video_id])
            
            for method, predictions in self.global_predictions[video_id].items():
                if method not in coverage_data:
                    coverage_data[method] = {
                        'temporal_coverage': [],
                        'action_density': [],
                        'prediction_lengths': []
                    }
                
                # Calculate temporal coverage
                non_zero_frames = np.sum(predictions, axis=1) > 0
                coverage_pct = np.sum(non_zero_frames) / len(non_zero_frames) * 100
                coverage_data[method]['temporal_coverage'].append(coverage_pct)
                
                # Calculate action density
                avg_actions_per_frame = np.mean(np.sum(predictions, axis=1))
                coverage_data[method]['action_density'].append(avg_actions_per_frame)
                
                # Prediction length
                coverage_data[method]['prediction_lengths'].append(len(predictions))
        
        # Plot 1: Temporal Coverage
        ax = axes[0, 0]
        methods = list(coverage_data.keys())
        coverage_means = [np.mean(coverage_data[m]['temporal_coverage']) for m in methods]
        coverage_stds = [np.std(coverage_data[m]['temporal_coverage']) for m in methods]
        
        colors = {'imitation_learning': 'green', 'ppo': 'red', 'sac': 'orange'}
        bar_colors = [colors.get(m, 'gray') for m in methods]
        
        bars = ax.bar(methods, coverage_means, yerr=coverage_stds, 
                     color=bar_colors, alpha=0.7, capsize=5)
        ax.set_title('Temporal Coverage (% of video with predictions)', fontweight='bold')
        ax.set_ylabel('Coverage (%)')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for bar, mean in zip(bars, coverage_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Action Density
        ax = axes[0, 1]
        density_means = [np.mean(coverage_data[m]['action_density']) for m in methods]
        density_stds = [np.std(coverage_data[m]['action_density']) for m in methods]
        
        bars = ax.bar(methods, density_means, yerr=density_stds,
                     color=bar_colors, alpha=0.7, capsize=5)
        ax.set_title('Action Density (avg actions per frame)', fontweight='bold')
        ax.set_ylabel('Actions per Frame')
        
        for bar, mean in zip(bars, density_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Coverage Distribution
        ax = axes[1, 0]
        for method in methods:
            coverage_values = coverage_data[method]['temporal_coverage']
            ax.hist(coverage_values, alpha=0.6, label=method.replace('_', ' ').title(),
                   color=colors.get(method, 'gray'), bins=10)
        
        ax.set_title('Coverage Distribution Across Videos', fontweight='bold')
        ax.set_xlabel('Temporal Coverage (%)')
        ax.set_ylabel('Number of Videos')
        ax.legend()
        
        # Plot 4: Method Comparison Summary
        ax = axes[1, 1]
        
        # Create summary statistics
        summary_data = []
        for method in methods:
            coverage = np.mean(coverage_data[method]['temporal_coverage'])
            density = np.mean(coverage_data[method]['action_density'])
            summary_data.append([method.replace('_', ' ').title(), f'{coverage:.1f}%', f'{density:.2f}'])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Method', 'Avg Coverage', 'Avg Density'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(summary_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Summary Statistics', fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'coverage_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_global_results(self):
        """Save global evaluation results"""
        
        # Save predictions
        results = {
            'global_predictions': {k: {method: pred.tolist() for method, pred in v.items()} 
                                 for k, v in self.global_predictions.items()},
            'ground_truth': {k: v.tolist() for k, v in self.ground_truth.items()},
            'evaluation_info': self.evaluation_info
        }
        
        with open(self.save_dir / 'global_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Global results saved to: {self.save_dir}")
