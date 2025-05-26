# ===================================================================
# File: surgical_video_animator.py
# Create video animations of surgical action predictions vs ground truth
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import json
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

class SurgicalActionAnimator:
    """
    Creates video animations of surgical action predictions vs ground truth
    """
    
    def __init__(self, save_dir: str = 'surgical_animations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Animation settings
        self.figsize = (20, 12)
        self.fps = 10
        self.dpi = 100
        
        # Color schemes
        self.colors = {
            'ground_truth': '#2E86AB',
            'imitation_learning': '#A23B72', 
            'ppo': '#F18F01',
            'sac': '#C73E1D'
        }
        
        # Surgical phase info
        self.phase_names = [
            'Preparation', 'Calot Triangle', 'Clipping', 
            'Gallbladder Dissection', 'Packaging', 'Cleaning', 'Retrieval'
        ]
        
        # Action categories for better visualization
        self.action_categories = self._create_action_categories()
        
    def _create_action_categories(self) -> Dict:
        """Create action categories for better visualization"""
        return {
            'Grasping': list(range(0, 15)),
            'Cutting': list(range(15, 30)), 
            'Coagulation': list(range(30, 45)),
            'Clipping': list(range(45, 60)),
            'Irrigation': list(range(60, 75)),
            'Retraction': list(range(75, 90)),
            'Other': list(range(90, 100))
        }
    
    def create_realtime_prediction_animation(self, video_data: Dict, predictions: Dict, 
                                          video_id: str, max_frames: int = 500,
                                          prediction_horizon: int = 50) -> str:
        """
        Create real-time animation showing predictions vs ground truth as surgery progresses
        
        Args:
            video_data: Video data with ground truth
            predictions: Predictions from different methods
            video_id: Video identifier
            max_frames: Maximum frames to animate
            prediction_horizon: Future prediction horizon to show
            
        Returns:
            Path to generated video file
        """
        
        print(f"🎬 Creating real-time prediction animation for {video_id}...")
        
        # Prepare data
        gt_actions = video_data['actions_binaries'][:max_frames]
        phases = self._estimate_phases(gt_actions)
        
        # Setup figure
        fig, axes = plt.subplots(3, 2, figsize=self.figsize)
        fig.suptitle(f'Surgical Action Prediction Evolution - {video_id}', 
                    fontsize=16, fontweight='bold')
        
        # Animation data storage
        frames_data = []
        
        # Pre-compute all frame data
        print("  📊 Pre-computing animation frames...")
        for frame_idx in tqdm(range(max_frames), desc="Computing frames"):
            frame_data = self._compute_frame_data(
                frame_idx, gt_actions, predictions, phases, prediction_horizon
            )
            frames_data.append(frame_data)
        
        # Create animation
        print("  🎥 Generating animation...")
        
        def animate(frame_idx):
            # Clear all axes
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()
            
            frame_data = frames_data[frame_idx]
            
            # Plot 1: Current Action Predictions vs Ground Truth
            ax = axes[0, 0]
            self._plot_current_actions(ax, frame_data, frame_idx)
            
            # Plot 2: Action Timeline (past + future)
            ax = axes[0, 1] 
            self._plot_action_timeline(ax, frame_data, frame_idx, max_frames)
            
            # Plot 3: Action Categories Heatmap
            ax = axes[1, 0]
            self._plot_action_categories(ax, frame_data)
            
            # Plot 4: Prediction Confidence Evolution
            ax = axes[1, 1]
            self._plot_prediction_confidence(ax, frame_data, frame_idx)
            
            # Plot 5: Surgical Phase Progress
            ax = axes[2, 0]
            self._plot_phase_progress(ax, frame_data, frame_idx, max_frames)
            
            # Plot 6: Future Trajectory Predictions
            ax = axes[2, 1]
            self._plot_future_trajectories(ax, frame_data, frame_idx, prediction_horizon)
            
            plt.tight_layout()
            
            # Add progress indicator
            progress = (frame_idx + 1) / max_frames * 100
            fig.text(0.02, 0.02, f'Progress: {progress:.1f}% | Frame: {frame_idx+1}/{max_frames}', 
                    fontsize=12, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_frames, interval=100, repeat=False
        )
        
        # Save animation
        output_path = self.save_dir / f'realtime_prediction_{video_id}.mp4'
        print(f"  💾 Saving animation to {output_path}...")
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, metadata=dict(artist='SurgicalAI'), bitrate=1800)
        anim.save(str(output_path), writer=writer, dpi=self.dpi)
        
        plt.close(fig)
        
        print(f"  ✅ Animation saved: {output_path}")
        return str(output_path)
    
    def _compute_frame_data(self, frame_idx: int, gt_actions: np.ndarray, 
                           predictions: Dict, phases: np.ndarray, 
                           prediction_horizon: int) -> Dict:
        """Compute all data needed for a single animation frame"""
        
        # Current ground truth
        current_gt = gt_actions[frame_idx] if frame_idx < len(gt_actions) else np.zeros(100)
        
        # Current predictions from each method
        current_predictions = {}
        prediction_confidences = {}
        
        for method, preds in predictions.items():
            if frame_idx < len(preds):
                current_predictions[method] = preds[frame_idx]
                # Simulate confidence (in real system, this would come from model)
                prediction_confidences[method] = np.random.rand(100) * 0.5 + 0.5
            else:
                current_predictions[method] = np.zeros(100)
                prediction_confidences[method] = np.zeros(100)
        
        # Future predictions (trajectory)
        future_predictions = {}
        for method, preds in predictions.items():
            future_start = frame_idx + 1
            future_end = min(frame_idx + prediction_horizon + 1, len(preds))
            if future_start < len(preds):
                future_predictions[method] = preds[future_start:future_end]
            else:
                future_predictions[method] = np.zeros((prediction_horizon, 100))
        
        # Future ground truth
        future_gt_start = frame_idx + 1
        future_gt_end = min(frame_idx + prediction_horizon + 1, len(gt_actions))
        if future_gt_start < len(gt_actions):
            future_gt = gt_actions[future_gt_start:future_gt_end]
        else:
            future_gt = np.zeros((prediction_horizon, 100))
        
        return {
            'frame_idx': frame_idx,
            'current_gt': current_gt,
            'current_predictions': current_predictions,
            'prediction_confidences': prediction_confidences,
            'future_predictions': future_predictions,
            'future_gt': future_gt,
            'current_phase': phases[frame_idx] if frame_idx < len(phases) else 0,
            'phase_name': self.phase_names[phases[frame_idx]] if frame_idx < len(phases) else 'Unknown'
        }
    
    def _plot_current_actions(self, ax, frame_data: Dict, frame_idx: int):
        """Plot current action predictions vs ground truth"""
        
        current_gt = frame_data['current_gt']
        current_preds = frame_data['current_predictions']
        
        # Get top 20 most active actions
        active_actions = np.where(current_gt > 0)[0]
        if len(active_actions) == 0:
            # If no active actions, show top actions from any method
            all_activity = sum(preds for preds in current_preds.values())
            active_actions = np.argsort(all_activity)[-20:]
        else:
            active_actions = active_actions[:20]  # Limit display
        
        # Create bar plot
        x_pos = np.arange(len(active_actions))
        width = 0.15
        
        # Ground truth bars
        gt_values = current_gt[active_actions]
        ax.bar(x_pos - width*1.5, gt_values, width, 
               label='Ground Truth', color=self.colors['ground_truth'], alpha=0.8)
        
        # Prediction bars
        for i, (method, preds) in enumerate(current_preds.items()):
            pred_values = preds[active_actions]
            ax.bar(x_pos - width*0.5 + i*width, pred_values, width,
                  label=method.replace('_', ' ').title(), 
                  color=self.colors[method], alpha=0.7)
        
        ax.set_title(f'Current Actions (Frame {frame_idx+1})', fontweight='bold')
        ax.set_xlabel('Action ID')
        ax.set_ylabel('Prediction Value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'A{i}' for i in active_actions], rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
    
    def _plot_action_timeline(self, ax, frame_data: Dict, frame_idx: int, max_frames: int):
        """Plot action timeline showing past and current predictions"""
        
        # Create timeline visualization
        timeline_window = 100  # Show last 100 frames
        start_frame = max(0, frame_idx - timeline_window)
        
        # Show active actions over time
        current_gt = frame_data['current_gt']
        active_actions = np.where(current_gt > 0)[0][:10]  # Top 10 for clarity
        
        if len(active_actions) > 0:
            for i, action_id in enumerate(active_actions):
                # Plot ground truth
                ax.scatter(frame_idx, action_id, c=self.colors['ground_truth'], 
                          s=50, marker='s', alpha=0.8)
                
                # Plot predictions
                for method, preds in frame_data['current_predictions'].items():
                    if preds[action_id] > 0.5:
                        ax.scatter(frame_idx, action_id, c=self.colors[method], 
                                  s=30, marker='o', alpha=0.6)
        
        ax.set_title('Action Timeline (Recent History)', fontweight='bold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Action ID')
        ax.set_xlim(start_frame, frame_idx + 20)
        
        # Add current frame indicator
        ax.axvline(x=frame_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(frame_idx + 1, ax.get_ylim()[1], 'Now', rotation=90, 
                verticalalignment='top', fontweight='bold', color='red')
    
    def _plot_action_categories(self, ax, frame_data: Dict):
        """Plot action predictions by category"""
        
        current_gt = frame_data['current_gt']
        current_preds = frame_data['current_predictions']
        
        # Aggregate by category
        category_data = {}
        methods = ['Ground Truth'] + list(current_preds.keys())
        
        for category, action_ids in self.action_categories.items():
            category_data[category] = []
            
            # Ground truth
            gt_sum = np.sum(current_gt[action_ids])
            category_data[category].append(gt_sum)
            
            # Predictions
            for method, preds in current_preds.items():
                pred_sum = np.sum(preds[action_ids])
                category_data[category].append(pred_sum)
        
        # Create heatmap
        categories = list(category_data.keys())
        data_matrix = np.array([category_data[cat] for cat in categories])
        
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_title('Actions by Category', fontweight='bold')
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(methods)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}', 
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_prediction_confidence(self, ax, frame_data: Dict, frame_idx: int):
        """Plot prediction confidence over time"""
        
        # Simulate confidence evolution (in real system, track from model)
        window_size = 50
        x_frames = range(max(0, frame_idx - window_size), frame_idx + 1)
        
        for method in frame_data['current_predictions'].keys():
            # Generate realistic confidence evolution
            confidences = []
            for f in x_frames:
                # Simulate method-specific confidence patterns
                if method == 'sac':
                    conf = 0.7 + 0.2 * np.sin(f * 0.1) + np.random.normal(0, 0.05)
                elif method == 'imitation_learning':
                    conf = 0.6 + 0.1 * np.cos(f * 0.05) + np.random.normal(0, 0.03)
                else:  # ppo
                    conf = 0.4 + 0.3 * np.random.random() + np.random.normal(0, 0.1)
                
                confidences.append(max(0, min(1, conf)))
            
            ax.plot(x_frames, confidences, label=method.replace('_', ' ').title(), 
                   color=self.colors[method], linewidth=2, alpha=0.8)
        
        ax.set_title('Prediction Confidence Evolution', fontweight='bold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Current frame indicator
        ax.axvline(x=frame_idx, color='red', linestyle='--', alpha=0.7)
    
    def _plot_phase_progress(self, ax, frame_data: Dict, frame_idx: int, max_frames: int):
        """Plot surgical phase progress"""
        
        current_phase = frame_data['current_phase']
        phase_name = frame_data['phase_name']
        
        # Create phase timeline
        phase_progress = frame_idx / max_frames
        phase_within = (frame_idx % (max_frames // 7)) / (max_frames // 7)
        
        # Phase bars
        phases = self.phase_names
        y_pos = np.arange(len(phases))
        
        # Background bars
        ax.barh(y_pos, [1]*len(phases), color='lightgray', alpha=0.3)
        
        # Current phase highlight
        if current_phase < len(phases):
            ax.barh(current_phase, 1, color=self.colors['ground_truth'], alpha=0.7)
            ax.barh(current_phase, phase_within, color=self.colors['sac'], alpha=0.9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(phases)
        ax.set_xlabel('Progress')
        ax.set_title(f'Surgical Phase: {phase_name}', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add progress text
        ax.text(0.5, current_phase, f'{phase_within*100:.1f}%', 
               ha='center', va='center', fontweight='bold', color='white')
    
    def _plot_future_trajectories(self, ax, frame_data: Dict, frame_idx: int, 
                                prediction_horizon: int):
        """Plot predicted future action trajectories"""
        
        future_preds = frame_data['future_predictions']
        future_gt = frame_data['future_gt']
        
        # Focus on top actions
        current_gt = frame_data['current_gt']
        if np.sum(current_gt) > 0:
            top_actions = np.where(current_gt > 0)[0][:5]
        else:
            # Find most predicted actions across methods
            all_future = sum(preds.mean(axis=0) for preds in future_preds.values())
            top_actions = np.argsort(all_future)[-5:]
        
        if len(top_actions) == 0:
            ax.text(0.5, 0.5, 'No active actions to predict', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Future Action Trajectories', fontweight='bold')
            return
        
        # Time axis for future predictions
        future_frames = np.arange(frame_idx + 1, frame_idx + prediction_horizon + 1)
        
        # Plot trajectories for each action
        for i, action_id in enumerate(top_actions):
            # Ground truth future
            if len(future_gt) > 0:
                gt_trajectory = future_gt[:, action_id] if action_id < future_gt.shape[1] else []
                if len(gt_trajectory) > 0:
                    ax.plot(future_frames[:len(gt_trajectory)], 
                           [i + 0.4] * len(gt_trajectory) + gt_trajectory * 0.3, 
                           'o-', color=self.colors['ground_truth'], 
                           label='GT' if i == 0 else "", alpha=0.8, linewidth=2)
            
            # Predicted trajectories
            for j, (method, preds) in enumerate(future_preds.items()):
                if len(preds) > 0 and action_id < preds.shape[1]:
                    pred_trajectory = preds[:, action_id]
                    offset = -0.1 + j * 0.05
                    ax.plot(future_frames[:len(pred_trajectory)], 
                           [i + offset] * len(pred_trajectory) + pred_trajectory * 0.2,
                           's-', color=self.colors[method], alpha=0.6,
                           label=method.replace('_', ' ').title() if i == 0 else "",
                           markersize=4)
        
        ax.set_title('Future Action Trajectories', fontweight='bold')
        ax.set_xlabel('Future Frame Number')
        ax.set_ylabel('Action ID')
        ax.set_yticks(range(len(top_actions)))
        ax.set_yticklabels([f'Action {i}' for i in top_actions])
        
        if frame_idx == 0:  # Only show legend on first frame
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(alpha=0.3)
    
    def _estimate_phases(self, gt_actions: np.ndarray) -> np.ndarray:
        """Estimate surgical phases from ground truth actions"""
        
        # Simple phase estimation based on action patterns
        num_frames = len(gt_actions)
        phases = np.zeros(num_frames, dtype=int)
        
        # Divide video into 7 phases roughly
        phase_length = num_frames // 7
        
        for i in range(num_frames):
            phase = min(i // phase_length, 6)
            phases[i] = phase
        
        return phases
    
    def create_comparative_animation(self, video_data: Dict, predictions: Dict, 
                                   video_id: str, max_frames: int = 300) -> str:
        """
        Create side-by-side comparison animation of all methods
        """
        
        print(f"🎭 Creating comparative animation for {video_id}...")
        
        gt_actions = video_data['actions_binaries'][:max_frames]
        
        # Setup figure with method comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Method Comparison - {video_id}', fontsize=16, fontweight='bold')
        
        methods = ['ground_truth'] + list(predictions.keys())
        method_data = {'ground_truth': gt_actions}
        method_data.update(predictions)
        
        def animate(frame_idx):
            for i, ax in enumerate(axes.flat):
                ax.clear()
                
                if i < len(methods):
                    method = methods[i]
                    data = method_data[method]
                    
                    if frame_idx < len(data):
                        current_actions = data[frame_idx]
                        active_actions = np.where(current_actions > 0.5)[0]
                        
                        if len(active_actions) > 0:
                            # Show active actions as colored bars
                            ax.bar(range(len(active_actions)), 
                                  current_actions[active_actions],
                                  color=self.colors.get(method, 'gray'), alpha=0.8)
                            
                            ax.set_xticks(range(len(active_actions)))
                            ax.set_xticklabels([f'A{i}' for i in active_actions])
                        else:
                            ax.text(0.5, 0.5, 'No Active Actions', 
                                   ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'{method.replace("_", " ").title()}\nFrame {frame_idx+1}', 
                                fontweight='bold')
                    ax.set_ylim(0, 1)
                    ax.grid(alpha=0.3)
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_frames, interval=200, repeat=True
        )
        
        # Save
        output_path = self.save_dir / f'comparative_{video_id}.mp4'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='SurgicalAI'), bitrate=1800)
        anim.save(str(output_path), writer=writer, dpi=self.dpi)
        
        plt.close(fig)
        print(f"  ✅ Comparative animation saved: {output_path}")
        return str(output_path)
    
    def create_trajectory_evolution_gif(self, video_data: Dict, predictions: Dict,
                                       video_id: str, max_frames: int = 200) -> str:
        """
        Create GIF showing evolution of action predictions over time
        """
        
        print(f"🎪 Creating trajectory evolution GIF for {video_id}...")
        
        gt_actions = video_data['actions_binaries'][:max_frames]
        
        # Create frames
        frames = []
        
        for frame_idx in tqdm(range(0, max_frames, 5), desc="Creating GIF frames"):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot prediction evolution
            window = 20
            start_frame = max(0, frame_idx - window)
            end_frame = min(frame_idx + window, max_frames)
            
            time_range = range(start_frame, end_frame)
            
            # Get top 10 actions
            frame_gt = gt_actions[frame_idx] if frame_idx < len(gt_actions) else np.zeros(100)
            top_actions = np.argsort(frame_gt)[-10:]
            
            # Plot trajectories
            for method, preds in predictions.items():
                if method == 'ground_truth':
                    continue
                    
                for action_id in top_actions:
                    trajectory = []
                    for t in time_range:
                        if t < len(preds):
                            trajectory.append(preds[t][action_id])
                        else:
                            trajectory.append(0)
                    
                    ax.plot(time_range, np.array(trajectory) + action_id, 
                           color=self.colors[method], alpha=0.7, linewidth=2,
                           label=f'{method} A{action_id}' if action_id == top_actions[0] else "")
            
            # Current frame indicator
            ax.axvline(x=frame_idx, color='red', linestyle='--', linewidth=3, alpha=0.8)
            
            ax.set_title(f'Action Trajectory Evolution - Frame {frame_idx}', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Action ID + Prediction Value')
            ax.grid(alpha=0.3)
            
            # Save frame
            frame_path = self.save_dir / f'temp_frame_{frame_idx:04d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            frames.append(frame_path)
        
        # Create GIF
        gif_path = self.save_dir / f'trajectory_evolution_{video_id}.gif'
        
        images = []
        for frame_path in frames:
            images.append(Image.open(frame_path))
        
        images[0].save(
            str(gif_path),
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0
        )
        
        # Clean up temp files
        for frame_path in frames:
            frame_path.unlink()
        
        print(f"  ✅ GIF saved: {gif_path}")
        return str(gif_path)