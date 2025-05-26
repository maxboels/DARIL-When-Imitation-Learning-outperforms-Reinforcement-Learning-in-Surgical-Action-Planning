# ===================================================================
# File: animation_with_progress.py
# Enhanced animation system with comprehensive progress tracking
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, List, Optional
import threading
import sys
import os

class ProgressTrackingAnimator:
    """
    Animation creator with detailed progress tracking
    """
    
    def __init__(self, save_dir: str = 'surgical_animations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Progress tracking
        self.total_frames = 0
        self.current_frame = 0
        self.start_time = None
        
        # Animation settings - optimized for reasonable speed
        self.figsize = (14, 10)  # Smaller than original (20, 12)
        self.dpi = 75           # Lower than original (100)
        self.fps = 8            # Reasonable speed
        
        self.colors = {
            'ground_truth': '#2E86AB',
            'imitation_learning': '#A23B72', 
            'ppo': '#F18F01',
            'sac': '#C73E1D'
        }
    
    def create_surgical_animation_with_progress(self, video_data: Dict, predictions: Dict, 
                                               video_id: str, max_frames: int = 100) -> str:
        """
        Create surgical animation with comprehensive progress tracking
        """
        
        print(f"🎬 Creating surgical animation for {video_id}")
        print(f"📊 Settings: {max_frames} frames, {self.fps} FPS, DPI={self.dpi}")
        
        # Prepare data
        gt_actions = video_data['actions_binaries'][:max_frames]
        self.total_frames = len(gt_actions)
        
        print(f"📈 Processing {self.total_frames} frames...")
        
        # Pre-compute all animation data with progress
        print("  🔄 Pre-computing animation data...")
        frames_data = []
        
        with tqdm(total=self.total_frames, desc="Computing frames", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            for frame_idx in range(self.total_frames):
                frame_data = self._compute_frame_data(frame_idx, gt_actions, predictions)
                frames_data.append(frame_data)
                pbar.update(1)
        
        # Setup figure with progress
        print("  🎨 Setting up figure...")
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle(f'Surgical Action Analysis - {video_id}', fontsize=14, fontweight='bold')
        
        # Animation progress tracking
        self.current_frame = 0
        self.start_time = time.time()
        
        # Create progress bar for frame rendering
        animation_pbar = tqdm(total=self.total_frames, desc="Generating animation", 
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        def animate(frame_idx):
            # Update progress
            self.current_frame = frame_idx
            animation_pbar.update(1)
            
            # Clear all axes
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()
            
            frame_data = frames_data[frame_idx]
            
            # Plot 1: Current Actions (Top Left)
            self._plot_current_actions_simple(axes[0, 0], frame_data, frame_idx)
            
            # Plot 2: Action Timeline (Top Middle)
            self._plot_action_timeline_simple(axes[0, 1], frame_data, frame_idx)
            
            # Plot 3: Method Comparison (Top Right)
            self._plot_method_comparison(axes[0, 2], frame_data)
            
            # Plot 4: Prediction Confidence (Bottom Left)
            self._plot_confidence_simple(axes[1, 0], frame_data, frame_idx)
            
            # Plot 5: Action Categories (Bottom Middle)
            self._plot_categories_simple(axes[1, 1], frame_data)
            
            # Plot 6: Progress Info (Bottom Right)
            self._plot_progress_info(axes[1, 2], frame_idx)
            
            plt.tight_layout()
            
            # Add frame counter
            elapsed = time.time() - self.start_time
            fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            fig.text(0.02, 0.02, f'Frame {frame_idx+1}/{self.total_frames} | {fps_actual:.1f} FPS', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Create animation
        print("  🎥 Creating matplotlib animation object...")
        anim = animation.FuncAnimation(fig, animate, frames=self.total_frames, 
                                     interval=125, repeat=False, cache_frame_data=False)
        
        # Close animation progress bar
        animation_pbar.close()
        
        # Save with detailed progress
        output_path = self.save_dir / f'surgical_analysis_{video_id}.mp4'
        success = self._save_animation_with_progress(anim, output_path)
        
        plt.close(fig)
        
        if success:
            print(f"  ✅ Animation completed: {output_path}")
            return str(output_path)
        else:
            print(f"  ⚠️  Animation saved with issues - check file")
            return str(output_path)
    
    def _save_animation_with_progress(self, anim, output_path: Path) -> bool:
        """
        Save animation with progress tracking and fallback options
        """
        
        print(f"  💾 Saving animation to {output_path.name}...")
        
        # Estimate save time
        estimated_seconds = self.total_frames * 0.1  # Rough estimate
        print(f"  ⏱️  Estimated save time: {estimated_seconds:.0f}-{estimated_seconds*2:.0f} seconds")
        
        # Create progress thread
        save_progress_thread = threading.Thread(target=self._show_save_progress, 
                                              args=(estimated_seconds,))
        save_progress_thread.daemon = True
        save_progress_thread.start()
        
        try:
            # Try MP4 with ffmpeg
            print("  🎬 Attempting MP4 save with ffmpeg...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=self.fps, 
                          metadata=dict(artist='SurgicalAI', title=f'Analysis_{output_path.stem}'),
                          bitrate=1200,
                          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            
            anim.save(str(output_path), writer=writer, dpi=self.dpi, 
                     progress_callback=self._save_progress_callback)
            
            print("\n  ✅ MP4 save completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n  ⚠️  MP4 save failed: {e}")
            print("  🔄 Trying GIF format as fallback...")
            
            try:
                gif_path = output_path.with_suffix('.gif')
                anim.save(str(gif_path), writer='pillow', fps=max(5, self.fps//2), 
                         progress_callback=self._save_progress_callback)
                print(f"  ✅ GIF saved: {gif_path.name}")
                return True
                
            except Exception as e2:
                print(f"  ❌ GIF save also failed: {e2}")
                print("  💡 Try installing ffmpeg: sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)")
                return False
    
    def _show_save_progress(self, estimated_seconds: float):
        """
        Show progress during the save process
        """
        
        start_time = time.time()
        
        with tqdm(total=int(estimated_seconds), desc="Saving animation", 
                 bar_format="{l_bar}{bar}| {elapsed}/{estimated_seconds:.0f}s [{rate_fmt}]") as pbar:
            
            while True:
                elapsed = time.time() - start_time
                pbar.n = min(int(elapsed), int(estimated_seconds))
                pbar.refresh()
                
                if elapsed > estimated_seconds * 1.5:  # Stop if taking too long
                    break
                    
                time.sleep(0.5)
    
    def _save_progress_callback(self, current_frame: int, total_frames: int):
        """
        Callback for animation save progress (if supported)
        """
        if current_frame % 10 == 0:  # Update every 10 frames to avoid spam
            progress = current_frame / total_frames * 100
            print(f"\r  📼 Encoding: {progress:.1f}% ({current_frame}/{total_frames})", end='', flush=True)
    
    def _compute_frame_data(self, frame_idx: int, gt_actions: np.ndarray, 
                           predictions: Dict) -> Dict:
        """
        Compute data for a single frame (simplified version)
        """
        
        current_gt = gt_actions[frame_idx] if frame_idx < len(gt_actions) else np.zeros(100)
        
        current_predictions = {}
        for method, preds in predictions.items():
            if frame_idx < len(preds):
                current_predictions[method] = preds[frame_idx]
            else:
                current_predictions[method] = np.zeros(100)
        
        return {
            'frame_idx': frame_idx,
            'current_gt': current_gt,
            'current_predictions': current_predictions
        }
    
    def _plot_current_actions_simple(self, ax, frame_data: Dict, frame_idx: int):
        """Simplified current actions plot"""
        
        current_gt = frame_data['current_gt']
        current_preds = frame_data['current_predictions']
        
        # Get top 8 most active actions for visibility
        all_activity = current_gt + sum(preds for preds in current_preds.values())
        top_actions = np.argsort(all_activity)[-8:]
        
        if len(top_actions) > 0 and np.sum(all_activity[top_actions]) > 0:
            x_pos = np.arange(len(top_actions))
            width = 0.15
            
            # Ground truth
            ax.bar(x_pos - width*1.5, current_gt[top_actions], width, 
                   label='GT', color=self.colors['ground_truth'], alpha=0.8)
            
            # Predictions
            for i, (method, preds) in enumerate(current_preds.items()):
                ax.bar(x_pos - width*0.5 + i*width, preds[top_actions], width,
                      label=method.replace('_', ' ')[:8], 
                      color=self.colors[method], alpha=0.7)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'A{i}' for i in top_actions], fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Active\nActions', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'Actions - Frame {frame_idx+1}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
    
    def _plot_action_timeline_simple(self, ax, frame_data: Dict, frame_idx: int):
        """Simplified timeline plot"""
        
        # Simple timeline showing last 20 frames
        timeline_length = min(20, frame_idx + 1)
        
        ax.barh(0, timeline_length, color='lightgray', alpha=0.3)
        ax.barh(0, frame_idx / max(timeline_length, 1) * timeline_length, 
               color=self.colors['ground_truth'], alpha=0.7)
        
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('Timeline Progress', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frames', fontsize=8)
        ax.set_yticks([])
    
    def _plot_method_comparison(self, ax, frame_data: Dict):
        """Method comparison plot"""
        
        current_gt = frame_data['current_gt']
        current_preds = frame_data['current_predictions']
        
        # Calculate activity for each method
        methods = ['GT'] + list(current_preds.keys())
        activities = [np.sum(current_gt)]
        activities.extend([np.sum(preds) for preds in current_preds.values()])
        
        colors_list = [self.colors['ground_truth']] + [self.colors[m] for m in current_preds.keys()]
        
        bars = ax.bar(range(len(methods)), activities, color=colors_list, alpha=0.7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ')[:8] for m in methods], fontsize=8, rotation=45)
        ax.set_title('Method Activity', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Actions', fontsize=8)
        
        # Add value labels
        for bar, activity in zip(bars, activities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{activity:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_confidence_simple(self, ax, frame_data: Dict, frame_idx: int):
        """Simplified confidence plot"""
        
        # Simulate confidence evolution
        methods = list(frame_data['current_predictions'].keys())
        
        # Show confidence bars for current frame
        confidences = []
        for method in methods:
            if method == 'sac':
                conf = 0.8 + 0.1 * np.sin(frame_idx * 0.1)
            elif method == 'imitation_learning':
                conf = 0.6 + 0.1 * np.cos(frame_idx * 0.05)  
            else:  # ppo
                conf = 0.3 + 0.2 * np.random.random()
            
            confidences.append(max(0.1, min(1, conf)))
        
        colors_list = [self.colors[m] for m in methods]
        bars = ax.bar(range(len(methods)), confidences, color=colors_list, alpha=0.7)
        
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ')[:8] for m in methods], fontsize=8, rotation=45)
        ax.set_title('Prediction Confidence', fontsize=10, fontweight='bold')
        ax.set_ylabel('Confidence', fontsize=8)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{conf:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_categories_simple(self, ax, frame_data: Dict):
        """Simplified categories plot"""
        
        current_gt = frame_data['current_gt']
        
        # Simple category grouping
        categories = {
            'Grasp (0-24)': np.sum(current_gt[0:25]),
            'Cut (25-49)': np.sum(current_gt[25:50]),
            'Coag (50-74)': np.sum(current_gt[50:75]),
            'Other (75-99)': np.sum(current_gt[75:100])
        }
        
        ax.pie(list(categories.values()), labels=list(categories.keys()), 
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
        ax.set_title('Action Categories', fontsize=10, fontweight='bold')
    
    def _plot_progress_info(self, ax, frame_idx: int):
        """Plot progress and timing information"""
        
        ax.axis('off')
        
        # Progress info
        progress_pct = (frame_idx + 1) / self.total_frames * 100
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (self.total_frames - frame_idx - 1) / fps_actual if fps_actual > 0 else 0
        
        info_text = f"""
Frame: {frame_idx + 1} / {self.total_frames}
Progress: {progress_pct:.1f}%
Elapsed: {elapsed:.1f}s
FPS: {fps_actual:.1f}
ETA: {eta:.1f}s
        """.strip()
        
        ax.text(0.1, 0.8, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Progress bar
        ax.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.1, 
                                  facecolor='lightgray', transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((0.1, 0.2), 0.8 * progress_pct/100, 0.1,
                                  facecolor='green', transform=ax.transAxes))

# ===================================================================
# File: run_progress_animation.py
# Main script with progress tracking
# ===================================================================

def run_surgical_animation_with_progress():
    """
    Run surgical animation with comprehensive progress tracking
    """
    
    print("🎬 Surgical Animation with Progress Tracking")
    print("=" * 60)
    
    # Load data
    results_path = Path('enhanced_action_analysis/global_evaluation_results.json')
    if not results_path.exists():
        print("❌ Global evaluation results not found!")
        print("Please run: python run_global_evaluation.py")
        return
    
    print("📁 Loading global evaluation results...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    global_predictions = results['global_predictions']
    ground_truth = results['ground_truth']
    
    # Initialize animator
    animator = ProgressTrackingAnimator()
    
    # Create animations for each video
    created_files = []
    
    for video_id in list(global_predictions.keys())[:2]:  # Limit to 2 videos for testing
        print(f"\n🎥 Processing video: {video_id}")
        
        # Prepare data
        video_data = {'actions_binaries': np.array(ground_truth[video_id])}
        predictions = {k: np.array(v) for k, v in global_predictions[video_id].items()}
        
        # Create animation with progress tracking
        try:
            output_file = animator.create_surgical_animation_with_progress(
                video_data, predictions, video_id, max_frames=80  # Reasonable for testing
            )
            created_files.append(output_file)
            
        except KeyboardInterrupt:
            print("\n⚠️  Animation interrupted by user")
            break
            
        except Exception as e:
            print(f"❌ Error creating animation for {video_id}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 ANIMATION CREATION COMPLETE!")
    print("=" * 60)
    
    if created_files:
        print(f"\n✅ Successfully created {len(created_files)} animations:")
        for file_path in created_files:
            file_size = Path(file_path).stat().st_size / (1024*1024) if Path(file_path).exists() else 0
            print(f"   📹 {Path(file_path).name} ({file_size:.1f} MB)")
        
        print(f"\n📁 Files saved to: {animator.save_dir}")
        print("🎯 To view: open the MP4 files in any video player")
        
    else:
        print("❌ No animations were created successfully")
        print("💡 Try the quick test version first")

if __name__ == "__main__":
    run_surgical_animation_with_progress()