import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class SurgicalActionVisualizer:
    """
    Visualizer for surgical action recognition and planning predictions.
    Designed to showcase model performance in predicting action transitions.
    """
    
    def __init__(self, figsize=(16, 10)):
        self.figsize = figsize
        self.setup_colormaps()
        
    def setup_colormaps(self):
        """Setup custom colormaps for different visualization components."""
        # Ground truth colormap (gold/yellow theme)
        self.gt_cmap = LinearSegmentedColormap.from_list(
            'ground_truth', ['white', '#FFD700', '#B8860B'], N=256
        )
        
        # Recognition predictions (blue theme)
        self.recog_cmap = LinearSegmentedColormap.from_list(
            'recognition', ['white', '#87CEEB', '#4682B4'], N=256
        )
        
        # Planning predictions (green theme)
        self.plan_cmap = LinearSegmentedColormap.from_list(
            'planning', ['white', '#90EE90', '#228B22'], N=256
        )
        
        # Transition highlight (red theme)
        self.transition_cmap = LinearSegmentedColormap.from_list(
            'transitions', ['white', '#FFB6C1', '#DC143C'], N=256
        )

    def find_interesting_transitions(self, 
                                   recognition_gt: np.ndarray,
                                   planning_gt: np.ndarray,
                                   planning_pred: np.ndarray,
                                   min_transition_gap: int = 5,
                                   min_actions: int = 2) -> List[Dict]:
        """
        Find timestamps where interesting action transitions occur that the model predicts well.
        
        Args:
            recognition_gt: Ground truth recognition [frames x classes]
            planning_gt: Ground truth planning [frames x horizon x classes] 
            planning_pred: Planning predictions [frames x horizon x classes]
            min_transition_gap: Minimum frames between transitions
            min_actions: Minimum number of actions in transition
            
        Returns:
            List of interesting transition points with metadata
        """
        interesting_points = []
        threshold = 0.5
        
        for frame in range(len(recognition_gt) - planning_pred.shape[1]):
            if frame >= len(planning_gt):
                continue
                
            # Get current active actions
            current_actions = set(np.where(recognition_gt[frame] > threshold)[0])
            
            # Look ahead in planning ground truth to find transitions
            max_horizon = min(planning_pred.shape[1], planning_gt.shape[1])
            
            transitions_found = []
            for horizon_idx in range(max_horizon):
                if horizon_idx < planning_gt.shape[1]:
                    # Get future actions from planning ground truth
                    future_actions = set(np.where(planning_gt[frame, horizon_idx] > threshold)[0])
                    
                    # Check for action changes from current to future
                    stopped_actions = current_actions - future_actions
                    started_actions = future_actions - current_actions
                    
                    if len(stopped_actions) > 0 or len(started_actions) > 0:
                        if horizon_idx < planning_pred.shape[1]:
                            # Check if model predicted this transition
                            pred_actions = set(np.where(planning_pred[frame, horizon_idx] > threshold)[0])
                            
                            # Calculate prediction accuracy for this transition
                            correct_stops = len(stopped_actions & (current_actions - pred_actions))
                            correct_starts = len(started_actions & pred_actions)
                            total_changes = len(stopped_actions) + len(started_actions)
                            
                            if total_changes > 0:
                                accuracy = (correct_stops + correct_starts) / total_changes
                                transitions_found.append({
                                    'horizon_step': horizon_idx,
                                    'future_frame': frame + horizon_idx + 1,
                                    'stopped_actions': stopped_actions,
                                    'started_actions': started_actions,
                                    'accuracy': accuracy,
                                    'gt_future_actions': future_actions,
                                    'pred_future_actions': pred_actions
                                })
            
            # Rate this timestamp based on transition complexity and prediction accuracy
            if len(transitions_found) >= min_actions:
                avg_accuracy = np.mean([t['accuracy'] for t in transitions_found])
                total_transitions = sum(len(t['stopped_actions']) + len(t['started_actions']) 
                                      for t in transitions_found)
                
                score = avg_accuracy * total_transitions  # Combine accuracy and complexity
                
                interesting_points.append({
                    'frame': frame,
                    'score': score,
                    'avg_accuracy': avg_accuracy,
                    'total_transitions': total_transitions,
                    'transitions': transitions_found,
                    'active_actions': current_actions
                })
        
        # Sort by score and return top candidates
        interesting_points.sort(key=lambda x: x['score'], reverse=True)
        return interesting_points

    def plot_recognition_and_planning(self,
                                    recognition_gt: np.ndarray,
                                    recognition_pred: np.ndarray,
                                    planning_gt: np.ndarray,
                                    planning_pred: np.ndarray,
                                    center_frame: int,
                                    time_window: int = 60,
                                    selected_actions: Optional[List[int]] = None,
                                    rollout_horizon: int = 20,
                                    threshold: float = 0.5,
                                    save_path: Optional[str] = None,
                                    title_suffix: str = "",
                                    show_transitions: bool = True) -> plt.Figure:
        """
        Create a comprehensive visualization of recognition and planning performance.
        
        Args:
            recognition_gt: Ground truth recognition [frames x classes]
            recognition_pred: Recognition predictions [frames x classes]
            planning_gt: Ground truth planning [frames x horizon x classes]
            planning_pred: Planning predictions [frames x horizon x classes]
            center_frame: Central frame to focus visualization around
            time_window: Total time window to show (frames)
            selected_actions: Specific action classes to show (if None, auto-select)
            rollout_horizon: Planning horizon to visualize
            threshold: Binary threshold for action detection
            save_path: Path to save figure (optional)
            title_suffix: Additional text for title
            show_transitions: Whether to highlight transitions
            
        Returns:
            matplotlib Figure object
        """
        
        # Define time range
        start_frame = max(0, center_frame - time_window // 2)
        end_frame = min(len(recognition_gt), start_frame + time_window)
        start_frame = max(0, end_frame - time_window)  # Adjust if near end
        
        # Auto-select interesting actions if not provided
        if selected_actions is None:
            selected_actions = self._auto_select_actions(
                recognition_gt[start_frame:end_frame], 
                planning_pred[start_frame:end_frame] if start_frame < len(planning_pred) else None,
                max_actions=15
            )
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 2, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Recognition subplot (top left)
        ax_recog = fig.add_subplot(gs[0, 0])
        self._plot_recognition_panel(ax_recog, recognition_gt, recognition_pred, 
                                   start_frame, end_frame, selected_actions, 
                                   center_frame, threshold)
        
        # Planning subplot (top right) 
        ax_plan = fig.add_subplot(gs[0, 1])
        self._plot_planning_panel(ax_plan, planning_gt, planning_pred,
                                start_frame, end_frame, selected_actions,
                                center_frame, rollout_horizon, threshold)
        
        # Combined overview (bottom span)
        ax_combined = fig.add_subplot(gs[1, :2])
        self._plot_combined_panel(ax_combined, recognition_gt, recognition_pred,
                                planning_gt, planning_pred, start_frame, end_frame,
                                selected_actions, center_frame, rollout_horizon, threshold)
        
        # Statistics panel (right)
        ax_stats = fig.add_subplot(gs[:, 2])
        self._plot_statistics_panel(ax_stats, recognition_gt, recognition_pred,
                                   planning_gt, planning_pred, start_frame, end_frame,
                                   selected_actions, center_frame, threshold)
        
        # Add transitions overlay if requested
        if show_transitions:
            self._add_transition_highlights(ax_combined, recognition_gt, planning_pred,
                                          start_frame, center_frame, selected_actions, threshold)
        
        # Set main title
        main_title = f"Surgical Action Recognition & Planning Analysis"
        if title_suffix:
            main_title += f" - {title_suffix}"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
        
        # Add timestamp info
        fig.text(0.02, 0.02, f"Center Frame: {center_frame} | Time Window: ±{time_window//2} frames | "
                            f"Actions: {len(selected_actions)} selected", fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
            
        return fig

    def _auto_select_actions(self, recognition_data: np.ndarray, 
                           planning_data: Optional[np.ndarray] = None,
                           max_actions: int = 15) -> List[int]:
        """Auto-select the most interesting action classes to display."""
        
        # Find actions that are active in the time window
        active_actions = np.where(np.any(recognition_data > 0.3, axis=0))[0]
        
        if planning_data is not None:
            # Also include actions that appear in planning
            planning_flat = planning_data.reshape(-1, planning_data.shape[-1])
            planning_active = np.where(np.any(planning_flat > 0.3, axis=0))[0]
            active_actions = np.unique(np.concatenate([active_actions, planning_active]))
        
        # If too many actions, select most variable ones
        if len(active_actions) > max_actions:
            variances = []
            for action in active_actions:
                variance = np.var(recognition_data[:, action])
                variances.append(variance)
            
            # Select actions with highest variance (most interesting dynamics)
            top_indices = np.argsort(variances)[-max_actions:]
            active_actions = active_actions[top_indices]
        
        return sorted(active_actions.tolist())

    def _plot_recognition_panel(self, ax, recognition_gt, recognition_pred,
                              start_frame, end_frame, selected_actions,
                              center_frame, threshold):
        """Plot recognition ground truth vs predictions."""
        
        frames = range(start_frame, end_frame)
        
        # Ground truth
        gt_data = recognition_gt[start_frame:end_frame][:, selected_actions].T
        im1 = ax.imshow(gt_data, aspect='auto', cmap=self.gt_cmap, 
                       vmin=0, vmax=1, alpha=0.7)
        
        # Predictions overlay
        pred_data = recognition_pred[start_frame:end_frame][:, selected_actions].T
        # Only show where predictions exceed threshold
        pred_mask = pred_data > threshold
        pred_overlay = np.where(pred_mask, pred_data, 0)
        ax.imshow(pred_overlay, aspect='auto', cmap=self.recog_cmap, 
                 vmin=0, vmax=1, alpha=0.8)
        
        # Current frame line
        current_pos = center_frame - start_frame
        ax.axvline(x=current_pos, color='red', linewidth=3, linestyle='--', alpha=0.8)
        
        ax.set_title('Recognition: Ground Truth + Predictions', fontweight='bold')
        ax.set_ylabel('Action Classes')
        ax.set_xlabel('Time (frames)')
        
        # Set action labels
        ax.set_yticks(range(len(selected_actions)))
        ax.set_yticklabels([f'A{a}' for a in selected_actions])

    def _plot_planning_panel(self, ax, planning_gt, planning_pred,
                           start_frame, end_frame, selected_actions,
                           center_frame, rollout_horizon, threshold):
        """Plot planning ground truth vs predictions."""
        
        # Create planning visualization matrix
        time_range = end_frame - start_frame
        
        # Ground truth planning (what should happen in the future)
        gt_planning_matrix = np.zeros((len(selected_actions), time_range))
        
        # Predicted planning (what model thinks will happen)
        pred_planning_matrix = np.zeros((len(selected_actions), time_range))
        
        for i, frame in enumerate(range(start_frame, end_frame)):
            # Add ground truth planning from this timestep
            if frame < len(planning_gt):
                max_horizon = min(rollout_horizon, planning_gt.shape[1])
                for h in range(max_horizon):
                    future_i = i + h + 1
                    if future_i < time_range:
                        for j, action in enumerate(selected_actions):
                            gt_val = planning_gt[frame, h, action]
                            # Take maximum value across overlapping predictions
                            gt_planning_matrix[j, future_i] = max(
                                gt_planning_matrix[j, future_i], gt_val
                            )
            
            # Add planning predictions from this timestep
            if frame < len(planning_pred) and frame >= center_frame:
                max_horizon = min(rollout_horizon, planning_pred.shape[1])
                for h in range(max_horizon):
                    future_i = i + h + 1
                    if future_i < time_range:
                        for j, action in enumerate(selected_actions):
                            pred_val = planning_pred[frame, h, action]
                            if pred_val > threshold:
                                pred_planning_matrix[j, future_i] = max(
                                    pred_planning_matrix[j, future_i], pred_val
                                )
        
        # Plot ground truth
        ax.imshow(gt_planning_matrix, aspect='auto', cmap=self.gt_cmap, 
                 vmin=0, vmax=1, alpha=0.7)
        
        # Overlay predictions
        pred_mask = pred_planning_matrix > 0
        pred_overlay = np.where(pred_mask, pred_planning_matrix, 0)
        ax.imshow(pred_overlay, aspect='auto', cmap=self.plan_cmap, 
                 vmin=0, vmax=1, alpha=0.8)
        
        # Current frame line
        current_pos = center_frame - start_frame
        ax.axvline(x=current_pos, color='red', linewidth=3, linestyle='--', alpha=0.8)
        
        # Planning horizon indicator
        if current_pos < time_range - rollout_horizon:
            rect = patches.Rectangle((current_pos, -0.5), rollout_horizon, 
                                   len(selected_actions), linewidth=2, 
                                   edgecolor='green', facecolor='none', alpha=0.6)
            ax.add_patch(rect)
        
        ax.set_title('Planning: Future Predictions', fontweight='bold')
        ax.set_ylabel('Action Classes')
        ax.set_xlabel('Time (frames)')
        
        # Set action labels
        ax.set_yticks(range(len(selected_actions)))
        ax.set_yticklabels([f'A{a}' for a in selected_actions])

    def _plot_combined_panel(self, ax, recognition_gt, recognition_pred,
                           planning_gt, planning_pred, start_frame, end_frame,
                           selected_actions, center_frame, rollout_horizon, threshold):
        """Plot combined recognition and planning view."""
        
        time_range = end_frame - start_frame
        combined_matrix = np.zeros((len(selected_actions), time_range, 3))  # RGB
        
        for i, frame in enumerate(range(start_frame, end_frame)):
            for j, action in enumerate(selected_actions):
                # Ground truth recognition (yellow channel)
                if frame < len(recognition_gt):
                    gt_val = recognition_gt[frame, action]
                    combined_matrix[j, i, 1] = gt_val  # Yellow = R+G
                    combined_matrix[j, i, 0] = gt_val
                
                # Recognition prediction (blue channel)
                if frame < len(recognition_pred):
                    recog_val = recognition_pred[frame, action]
                    if recog_val > threshold:
                        combined_matrix[j, i, 2] = recog_val
                
                # Planning ground truth (yellow/gold channel overlay)
                if frame < len(planning_gt):
                    max_horizon = min(rollout_horizon, planning_gt.shape[1])
                    for h in range(max_horizon):
                        future_frame_idx = i + h + 1
                        if future_frame_idx < time_range:
                            gt_plan_val = planning_gt[frame, h, action]
                            if gt_plan_val > threshold:
                                # Add to yellow channels for ground truth
                                combined_matrix[j, future_frame_idx, 1] = max(
                                    combined_matrix[j, future_frame_idx, 1], gt_plan_val * 0.8
                                )
                                combined_matrix[j, future_frame_idx, 0] = max(
                                    combined_matrix[j, future_frame_idx, 0], gt_plan_val * 0.8
                                )
                
                # Planning prediction (green channel)
                if frame >= center_frame and frame < len(planning_pred):
                    max_horizon = min(rollout_horizon, planning_pred.shape[1])
                    for h in range(max_horizon):
                        future_frame_idx = i + h + 1
                        if future_frame_idx < time_range:
                            pred_val = planning_pred[frame, h, action]
                            if pred_val > threshold:
                                combined_matrix[j, future_frame_idx, 1] = max(
                                    combined_matrix[j, future_frame_idx, 1], pred_val
                                )
        
        # Normalize and display
        combined_matrix = np.clip(combined_matrix, 0, 1)
        ax.imshow(combined_matrix, aspect='auto')
        
        # Current frame line
        current_pos = center_frame - start_frame
        ax.axvline(x=current_pos, color='white', linewidth=4, linestyle='-', alpha=0.9)
        
        ax.set_title('Combined View: Recognition + Planning', fontweight='bold')
        ax.set_ylabel('Action Classes')
        ax.set_xlabel('Time (frames)')
        
        # Set action labels
        ax.set_yticks(range(len(selected_actions)))
        ax.set_yticklabels([f'A{a}' for a in selected_actions])

    def _plot_statistics_panel(self, ax, recognition_gt, recognition_pred,
                             planning_gt, planning_pred, start_frame, end_frame,
                             selected_actions, center_frame, threshold):
        """Plot performance statistics."""
        
        ax.axis('off')
        
        # Calculate recognition accuracy
        recog_gt_window = recognition_gt[start_frame:end_frame][:, selected_actions]
        recog_pred_window = recognition_pred[start_frame:end_frame][:, selected_actions]
        
        recog_gt_binary = (recog_gt_window > threshold).astype(int)
        recog_pred_binary = (recog_pred_window > threshold).astype(int)
        
        recog_accuracy = np.mean(recog_gt_binary == recog_pred_binary)
        
        # Calculate planning accuracy (simplified)
        planning_accuracy = 0.0
        if center_frame < len(planning_pred):
            plan_horizon = min(20, planning_pred.shape[1])
            correct_predictions = 0
            total_predictions = 0
            
            for h in range(plan_horizon):
                future_frame = center_frame + h + 1
                if future_frame < len(planning_gt):
                    gt_future = (planning_gt[future_frame, selected_actions] > threshold).astype(int)
                    pred_future = (planning_pred[center_frame, h, selected_actions] > threshold).astype(int)
                    
                    correct_predictions += np.sum(gt_future == pred_future)
                    total_predictions += len(selected_actions)
            
            if total_predictions > 0:
                planning_accuracy = correct_predictions / total_predictions
        
        # Statistics text
        stats_text = f"""
PERFORMANCE METRICS

Recognition Accuracy:
{recog_accuracy:.1%}

Planning Accuracy:
{planning_accuracy:.1%}

Active Actions:
{np.sum(recog_gt_binary[center_frame - start_frame] if center_frame - start_frame < len(recog_gt_binary) else 0)}

Time Window:
{end_frame - start_frame} frames

Current Frame:
{center_frame}

Selected Actions:
{len(selected_actions)} classes

Threshold:
{threshold}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    def _add_transition_highlights(self, ax, recognition_gt, planning_pred,
                                 start_frame, center_frame, selected_actions, threshold):
        """Add visual highlights for action transitions."""
        
        if center_frame >= len(planning_pred):
            return
            
        # Find transitions in current planning prediction
        for h in range(min(10, planning_pred.shape[1] - 1)):
            current_step = planning_pred[center_frame, h, selected_actions]
            next_step = planning_pred[center_frame, h + 1, selected_actions]
            
            # Find actions that change
            current_active = current_step > threshold
            next_active = next_step > threshold
            
            transitions = current_active != next_active
            
            if np.any(transitions):
                # Highlight transition point
                transition_frame = center_frame + h + 1 - start_frame
                for j, action_idx in enumerate(selected_actions):
                    if transitions[j]:
                        # Add colored dot to highlight transition
                        ax.scatter(transition_frame, j, s=100, c='red', 
                                 marker='o', alpha=0.8, edgecolors='white', linewidth=2)

def create_sample_data(num_frames: int = 200, num_classes: int = 100, 
                      rollout_horizon: int = 20) -> Tuple[np.ndarray, ...]:
    """
    Create sample data for testing the visualizer.
    This simulates realistic surgical action patterns.
    """
    
    # Recognition ground truth with realistic action patterns
    recognition_gt = np.zeros((num_frames, num_classes))
    
    # Create some realistic action sequences
    active_actions = [5, 12, 23, 34, 45, 56, 67, 78, 89]  # Common surgical actions
    
    for i, action in enumerate(active_actions):
        # Create smooth action curves with transitions
        start_frame = i * 20 + np.random.randint(-5, 5)
        duration = 30 + np.random.randint(-10, 20)
        
        for f in range(max(0, start_frame), min(num_frames, start_frame + duration)):
            # Smooth activation with noise
            phase = (f - start_frame) / duration * np.pi
            activation = 0.8 * np.sin(phase) + 0.1 * np.random.randn()
            recognition_gt[f, action] = max(0, min(1, activation))
    
    # Recognition predictions (add some noise and prediction errors)
    recognition_pred = recognition_gt + 0.1 * np.random.randn(num_frames, num_classes)
    recognition_pred = np.clip(recognition_pred, 0, 1)
    
    # Planning ground truth (future action sequences from recognition_gt)
    planning_gt = np.zeros((num_frames, rollout_horizon, num_classes))
    
    for f in range(num_frames):
        for h in range(rollout_horizon):
            future_frame = f + h + 1
            if future_frame < num_frames:
                # Future ground truth is what actually happens in recognition_gt
                planning_gt[f, h] = recognition_gt[future_frame]
    
    # Planning predictions (model's future predictions)
    planning_pred = np.zeros((num_frames, rollout_horizon, num_classes))
    
    for f in range(num_frames):
        for h in range(rollout_horizon):
            future_frame = f + h + 1
            if future_frame < num_frames:
                # Model prediction with some decay and uncertainty
                decay = 0.95 ** h  # Confidence decreases with horizon
                noise = 0.05 * h * np.random.randn(num_classes)
                planning_pred[f, h] = planning_gt[f, h] * decay + noise
                planning_pred[f, h] = np.clip(planning_pred[f, h], 0, 1)
    
    return recognition_gt, recognition_pred, planning_gt, planning_pred

def demo_visualization():
    """Demonstrate the visualization with sample data."""
    
    print("Creating sample surgical action data...")
    recognition_gt, recognition_pred, planning_gt, planning_pred = create_sample_data()
    
    print("Initializing visualizer...")
    visualizer = SurgicalActionVisualizer()
    
    print("Finding interesting transition points...")
    interesting_points = visualizer.find_interesting_transitions(
        recognition_gt, planning_gt, planning_pred
    )
    
    if len(interesting_points) > 0:
        best_point = interesting_points[0]
        center_frame = best_point['frame']
        print(f"Selected frame {center_frame} with score {best_point['score']:.2f}")
        print(f"Average accuracy: {best_point['avg_accuracy']:.1%}")
        print(f"Total transitions: {best_point['total_transitions']}")
    else:
        center_frame = 100  # Default center
        print("No interesting transitions found, using default center frame")
    
    print("Creating visualization...")
    fig = visualizer.plot_recognition_and_planning(
        recognition_gt=recognition_gt,
        recognition_pred=recognition_pred,
        planning_gt=planning_gt,
        planning_pred=planning_pred,
        center_frame=center_frame,
        time_window=80,
        title_suffix="Sample Data Demo",
        show_transitions=True,
        save_path="surgical_action_analysis.png"
    )
    
    plt.show()
    print("Visualization complete!")

if __name__ == "__main__":
    # Run demo
    demo_visualization()
    
    # Example usage with real data:
    """
    # Load your numpy arrays
    recognition_gt = np.load('recognition_gt.npy')      # [frames x 100]
    recognition_pred = np.load('recognition_pred.npy')  # [frames x 100]  
    planning_gt = np.load('planning_gt.npy')            # [frames x 20 x 100]
    planning_pred = np.load('planning_pred.npy')        # [frames x 20 x 100]
    
    # Create visualizer
    visualizer = SurgicalActionVisualizer(figsize=(20, 12))
    
    # Find best transition points
    transitions = visualizer.find_interesting_transitions(
        recognition_gt, planning_gt, planning_pred
    )
    
    # Create publication-quality figure
    for i, point in enumerate(transitions[:3]):  # Top 3 examples
        fig = visualizer.plot_recognition_and_planning(
            recognition_gt=recognition_gt,
            recognition_pred=recognition_pred,  
            planning_gt=planning_gt,
            planning_pred=planning_pred,
            center_frame=point['frame'],
            time_window=100,
            selected_actions=None,  # Auto-select interesting actions
            save_path=f'paper_figure_example_{i+1}.png',
            title_suffix=f"Transition Example {i+1}",
            show_transitions=True
        )
    """