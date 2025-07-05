import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class SurgicalActionVisualizer:
    """
    Visualizer for surgical action recognition and planning predictions.
    Designed to showcase model performance in predicting action transitions.
    Uses TP/FN/FP/TN color scheme for journal-quality figures.
    """
    
    def __init__(self, figsize=(16, 10), fps=1.0):
        self.figsize = figsize
        self.fps = fps  # Frames per second for time conversion
        self.setup_colormaps()
        self.load_class_mapping('data/labels.json')  # Load class mapping from JSON file

    def load_class_mapping(self, class_labels_path: str) -> Dict[int, str]:
        import json
        with open(class_labels_path, 'r') as f:
            self.class_mapping = json.load(f)
        
    def setup_colormaps(self):
        """Setup custom colormaps using TP/FN/FP/TN color scheme."""
        # Define the four classification colors
        self.tp_color = '#228B22'      # Green for True Positives
        self.fn_color = '#D2B48C'      # Light brown/sand for False Negatives  
        self.fp_color = '#4682B4'      # Blue for False Positives
        self.tn_color = '#FFFFFF'      # White for True Negatives
        
        # Create discrete colormap for classification results
        self.classification_colors = [self.tn_color, self.fp_color, self.fn_color, self.tp_color]
        self.classification_cmap = ListedColormap(self.classification_colors)
        
        # Ground truth colormap (sand/brown theme for what should be there)
        self.gt_cmap = LinearSegmentedColormap.from_list(
            'ground_truth', [self.tn_color, '#F5DEB3', self.fn_color], N=256
        )
        
        # Recognition predictions (green theme for what was detected)
        self.recog_cmap = LinearSegmentedColormap.from_list(
            'recognition', [self.tn_color, '#90EE90', self.tp_color], N=256
        )
        
        # Planning predictions (green theme)
        self.plan_cmap = LinearSegmentedColormap.from_list(
            'planning', [self.tn_color, '#90EE90', self.tp_color], N=256
        )
        
        # Error highlighting (blue theme for false positives)
        self.error_cmap = LinearSegmentedColormap.from_list(
            'errors', [self.tn_color, '#87CEEB', self.fp_color], N=256
        )

    def _classify_predictions(self, ground_truth: np.ndarray, predictions: np.ndarray, 
                            threshold: float = 0.5) -> np.ndarray:
        """
        Classify predictions into TP/FN/FP/TN.
        
        Args:
            ground_truth: Ground truth binary matrix
            predictions: Prediction matrix  
            threshold: Threshold for binarizing predictions
            
        Returns:
            Classification matrix: 0=TN, 1=FP, 2=FN, 3=TP
        """
        gt_binary = (ground_truth > threshold).astype(int)
        pred_binary = (predictions > threshold).astype(int)
        
        # Create classification matrix
        classification = np.zeros_like(gt_binary)
        
        # True Positives: gt=1, pred=1
        classification[(gt_binary == 1) & (pred_binary == 1)] = 3
        
        # False Negatives: gt=1, pred=0  
        classification[(gt_binary == 1) & (pred_binary == 0)] = 2
        
        # False Positives: gt=0, pred=1
        classification[(gt_binary == 0) & (pred_binary == 1)] = 1
        
        # True Negatives: gt=0, pred=0 (remains 0)
        
        return classification

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
                                    show_transitions: bool = False,
                                    use_time_format: bool = True) -> plt.Figure:
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
            use_time_format: Whether to show HH:MM:SS format (True) or frame indices (False)
            
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
        fig = plt.figure(figsize=self.figsize, facecolor='white')
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 2, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Recognition subplot (top left)
        ax_recog = fig.add_subplot(gs[0, 0], facecolor='white')
        self._plot_recognition_panel(ax_recog, recognition_gt, recognition_pred, 
                                   start_frame, end_frame, selected_actions, 
                                   center_frame, threshold, use_time_format)
        
        # Planning subplot (top right) 
        ax_plan = fig.add_subplot(gs[0, 1], facecolor='white')
        self._plot_planning_panel(ax_plan, recognition_gt, planning_pred,
                                start_frame, end_frame, selected_actions,
                                center_frame, rollout_horizon, threshold, use_time_format)
        
        # Combined overview (bottom span)
        ax_combined = fig.add_subplot(gs[1, :2], facecolor='white')
        self._plot_combined_panel(ax_combined, recognition_gt, recognition_pred,
                                planning_gt, planning_pred, start_frame, end_frame,
                                selected_actions, center_frame, rollout_horizon, threshold, use_time_format)
        
        # Statistics panel (right)
        ax_stats = fig.add_subplot(gs[:, 2], facecolor='white')
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

    def _format_time_axis(self, ax, start_frame, end_frame, center_frame, use_time_format=True, axis_type="combined"):
        """Format x-axis with proper frame indices or time stamps relative to center frame."""
        
        # Set major ticks every 10-20 frames depending on window size
        time_range = end_frame - start_frame
        if time_range <= 50:
            tick_interval = 10
        elif time_range <= 100:
            tick_interval = 20
        else:
            tick_interval = 30
            
        # Generate tick positions and labels relative to center frame
        tick_positions = []
        tick_labels = []
        
        for frame in range(start_frame, end_frame + 1, tick_interval):
            if frame <= end_frame:
                position = frame - start_frame
                relative_to_center = frame - center_frame
                tick_positions.append(position)
                
                if use_time_format:
                    # Convert frame to HH:MM:SS (assuming fps from init)
                    total_seconds = int(frame / self.fps)
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    if axis_type == "combined":
                        # Show relative time with +/- indicators
                        if relative_to_center == 0:
                            tick_labels.append(f"T₀\n{hours:02d}:{minutes:02d}:{seconds:02d}")
                        elif relative_to_center > 0:
                            tick_labels.append(f"+{relative_to_center}\n{hours:02d}:{minutes:02d}:{seconds:02d}")
                        else:
                            tick_labels.append(f"{relative_to_center}\n{hours:02d}:{minutes:02d}:{seconds:02d}")
                    else:
                        tick_labels.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
                else:
                    # Show relative frame indices
                    if axis_type == "combined":
                        if relative_to_center == 0:
                            tick_labels.append(f"T₀ ({frame})")
                        elif relative_to_center > 0:
                            tick_labels.append(f"+{relative_to_center}")
                        else:
                            tick_labels.append(f"{relative_to_center}")
                    else:
                        tick_labels.append(f"{frame}")
        
        # Ensure center frame is always marked
        center_position = center_frame - start_frame
        if center_position not in tick_positions:
            tick_positions.append(center_position)
            if use_time_format:
                total_seconds = int(center_frame / self.fps)
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                if axis_type == "combined":
                    tick_labels.append(f"T₀\n{hours:02d}:{minutes:02d}:{seconds:02d}")
                else:
                    tick_labels.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            else:
                if axis_type == "combined":
                    tick_labels.append(f"T₀ ({center_frame})")
                else:
                    tick_labels.append(f"{center_frame}")
        
        # Sort tick positions and labels together
        sorted_pairs = sorted(zip(tick_positions, tick_labels))
        tick_positions, tick_labels = zip(*sorted_pairs)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45 if use_time_format else 0, fontsize=9)
        
        if axis_type == "combined":
            ax.set_xlabel('Relative Time (Past ← T₀ → Future)', fontweight='bold')
        elif use_time_format:
            ax.set_xlabel('Time (HH:MM:SS)', fontweight='bold')
        else:
            ax.set_xlabel('Frame Index', fontweight='bold')

    def _add_color_legend(self, ax, legend_type="recognition"):
        """Add color legend to subplot."""
        
        if legend_type == "recognition":
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=self.fn_color, alpha=0.8, label='Ground Truth (FN)'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.tp_color, alpha=0.8, label='True Positive'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.fp_color, alpha=0.8, label='False Positive')
            ]
        elif legend_type == "planning":
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=self.fn_color, alpha=0.8, label='Ground Truth'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.tp_color, alpha=0.8, label='True Positive'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.fp_color, alpha=0.8, label='False Positive'),
            ]
        elif legend_type == "combined":
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=self.fn_color, alpha=0.8, label='Ground Truth (FN)'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.tp_color, alpha=0.8, label='True Positive'),
                plt.Rectangle((0, 0), 1, 1, facecolor=self.fp_color, alpha=0.8, label='False Positive'),
                plt.Line2D([0], [0], color='red', linewidth=4, label='Current Frame')
            ]
        
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                         frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.5)

    def _auto_select_actions(self, recognition_data: np.ndarray, 
                           planning_data: Optional[np.ndarray] = None,
                           max_actions: int = 15) -> List[int]:
        
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
                              center_frame, threshold, use_time_format):
        """Plot recognition ground truth vs predictions using TP/FN/FP/TN colors.
        Only shows observed frames (up to current time)."""
        
        # Only show frames up to current time for recognition (observed data only)
        observed_end_frame = min(end_frame, center_frame + 1)
        
        # Get data for selected time window and actions (only observed frames)
        gt_data = recognition_gt[start_frame:observed_end_frame][:, selected_actions]
        pred_data = recognition_pred[start_frame:observed_end_frame][:, selected_actions]
        
        # Classify predictions for observed frames only
        classification = self._classify_predictions(gt_data, pred_data, threshold)
        
        # Create full matrix with white (TN) for future frames
        time_range = end_frame - start_frame
        full_classification = np.zeros((len(selected_actions), time_range))
        
        # Fill in observed data
        observed_range = observed_end_frame - start_frame
        full_classification[:, :observed_range] = classification.T
        
        # Create visualization showing classification results
        im = ax.imshow(full_classification, aspect='auto', cmap=self.classification_cmap, 
                      vmin=0, vmax=3, interpolation='nearest')
        
        # Current frame line
        current_pos = center_frame - start_frame
        ax.axvline(x=current_pos, color='red', linewidth=3, linestyle='--', alpha=0.8)
        
        # Add shaded region for future (unobserved) frames
        future_start = current_pos + 0.5
        if future_start < time_range:
            ax.axvspan(future_start, time_range, alpha=0.2, color='lightgray', 
                      label='Future (Unobserved)')
        
        ax.set_title('Recognition: Observed Performance', fontweight='bold')
        ax.set_ylabel('Action Classes')
        
        # Format time axis
        self._format_time_axis(ax, start_frame, end_frame, center_frame, use_time_format, "recognition")

        # Define y-ticks names based on selected actions and class mapping
        y_ticks_labels = []
        for action in selected_actions:
            action_name = f"'{self.class_mapping['action'][str(action)]}' (A{action})"
            y_ticks_labels.append(action_name)
        
        # Set action labels
        ax.set_yticks(range(len(selected_actions)))
        ax.set_yticklabels(y_ticks_labels, rotation=45, fontsize=9)
        
        
        # Add color legend
        self._add_color_legend(ax, "recognition")

    def _plot_planning_panel(self, ax, recognition_gt, planning_pred,
                           start_frame, end_frame, selected_actions,
                           center_frame, rollout_horizon, threshold, use_time_format):
        """Plot planning ground truth vs predictions using TP/FN/FP/TN colors.
        Focuses on future frames (right of current time)."""
        
        # Create planning visualization matrix
        time_range = end_frame - start_frame
        classification_matrix = np.zeros((len(selected_actions), time_range))
        
        # Only show planning for future frames (from center_frame onwards)
        current_pos = center_frame - start_frame
        
        if center_frame < len(planning_pred):
            for h in range(min(rollout_horizon, planning_pred.shape[1])):
                future_frame_idx = current_pos + h + 1
                if future_frame_idx < time_range:
                    future_frame = center_frame + h + 1
                    
                    if future_frame < len(recognition_gt):  # Use actual future for ground truth
                        for j, action in enumerate(selected_actions):
                            # Ground truth: what actually happens in the future
                            gt_val = recognition_gt[future_frame, action]
                            # Prediction: what model predicted at center_frame for this future time
                            pred_val = planning_pred[center_frame, h, action]
                            
                            # Classify this prediction
                            gt_binary = gt_val > threshold
                            pred_binary = pred_val > threshold
                            
                            if gt_binary and pred_binary:
                                classification_matrix[j, future_frame_idx] = 3  # TP
                            elif gt_binary and not pred_binary:
                                classification_matrix[j, future_frame_idx] = 2  # FN
                            elif not gt_binary and pred_binary:
                                classification_matrix[j, future_frame_idx] = 1  # FP
                            # TN remains 0
        
        # Plot classification matrix
        ax.imshow(classification_matrix, aspect='auto', cmap=self.classification_cmap, 
                 vmin=0, vmax=3, interpolation='nearest')
        
        # Current frame line
        ax.axvline(x=current_pos, color='red', linewidth=3, linestyle='--', alpha=0.8)
        
        # Add shaded region for past (not relevant for planning)
        if current_pos > 0:
            ax.axvspan(0, current_pos + 0.5, alpha=0.2, color='lightgray', 
                      label='Past (Not Predicted)')
        
        # Planning horizon indicator
        horizon_end = min(current_pos + rollout_horizon, time_range)
        if horizon_end > current_pos:
            rect = patches.Rectangle((current_pos + 0.5, -0.5), horizon_end - current_pos - 0.5, 
                                   len(selected_actions), linewidth=2, 
                                   edgecolor='darkgreen', facecolor='none', alpha=0.6,
                                   label='Planning Horizon')
            ax.add_patch(rect)
        
        ax.set_title('Planning: Future Prediction Performance', fontweight='bold')
        ax.set_ylabel('Action Classes')
        
        # Format time axis
        self._format_time_axis(ax, start_frame, end_frame, center_frame, use_time_format, "planning")
        
        
        # Set action labels
        ax.set_yticks(range(len(selected_actions)))
        ax.set_yticklabels([f'A{a}' for a in selected_actions])
        
        # Add color legend
        self._add_color_legend(ax, "planning")

    def _plot_combined_panel(self, ax, recognition_gt, recognition_pred,
                           planning_gt, planning_pred, start_frame, end_frame,
                           selected_actions, center_frame, rollout_horizon, threshold, use_time_format):
        """Plot combined recognition and planning view with clear temporal separation."""
        
        time_range = end_frame - start_frame
        combined_matrix = np.zeros((len(selected_actions), time_range))  # Start with TN (0)
        current_pos = center_frame - start_frame
        
        # RECOGNITION PART: Process observed frames (past and present - left of and at current time)
        observed_end_frame = min(end_frame, center_frame + 1)
        for i, frame in enumerate(range(start_frame, observed_end_frame)):
            if frame < len(recognition_gt) and frame < len(recognition_pred):
                for j, action in enumerate(selected_actions):
                    gt_val = recognition_gt[frame, action]
                    pred_val = recognition_pred[frame, action]
                    
                    # Classify recognition
                    gt_binary = gt_val > threshold
                    pred_binary = pred_val > threshold
                    
                    if gt_binary and pred_binary:
                        combined_matrix[j, i] = 3  # TP
                    elif gt_binary and not pred_binary:
                        combined_matrix[j, i] = 2  # FN
                    elif not gt_binary and pred_binary:
                        combined_matrix[j, i] = 1  # FP
                    # TN remains 0
        
        # PLANNING PART: Process future frames (right of current time)
        if center_frame < len(planning_pred):
            for h in range(min(rollout_horizon, planning_pred.shape[1])):
                future_frame_idx = current_pos + h + 1
                if future_frame_idx < time_range:
                    future_frame = center_frame + h + 1
                    
                    if future_frame < len(recognition_gt):  # Use actual future as ground truth
                        for j, action in enumerate(selected_actions):
                            # Get planning prediction made at center_frame
                            pred_val = planning_pred[center_frame, h, action]
                            pred_binary = pred_val > threshold
                            
                            # Get actual future ground truth
                            gt_val = recognition_gt[future_frame, action]
                            gt_binary = gt_val > threshold
                            
                            # Classify planning prediction
                            if gt_binary and pred_binary:
                                combined_matrix[j, future_frame_idx] = 3  # TP (planning)
                            elif gt_binary and not pred_binary:
                                combined_matrix[j, future_frame_idx] = 2  # FN (planning)
                            elif not gt_binary and pred_binary:
                                combined_matrix[j, future_frame_idx] = 1  # FP (planning)
                            # TN remains 0
        
        # Display with white background
        ax.imshow(combined_matrix, aspect='auto', cmap=self.classification_cmap, 
                 vmin=0, vmax=3, interpolation='nearest')
        
        # Current frame line (separator between recognition and planning)
        ax.axvline(x=current_pos, color='red', linewidth=3, linestyle='--', alpha=0.8)
        
        # Add region labels
        if current_pos > 0:
            ax.text(current_pos/2, len(selected_actions) + 0.5, 'RECOGNITION\n(Observed)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        if current_pos < time_range - 1:
            future_center = current_pos + (time_range - current_pos)/2
            ax.text(future_center, len(selected_actions) + 0.5, 'PLANNING\n(Predicted)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax.set_title('Combined View: Recognition (Past) + Planning (Future)', fontweight='bold')
        ax.set_ylabel('Action Classes')
        
        # Format time axis with relative indexing
        self._format_time_axis(ax, start_frame, end_frame, center_frame, use_time_format, "combined")
        
        # Define y-ticks names based on selected actions and class mapping
        y_ticks_labels = []
        for action in selected_actions:
            action_name = f"'{self.class_mapping['action'][str(action)]}' (A{action})"
            y_ticks_labels.append(action_name)

        # Set action labels
        ax.set_yticks(range(len(selected_actions)))
        ax.set_yticklabels(y_ticks_labels, rotation=45, fontsize=9)
        
        # Add color legend for combined view
        self._add_color_legend(ax, "combined")

    def _plot_statistics_panel(self, ax, recognition_gt, recognition_pred,
                             planning_gt, planning_pred, start_frame, end_frame,
                             selected_actions, center_frame, threshold):
        """Plot performance statistics for observed recognition and future planning."""
        
        ax.axis('off')
        
        # Calculate recognition metrics (observed frames only)
        observed_end_frame = min(end_frame, center_frame + 1)
        recog_gt_window = recognition_gt[start_frame:observed_end_frame][:, selected_actions]
        recog_pred_window = recognition_pred[start_frame:observed_end_frame][:, selected_actions]
        
        if len(recog_gt_window) > 0:
            recog_classification = self._classify_predictions(recog_gt_window, recog_pred_window, threshold)
            
            # Count TP, FP, FN, TN for recognition
            recog_tp = np.sum(recog_classification == 3)
            recog_fp = np.sum(recog_classification == 1)
            recog_fn = np.sum(recog_classification == 2)
            recog_tn = np.sum(recog_classification == 0)
            
            # Calculate metrics
            recog_precision = recog_tp / (recog_tp + recog_fp) if (recog_tp + recog_fp) > 0 else 0
            recog_recall = recog_tp / (recog_tp + recog_fn) if (recog_tp + recog_fn) > 0 else 0
            recog_accuracy = (recog_tp + recog_tn) / (recog_tp + recog_fp + recog_fn + recog_tn)
        else:
            recog_tp = recog_fp = recog_fn = recog_tn = 0
            recog_precision = recog_recall = recog_accuracy = 0
        
        # Calculate planning metrics (future frames only)
        planning_tp = planning_fp = planning_fn = planning_tn = 0
        if center_frame < len(planning_pred):
            plan_horizon = min(20, planning_pred.shape[1])
            
            for h in range(plan_horizon):
                future_frame = center_frame + h + 1
                if future_frame < len(recognition_gt) and future_frame < end_frame:
                    # Use actual future as ground truth for planning
                    gt_future = (recognition_gt[future_frame, selected_actions] > threshold).astype(int)
                    pred_future = (planning_pred[center_frame, h, selected_actions] > threshold).astype(int)
                    
                    planning_tp += np.sum((gt_future == 1) & (pred_future == 1))
                    planning_fp += np.sum((gt_future == 0) & (pred_future == 1))
                    planning_fn += np.sum((gt_future == 1) & (pred_future == 0))
                    planning_tn += np.sum((gt_future == 0) & (pred_future == 0))
        
        plan_precision = planning_tp / (planning_tp + planning_fp) if (planning_tp + planning_fp) > 0 else 0
        plan_recall = planning_tp / (planning_tp + planning_fn) if (planning_tp + planning_fn) > 0 else 0
        plan_accuracy = (planning_tp + planning_tn) / (planning_tp + planning_fp + planning_fn + planning_tn) if (planning_tp + planning_fp + planning_fn + planning_tn) > 0 else 0
        
        # Count observed vs future frames
        observed_frames = observed_end_frame - start_frame
        future_frames = end_frame - center_frame - 1
        
        # Statistics text
        stats_text = f"""
PERFORMANCE METRICS

Recognition (Observed):
  Frames:    {observed_frames:3d}
  Accuracy:  {recog_accuracy:.1%}
  Precision: {recog_precision:.1%}
  Recall:    {recog_recall:.1%}
  
  TP: {recog_tp:4d}  FP: {recog_fp:4d}
  FN: {recog_fn:4d}  TN: {recog_tn:4d}

Planning (Future):
  Frames:    {future_frames:3d}
  Accuracy:  {plan_accuracy:.1%}
  Precision: {plan_precision:.1%}
  Recall:    {plan_recall:.1%}
  
  TP: {planning_tp:4d}  FP: {planning_fp:4d}
  FN: {planning_fn:4d}  TN: {planning_tn:4d}

Current Frame: {center_frame}
Actions: {len(selected_actions)} classes
Threshold: {threshold}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
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
                        ax.scatter(transition_frame, j, s=100, c='darkred', 
                                 marker='o', alpha=0.9, edgecolors='white', linewidth=2)

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
    visualizer = SurgicalActionVisualizer(fps=1.0)  # 1 frame per second
    
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
        title_suffix="Enhanced TP/FN/FP/TN Analysis",
        show_transitions=False,
        use_time_format=True,
        save_path="surgical_action_analysis_enhanced.png"
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
            show_transitions=True,
            use_time_format=True  # Show HH:MM:SS timestamps
        )
    """