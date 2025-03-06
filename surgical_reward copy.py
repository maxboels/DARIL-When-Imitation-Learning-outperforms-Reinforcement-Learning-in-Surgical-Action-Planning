import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
import json
from collections import defaultdict

class SurgicalRewardFunction:
    """
    Reward function for surgical procedures that balances task completion with risk assessment.
    Designed to be used with TD-MPC2 for real-time prediction during surgery.
    """
    
    def __init__(self, 
                 phase_weights=None, 
                 risk_weight=1.0, 
                 time_penalty=0.01,
                 transition_bonus=5.0,
                 completion_bonus=20.0,
                 critical_risk_threshold=4.0,
                 critical_risk_penalty=10.0,
                 smoothing_factor=3.0):
        """
        Initialize the reward function with configurable weights.
        
        Args:
            phase_weights: Dictionary mapping phase IDs to their relative importance weights
                           (default: equal weighting for all phases)
            risk_weight: Weight for the risk penalty component (higher means risk is more important)
            time_penalty: Small penalty per timestep to encourage efficiency
            transition_bonus: Reward for successfully transitioning between phases
            completion_bonus: Final reward for completing the entire procedure
            critical_risk_threshold: Threshold above which risk is considered critical
            critical_risk_penalty: Additional penalty for exceeding critical risk threshold
            smoothing_factor: Smoothing factor for reward calculation to reduce noise
        """
        # Default phase weights if not provided (equal weight to all phases)
        self.phase_weights = phase_weights or {i: 1.0 for i in range(7)}  # Assuming phases 0-6
        
        self.risk_weight = risk_weight
        self.time_penalty = time_penalty
        self.transition_bonus = transition_bonus
        self.completion_bonus = completion_bonus
        self.critical_risk_threshold = critical_risk_threshold
        self.critical_risk_penalty = critical_risk_penalty
        self.smoothing_factor = smoothing_factor
        
        # Expected phase sequence (can be customized)
        self.expected_phase_sequence = list(range(7))  # Phases 0 to 6 in order
        
        # Phase progress tracking
        self.phase_progress = {i: 0.0 for i in range(7)}
        self.current_phase = None
        self.completed_phases = set()
        
        # Historical data for training and normalization
        self.historical_phase_durations = defaultdict(list)
        self.historical_risk_distributions = defaultdict(list)
        
    def load_historical_data(self, metadata_path, video_ids=None):
        """
        Load historical surgery data to establish baseline distributions for phases and risks.
        This helps with reward normalization across different procedures.
        
        Args:
            metadata_path: Path to metadata CSV with phase and risk information
            video_ids: Optional list of specific video IDs to include
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        metadata_df = pd.read_csv(metadata_path)
        
        if video_ids:
            metadata_df = metadata_df[metadata_df['video'].isin(video_ids)]
        
        # For each video, calculate phase durations and risk distributions
        for video_id in metadata_df['video'].unique():
            video_data = metadata_df[metadata_df['video'] == video_id]
            
            # Phase durations
            phase_columns = [f'p{i}' for i in range(7) if f'p{i}' in video_data.columns]
            for phase_col in phase_columns:
                phase_id = int(phase_col.replace('p', ''))
                phase_frames = video_data[video_data[phase_col] == 1]
                self.historical_phase_durations[phase_id].append(len(phase_frames))
            
            # Risk distributions
            risk_columns = [col for col in video_data.columns if col.startswith('risk_score_')]
            for risk_col in risk_columns:
                for phase_col in phase_columns:
                    phase_id = int(phase_col.replace('p', ''))
                    phase_frames = video_data[video_data[phase_col] == 1]
                    if not phase_frames.empty:
                        risks = phase_frames[risk_col].values
                        self.historical_risk_distributions[(phase_id, risk_col)].extend(risks)
        
        # Calculate statistics for normalization
        self.phase_duration_stats = {}
        for phase_id, durations in self.historical_phase_durations.items():
            if durations:
                self.phase_duration_stats[phase_id] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations)
                }
        
        self.risk_distribution_stats = {}
        for (phase_id, risk_col), risks in self.historical_risk_distributions.items():
            if risks:
                self.risk_distribution_stats[(phase_id, risk_col)] = {
                    'mean': np.mean(risks),
                    'std': np.std(risks),
                    'min': np.min(risks),
                    'max': np.max(risks)
                }
        
        print(f"Loaded historical data from {len(metadata_df['video'].unique())} videos")
    
    def calculate_reward(self, frame_data, is_terminal=False):
        """
        Calculate the reward for a single frame in the surgical procedure.
        
        Args:
            frame_data: Dictionary containing:
                - 'phase_id': Current surgical phase ID
                - 'risk_scores': Dictionary of risk scores (different types)
                - 'frame_id': Current frame ID
                - 'action_data': Optional dictionary with detailed action information
                - 'progress_within_phase': Optional estimate of progress within current phase (0-1)
            is_terminal: Whether this is the terminal state (procedure completed)
            
        Returns:
            total_reward: The calculated reward value
            reward_components: Dictionary breaking down reward components
        """
        # Extract data
        phase_id = frame_data.get('phase_id')
        risk_scores = frame_data.get('risk_scores', {})
        frame_id = frame_data.get('frame_id', 0)
        progress_within_phase = frame_data.get('progress_within_phase', None)
        
        reward_components = {}
        
        # 1. Phase completion/progress component
        phase_reward = 0.0
        phase_transition_reward = 0.0
        
        # Check if phase has changed
        if self.current_phase is not None and phase_id != self.current_phase:
            # Successfully completed previous phase
            if phase_id in self.expected_phase_sequence:
                expected_idx = self.expected_phase_sequence.index(phase_id)
                prev_idx = self.expected_phase_sequence.index(self.current_phase)
                
                # Only reward forward progress in the expected sequence
                if expected_idx > prev_idx:
                    # Phase transition reward
                    phase_transition_reward = self.transition_bonus * self.phase_weights.get(self.current_phase, 1.0)
                    self.completed_phases.add(self.current_phase)
                    
        self.current_phase = phase_id
        
        # Reward for progress within the current phase
        if phase_id is not None:
            if progress_within_phase is not None:
                # If the caller provides a progress estimate, use it
                self.phase_progress[phase_id] = progress_within_phase
                phase_reward = progress_within_phase * self.phase_weights.get(phase_id, 1.0)
            else:
                # Otherwise use a simple incrementing counter (normalized by expected duration)
                expected_duration = (
                    self.phase_duration_stats.get(phase_id, {}).get('mean', 100) 
                    if phase_id in self.phase_duration_stats 
                    else 100
                )
                progress_increment = 1.0 / expected_duration
                self.phase_progress[phase_id] += progress_increment
                # Cap at 1.0
                self.phase_progress[phase_id] = min(self.phase_progress[phase_id], 1.0)
                phase_reward = self.phase_progress[phase_id] * self.phase_weights.get(phase_id, 1.0)
        
        reward_components['phase_progress_reward'] = phase_reward
        reward_components['phase_transition_reward'] = phase_transition_reward
        
        # 2. Risk assessment component (penalty)
        risk_penalty = 0.0
        
        # Calculate average risk score (or use a more sophisticated aggregation)
        if risk_scores:
            # Can be weighted by risk type importance if needed
            avg_risk = np.mean(list(risk_scores.values()))
            
            # Basic risk penalty proportional to risk score
            risk_penalty = self.risk_weight * avg_risk
            
            # Additional penalty for exceeding critical threshold
            if avg_risk > self.critical_risk_threshold:
                risk_penalty += self.critical_risk_penalty * (avg_risk - self.critical_risk_threshold)
                
            # Optional: Normalize based on historical risk distributions for this phase
            if phase_id is not None:
                for risk_type, score in risk_scores.items():
                    if (phase_id, risk_type) in self.risk_distribution_stats:
                        stats = self.risk_distribution_stats[(phase_id, risk_type)]
                        # Higher penalty for unusual risk in this phase
                        if score > stats['mean'] + stats['std']:
                            normalized_excess = (score - stats['mean']) / stats['std']
                            risk_penalty += self.risk_weight * normalized_excess
        
        reward_components['risk_penalty'] = -risk_penalty  # Negative because it's a penalty
        
        # 3. Time efficiency penalty
        time_penalty = self.time_penalty
        reward_components['time_penalty'] = -time_penalty  # Negative because it's a penalty
        
        # 4. Completion bonus (if terminal state)
        completion_reward = 0.0
        if is_terminal:
            # Check if all expected phases were completed
            all_phases_completed = all(phase in self.completed_phases for phase in self.expected_phase_sequence)
            if all_phases_completed:
                completion_reward = self.completion_bonus
        
        reward_components['completion_reward'] = completion_reward
        
        # Calculate total reward
        total_reward = (
            phase_reward + 
            phase_transition_reward + 
            -risk_penalty +  # Negative because it's a penalty
            -time_penalty +  # Negative because it's a penalty
            completion_reward
        )
        
        # Apply smoothing if needed (could be based on previous rewards)
        # This would require maintaining a history of rewards
        
        return total_reward, reward_components
    
    def reset(self):
        """Reset the reward function state for a new procedure"""
        self.phase_progress = {i: 0.0 for i in range(7)}
        self.current_phase = None
        self.completed_phases = set()
    
    def get_state_features(self):
        """
        Return features about the current state that might be useful for prediction models.
        This can be used as input to TD-MPC2 for reward prediction.
        """
        features = {
            'current_phase': self.current_phase,
            'phase_progress': dict(self.phase_progress),
            'completed_phases': list(self.completed_phases),
            'num_phases_completed': len(self.completed_phases),
            'normalized_procedure_progress': (
                sum(self.phase_progress[p] * self.phase_weights.get(p, 1.0) for p in range(7)) / 
                sum(self.phase_weights.values())
            )
        }
        return features
    
    def analyze_trajectory(self, trajectory_data):
        """
        Analyze a complete surgical trajectory to compute rewards and statistics.
        Useful for preparing training data for TD-MPC2.
        
        Args:
            trajectory_data: List of frame data dictionaries for a complete procedure
            
        Returns:
            Dictionary with:
            - 'rewards': List of rewards for each frame
            - 'cumulative_reward': Total reward for the procedure
            - 'reward_components': Breakdown of reward components over time
            - 'statistics': Summary statistics about the procedure
        """
        self.reset()
        rewards = []
        reward_components_history = defaultdict(list)
        
        for i, frame_data in enumerate(trajectory_data):
            is_terminal = (i == len(trajectory_data) - 1)
            reward, components = self.calculate_reward(frame_data, is_terminal)
            rewards.append(reward)
            
            for component, value in components.items():
                reward_components_history[component].append(value)
        
        # Optionally apply smoothing to the reward sequence
        if self.smoothing_factor > 0:
            rewards = gaussian_filter1d(rewards, sigma=self.smoothing_factor)
        
        # Calculate statistics
        statistics = {
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'reward_variance': np.var(rewards),
            'completed_phases': list(self.completed_phases),
            'phase_completion': {phase: self.phase_progress[phase] for phase in range(7)}
        }
        
        # Calculate component statistics
        component_stats = {}
        for component, values in reward_components_history.items():
            component_stats[component] = {
                'total': sum(values),
                'mean': np.mean(values),
                'min': min(values),
                'max': max(values)
            }
        
        return {
            'rewards': rewards,
            'cumulative_reward': sum(rewards),
            'reward_components': dict(reward_components_history),
            'component_statistics': component_stats,
            'statistics': statistics
        }
    
    def prepare_td_mpc2_training_data(self, video_ids, metadata_df, risk_column_name='risk_score_max'):
        """
        Prepare training data for TD-MPC2 from a set of surgical videos.
        
        Args:
            video_ids: List of video IDs to process
            metadata_df: DataFrame with metadata including phases and risk scores
            risk_column_name: Column name for risk scores
            
        Returns:
            Dictionary with training data structured for TD-MPC2
        """
        training_data = []
        
        for video_id in video_ids:
            # Extract data for this video
            video_metadata = metadata_df[metadata_df['video'] == video_id]
            
            if video_metadata.empty:
                print(f"Warning: No data found for video {video_id}")
                continue
            
            # Sort by frame
            video_metadata = video_metadata.sort_values('frame')
            
            # Prepare trajectory data
            trajectory = []
            
            for _, row in video_metadata.iterrows():
                frame_id = row['frame']
                
                # Determine current phase
                phase_id = None
                for i in range(7):
                    phase_col = f'p{i}'
                    if phase_col in row and row[phase_col] == 1:
                        phase_id = i
                        break
                
                # Get risk score
                risk_score = row[risk_column_name] if risk_column_name in row else None
                
                # Frame data
                frame_data = {
                    'frame_id': frame_id,
                    'phase_id': phase_id,
                    'risk_scores': {risk_column_name: risk_score} if risk_score is not None else {}
                }
                
                trajectory.append(frame_data)
            
            # Analyze trajectory to get rewards
            trajectory_analysis = self.analyze_trajectory(trajectory)
            
            # Add to training data
            video_data = {
                'video_id': video_id,
                'trajectory': trajectory,
                'rewards': trajectory_analysis['rewards'],
                'cumulative_reward': trajectory_analysis['cumulative_reward'],
                'statistics': trajectory_analysis['statistics']
            }
            
            training_data.append(video_data)
        
        return training_data


# Example usage:
if __name__ == "__main__":
    # Initialize reward function with custom weights
    reward_function = SurgicalRewardFunction(
        phase_weights={
            0: 0.5,  # Preparation phase less important
            1: 1.0,
            2: 2.0,  # Critical phase with double importance
            3: 1.5,
            4: 1.0,
            5: 0.8,
            6: 0.5   # Final phase less important
        },
        risk_weight=2.0,  # Risk is twice as important as default
        critical_risk_threshold=3.5  # Lower threshold for critical risk
    )
    
    # Load historical data (if available)
    reward_function.load_historical_data("/home/maxboels/datasets/CholecT50/embeddings_train_set/fold0/embeddings_f0_swin_bas_129.csv")
    
    # Example frame data
    frame_data = {
        'phase_id': 2,
        'risk_scores': {'risk_score_max': 2.5, 'risk_score_mean': 1.8},
        'frame_id': 1000,
        'progress_within_phase': 0.6  # 60% through current phase
    }
    
    # Calculate reward
    reward, components = reward_function.calculate_reward(frame_data)
    print(f"Reward: {reward}")
    print("Reward components:")
    for component, value in components.items():
        print(f"  {component}: {value}")