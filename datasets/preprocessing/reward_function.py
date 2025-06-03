import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.special import kl_div

class SurgicalRewardFunction:
    """
    A comprehensive reward function for surgical procedures that combines:
    1. Imitation Learning: KL divergence between action distributions
    2. Expert Knowledge: Risk scores from expert annotations
    3. Grounded Signals: Phase progression, instrument usage, duration
    4. Global Outcome: Long-term value prediction
    """
    
    def __init__(self, config=None, action_phase_distributions=None, phase_statistics=None):
        """
        Initialize the reward function based on config settings.
        
        Args:
            config: Configuration dictionary with reward components
            action_phase_distributions: DataFrame with expert action distributions per phase
            phase_statistics: Dictionary with statistics about phases from training data
        """
        # Default config if none provided
        if config is None:
            config = {
                'preprocess': {
                    'rewards': {
                        'imitation_learning': {
                            'action_probability_distribution': True
                        },
                        'prior_expert_knowledge': {
                            'risk_score': True
                        },
                        'grounded_signals': {
                            'blood_loss': False,
                            'phase_progression': True,
                            'instrument_usage': True,
                            'global_duration': True
                        }
                    },
                    'value': {
                        'global_outcome': True
                    }
                }
            }
            
        self.config = config
        self.reward_config = config['preprocess']['rewards']
        self.action_phase_distributions = action_phase_distributions
        self.phase_statistics = phase_statistics or {}
        
        # Component weights (can be adjusted based on importance)
        self.weights = {
            'imitation': 0.7,     # Reduced to rely less on imitation
            'risk': 2.0,          # Higher weight as risk is critical
            'progression': 1.2,    # Increased to emphasize progression
            'instrument': 0.4,    # Reduced to avoid prescriptive rules
            'duration': 0.2,      # Reduced to avoid time pressure
            'global': 1.0
        }
        
        # Track phase information
        self.current_phase = None
        self.phase_start_frames = {}
        self.phase_transition_frames = []
        
        # Record per-frame components for analysis
        self.component_values = defaultdict(dict)
        
        # Track the history of frames for each video
        self.video_frames = defaultdict(list)
        
        # Previous risk scores to track changes
        self.prev_risk_scores = {}
        
        # Keep track of completed phases
        self.completed_phases = set()
    
    def calculate_reward(self, frame_data):
        """
        Calculate the combined reward for a frame.
        
        Args:
            frame_data: Dictionary containing frame information:
                - 'actions': One-hot action vector (tri0-tri99)
                - 'phase_id': Current surgical phase ID (0-6)
                - 'phase_progress': Progress within the current phase (0-1)
                - 'risk_score': Risk score for the frame
                - 'instruments': One-hot instrument vector
                - 'frame_id': Frame ID within the video
                - 'video_id': Video identifier
                
        Returns:
            total_reward: Combined reward value
            components: Dictionary of individual reward components
        """
        # Initialize reward components
        components = {}
        video_id = frame_data.get('video_id')
        frame_id = frame_data.get('frame_id', 0)
        phase_id = frame_data.get('phase_id')
        
        # Track frame for this video
        if video_id is not None:
            self.video_frames[video_id].append(frame_id)
        
        # Check for phase transition
        phase_transition = False
        if phase_id is not None and self.current_phase is not None and phase_id != self.current_phase:
            phase_transition = True
            self.completed_phases.add(self.current_phase)
            self.phase_transition_frames.append(frame_id)
        
        # Update current phase
        if phase_id is not None:
            self.current_phase = phase_id
            if phase_id not in self.phase_start_frames:
                self.phase_start_frames[phase_id] = frame_id
        
        # 1. Imitation Learning Reward - Based on action distribution alignment
        if self.reward_config['imitation_learning']['action_probability_distribution']:
            imitation_reward = self.calculate_imitation_reward(
                frame_data.get('actions'), 
                phase_id
            )
            components['imitation_reward'] = imitation_reward
        else:
            components['imitation_reward'] = 0.0
            
        # 2. Risk-based Reward - Inverse of risk score (lower risk is better)
        if self.reward_config['prior_expert_knowledge']['risk_score']:
            risk_reward = self.calculate_risk_reward(
                frame_data.get('risk_score'), 
                video_id, 
                frame_id
            )
            components['risk_reward'] = risk_reward
        else:
            components['risk_reward'] = 0.0
            
        # 3. Grounded Signals
        
        # 3.1 Phase Progression Reward
        if self.reward_config['grounded_signals']['phase_progression']:
            progression_reward = self.calculate_progression_reward(
                frame_data.get('phase_progress', 0.0),
                phase_transition
            )
            components['progression_reward'] = progression_reward
        else:
            components['progression_reward'] = 0.0
            
        # 3.2 Instrument Usage Reward
        if self.reward_config['grounded_signals']['instrument_usage']:
            instrument_reward = self.calculate_instrument_reward(
                frame_data.get('instruments'),
                phase_id
            )
            components['instrument_reward'] = instrument_reward
        else:
            components['instrument_reward'] = 0.0
            
        # 3.3 Global Duration Efficiency Reward
        if self.reward_config['grounded_signals']['global_duration']:
            duration_reward = self.calculate_duration_reward(
                frame_id, 
                video_id,
                phase_id
            )
            components['duration_reward'] = duration_reward
        else:
            components['duration_reward'] = 0.0
            
        # Phase transition bonus (immediate reward for completing a phase)
        if phase_transition:
            components['transition_bonus'] = 5.0  # Fixed bonus for phase transition
        else:
            components['transition_bonus'] = 0.0
            
        # Calculate total weighted reward
        total_reward = (
            self.weights['imitation'] * components['imitation_reward'] +
            self.weights['risk'] * components['risk_reward'] +
            self.weights['progression'] * components['progression_reward'] +
            self.weights['instrument'] * components['instrument_reward'] +
            self.weights['duration'] * components['duration_reward'] +
            components['transition_bonus']  # Transition bonus is not weighted
        )
        
        # Record components for this frame
        self.component_values[frame_id] = components
        
        return total_reward, components
    
    def calculate_imitation_reward(self, action_vector, phase_id):
        """
        Calculate reward based on KL divergence between predicted action distribution
        and expert action distribution for the current phase.
        
        Args:
            action_vector: One-hot encoded action vector (tri0-tri99)
            phase_id: Current surgical phase ID (0-6)
            
        Returns:
            float: Imitation reward (-KL divergence, higher is better)
        """
        if action_vector is None or phase_id is None or self.action_phase_distributions is None:
            return 0.0
            
        # Get expert action distribution for this phase
        phase_key = f'p{phase_id}'
        if phase_key not in self.action_phase_distributions.index:
            return 0.0
            
        expert_dist = self.action_phase_distributions.loc[phase_key].values
        
        # Ensure action_vector is properly normalized
        action_sum = np.sum(action_vector)
        if action_sum > 0:
            action_dist = action_vector / action_sum
        else:
            return 0.0  # No action detected
            
        # Calculate KL divergence (smaller is better)
        try:
            # Add small epsilon to prevent division by zero or log(0)
            epsilon = 1e-10
            expert_dist = expert_dist + epsilon
            action_dist = action_dist + epsilon
            
            # Normalize again
            expert_dist = expert_dist / np.sum(expert_dist)
            action_dist = action_dist / np.sum(action_dist)
            
            # Calculate KL divergence
            kld = np.sum(expert_dist * np.log(expert_dist / action_dist))
            
            # Convert to reward (negative KL divergence, higher is better)
            # Apply scaling to keep reward in reasonable range
            reward = -min(kld, 10.0)  # Cap at -10 to avoid extreme penalties
            
            # Normalize to a more reasonable range [-1, 1]
            reward = np.clip(reward / 5.0, -1.0, 1.0)
            
            return reward
        except Exception as e:
            print(f"Error calculating KL divergence: {e}")
            return 0.0
    
    def calculate_risk_reward(self, risk_score, video_id, frame_id):
        """
        Calculate reward based on risk score (lower risk is better).
        Also rewards risk reduction over time.
        
        Args:
            risk_score: Risk score for the frame (typically 0-5)
            video_id: Video identifier
            frame_id: Frame identifier
            
        Returns:
            float: Risk-based reward
        """
        if risk_score is None:
            return 0.0
            
        # Convert risk to reward (inverse relationship)
        # Assuming risk is on a 0-5 scale, transform to a 1 to -4 reward
        # No risk (0) gives +1 reward, maximum risk (5) gives -4 reward
        max_risk = 5.0
        risk_reward = 1.0 - risk_score
        
        # Penalize high risk more severely with quadratic penalty
        if risk_score > 3.0:
            risk_reward -= 0.2 * (risk_score - 3.0)**2
            
        # Additional reward for risk reduction over time
        if video_id is not None:
            key = f"{video_id}_{self.current_phase}"
            prev_risk = self.prev_risk_scores.get(key, risk_score)
            
            # Reward for reducing risk
            risk_reduction = prev_risk - risk_score
            if risk_reduction > 0:
                risk_reward += 0.3 * risk_reduction  # Bonus for reducing risk
            
            # Update risk score history
            self.prev_risk_scores[key] = risk_score
            
        return risk_reward
    
    def calculate_progression_reward(self, phase_progress, phase_transition):
        """
        Calculate reward based on progress within the current phase.
        Rewards progress with higher values for completing a phase.
        
        Args:
            phase_progress: Progress within the current phase (0-1)
            phase_transition: Whether this frame represents a phase transition
            
        Returns:
            float: Progression reward
        """
        if phase_progress is None:
            return 0.0
            
        # Base reward for progress within phase
        if phase_transition:
            # Higher reward for completing a phase
            progression_reward = 1.0  # Full reward for completion
        else:
            # Progress reward with diminishing returns
            progression_reward = np.sqrt(phase_progress) * 0.5
            
        return progression_reward
    
    def calculate_instrument_reward(self, instruments, phase_id):
        """
        Calculate reward based on instrument usage variability and activity.
        Instead of assuming optimal counts, we reward using any instruments (activity)
        and using a variety of instruments appropriately (adaptability).
        
        Args:
            instruments: One-hot encoded instrument vector
            phase_id: Current surgical phase (used for tracking, not for prescriptive rules)
            
        Returns:
            float: Instrument usage reward
        """
        if instruments is None:
            return 0.0
            
        # Count instruments used
        num_instruments = np.sum(instruments)
        
        # Basic activity reward - using any instruments is better than none
        if num_instruments == 0:
            return -0.1  # Small penalty for no instruments
        
        # Calculate entropy of instrument distribution as a measure of appropriate variety
        # Higher entropy means more balanced use of different instruments
        if num_instruments > 0:
            instruments_normalized = instruments / num_instruments
            non_zero_mask = instruments_normalized > 0
            if np.any(non_zero_mask):
                entropy = -np.sum(instruments_normalized[non_zero_mask] * 
                                  np.log(instruments_normalized[non_zero_mask]))
                # Normalize entropy to a reasonable range [0, 0.2]
                max_entropy = np.log(len(instruments))  # Max possible entropy
                normalized_entropy = entropy / max_entropy * 0.2
                return normalized_entropy
        
        # Default small positive reward for using instruments
        return 0.05
    
    def calculate_duration_reward(self, frame_id, video_id, phase_id):
        """
        Calculate reward based on time efficiency without strict expectations.
        Instead of using predefined expected durations, we use a gentle time penalty
        to encourage efficient progress without prescribing specific timelines.
        
        Args:
            frame_id: Current frame ID
            video_id: Video identifier
            phase_id: Current phase ID
            
        Returns:
            float: Duration-based reward
        """
        if frame_id is None:
            return 0.0
            
        # Simple time penalty to encourage efficiency
        # This gentle penalty (-0.001 per frame) creates pressure to make progress
        # without prescribing specific time windows for each phase
        time_penalty = -0.001
        
        # We could track time in phase for analytics but don't use it for rewards
        if phase_id is not None and video_id is not None:
            phase_start = self.phase_start_frames.get(phase_id, frame_id)
            time_in_phase = frame_id - phase_start
            
            # Store this information but don't use it for reward calculation
            phase_key = f"{video_id}_{phase_id}"
            self.component_values.get(frame_id, {})['time_in_phase'] = time_in_phase
        
        return time_penalty
    
    def calculate_global_outcome_reward(self, metrics):
        """
        Calculate reward based on global outcome metrics.
        This is typically used for terminal rewards or value estimation.
        
        Args:
            metrics: Dictionary of global outcome metrics
            
        Returns:
            float: Global outcome reward
        """
        if not metrics or 'global_outcome_score' not in metrics:
            return 0.0
            
        # Direct mapping from global outcome score (assumed to be 0-10 scale)
        outcome_score = metrics['global_outcome_score']
        
        # Normalize to [-1, 1] range assuming 0-10 input scale
        normalized_score = (outcome_score / 5.0) - 1.0
        
        return normalized_score
    
    def reset(self):
        """Reset the reward function state for a new procedure"""
        self.current_phase = None
        self.phase_start_frames = {}
        self.phase_transition_frames = []
        self.component_values = defaultdict(dict)
        self.video_frames = defaultdict(list)
        self.prev_risk_scores = {}
        self.completed_phases = set()


# Helper functions for data extraction
def get_phase_id_from_row(row):
    """Extract phase ID from row with phase columns p0-p6"""
    for i in range(7):
        if f'p{i}' in row and row[f'p{i}'] == 1:
            return i
    return None

def get_action_vector_from_row(row):
    """Extract action vector from row with action columns tri0-tri99"""
    action_vector = np.zeros(100)
    for i in range(100):
        col = f'tri{i}'
        if col in row:
            action_vector[i] = row[col]
    return action_vector

def get_instrument_vector_from_row(row):
    """Extract instrument vector from row with instrument columns inst0-inst5"""
    instrument_vector = np.zeros(6)
    for i in range(6):
        col = f'inst{i}'
        if col in row:
            instrument_vector[i] = row[col]
    return instrument_vector

# Function to compute action-phase distribution
def compute_action_phase_distribution(metadata_df, n_phases=7, n_actions=100):
    """
    Compute the distribution of expert actions (tri0…tri99) in each phase (p0…p{n_phases-1})
    across all videos.

    Args:
        metadata_df: DataFrame with metadata
        n_phases: Number of phases
        n_actions: Number of action triplets

    Returns:
        DataFrame with action distributions per phase
    """
    # Build column lists
    phase_cols  = [f"p{p}" for p in range(n_phases)]
    action_cols = [f"tri{i}" for i in range(n_actions)]

    # Prepare a DataFrame to hold the distributions
    dist_df = pd.DataFrame(
        data=0.0, 
        index=[f"p{p}" for p in range(n_phases)],
        columns=[f"tri{i}" for i in range(n_actions)]
    )

    # For each phase, sum up action counts and normalize
    for p, pcol in enumerate(phase_cols):
        df_phase = metadata_df[metadata_df[pcol] == 1.0]
        if df_phase.empty:
            continue
        counts = df_phase[action_cols].sum(axis=0).astype(float)

        # Add small non-zero probabilities to all actions
        for i in range(n_actions):
            if counts[f"tri{i}"] == 0:
                counts[f"tri{i}"] = 1e-8
        
        # Normalize counts to get probabilities
        total = counts.sum()
        if total > 0:
            dist_df.loc[f"p{p}"] = counts / total
        
    # Fill NaN values with 0.0
    dist_df.fillna(0.0, inplace=True)
    return dist_df.astype(float)

def add_rewards_to_metadata(metadata_df, config=None, video_ids=None):
    """
    Add reward columns to the metadata DataFrame.
    
    Args:
        metadata_df: DataFrame with metadata
        config: Configuration dictionary
        video_ids: List of video IDs to process (None for all)
        
    Returns:
        DataFrame with added reward columns
    """
    # Compute action-phase distribution
    print("Computing action-phase distribution...")
    action_dist_df = compute_action_phase_distribution(metadata_df)
    
    # Create reward function
    reward_fn = SurgicalRewardFunction(config, action_dist_df)
    
    # Get list of videos to process
    if video_ids is None:
        video_ids = metadata_df['video'].unique()
    
    # Process each video
    print(f"Processing {len(video_ids)} videos...")
    enhanced_metadata = []
    
    for video_id in video_ids:
        # Reset reward function for new video
        reward_fn.reset()
        
        # Get video metadata
        video_df = metadata_df[metadata_df['video'] == video_id].sort_values('frame')
        
        # Total and component rewards for this video
        video_rewards = []
        video_components = {}
        
        # Process each frame
        for _, row in video_df.iterrows():
            # Create frame data from row
            frame_data = {
                'video_id': video_id,
                'frame_id': row['frame'],
                'phase_id': get_phase_id_from_row(row),
                'phase_progress': row.get('phase_progression', 0.0),
                'risk_score': row.get('risk_score_max', None),
                'actions': get_action_vector_from_row(row),
                'instruments': get_instrument_vector_from_row(row)
            }
            
            # Calculate reward
            reward, components = reward_fn.calculate_reward(frame_data)
            
            # Store reward and components
            video_rewards.append(reward)
            
            # Initialize component columns if needed
            for key, value in components.items():
                if key not in video_components:
                    video_components[key] = []
                video_components[key].append(value)
        
        # Add rewards to video dataframe
        video_df = video_df.copy()
        video_df['total_reward'] = video_rewards
        
        # Add component rewards
        for key, values in video_components.items():
            video_df[key] = values
        
        # Calculate cumulative reward
        video_df['cumulative_reward'] = np.cumsum(video_rewards)
        
        # Add to enhanced metadata
        enhanced_metadata.append(video_df)
    
    # Combine all enhanced metadata
    enhanced_df = pd.concat(enhanced_metadata)
    
    return enhanced_df