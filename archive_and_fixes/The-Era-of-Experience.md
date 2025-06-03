# Surgical Robotics World Model

## Project Overview
This project develops a non-ceilinged reward function and world model for surgical robotics using the CholecT50 dataset of cholecystectomy videos. The core approach combines grounded, environment-derived signals with a learned reward network through bi-level optimization and inverse reinforcement learning (IRL).

## Theoretical Foundation
The project is based on principles from the "Era of Experience" paradigm (Silver & Sutton, 2025), where:
- Agents learn from their own experiences rather than just human examples
- Rewards are grounded in environmental feedback, not human prejudgment
- Learning systems can discover strategies beyond initial expert performance

## Core Components

### 1. Grounded Surgical Signals
We extract the following from surgical videos and sensor data:
- **Procedure progress**: Phase transitions detected from action-triplet labels
- **Efficiency metrics**: Path length, average speed, smoothness of instruments
- **Safety metrics**: Tissue collisions, force measurements
- **Outcome proxies**: Completion time, bleeding area, occlusion events

### 2. Reward Function
```
R_φ(s_t, a_t, s_{t+1}) = MLP_φ(f_surgical(s_t, a_t, s_{t+1}))
```
Where:
- `f_surgical(·)` is a vector of the metrics above
- `MLP_φ` is a neural network mapping those metrics to a scalar reward

### 3. Bi-level Optimization & IRL
The reward function parameters φ are learned through:
- **Inverse RL**: Making expert demonstrations have higher cumulative reward than agent rollouts
- **Bi-level optimization**: 
  - Outer loop: Adjust φ to maximize sparse surgeon feedback
  - Inner loop: Train policy π_θ to maximize cumulative R_φ

### 4. World Model Architecture
We implement a latent dynamics model (based on Dreamer/PlaNet) with:
- **Encoder** `z_t = E_ψ(o_t)` from video frame to latent
- **Dynamics** `(z_{t+1}, r_t) = D_ψ(z_t, a_t)` for latent transitions and rewards
- **Decoder** to reconstruct observations or predict auxiliary targets

## Implementation Guidelines

### Data Preprocessing
```python
# Example preprocessing pipeline
def preprocess_cholect50_video(video_path):
    # Segment instruments and anatomy
    segmentation_masks = UNet_segmentation(video_path)
    
    # Track instrument trajectories
    instrument_tracks = track_instruments(segmentation_masks)
    
    # Compute surgical features for each frame
    features = []
    for t in range(num_frames):
        features.append({
            "in_phase_progress": one_hot(phases[t+1]),  # next-phase indicator
            "path_length": delta_path_length(t, t+1),
            "jerk": compute_jerk(t), 
            "collision": collision_flag(t),
            # Additional features...
        })
    
    return features, instrument_tracks, segmentation_masks
```

### Reward Network
```python
class RewardHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features):
        return self.mlp(features)
```

### World Model
```python
class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim):
        super().__init__()
        # Encoder: observation -> latent state
        self.encoder = nn.Sequential(...)
        
        # Dynamics: predicts next latent state and reward
        self.dynamics = nn.Sequential(...)
        
        # Decoder: reconstructs observation from latent
        self.decoder = nn.Sequential(...)
        
    def encode(self, obs):
        return self.encoder(obs)
        
    def dynamics_step(self, latent, action):
        # Return next_latent, predicted_reward
        return self.dynamics(torch.cat([latent, action], dim=-1))
        
    def decode(self, latent):
        return self.decoder(latent)
```

### Bi-level Optimization Loop
```python
def bilevel_optimization(expert_trajectories, world_model, policy, reward_head):
    for iteration in range(NUM_ITERATIONS):
        # 1. INNER LOOP: Policy optimization
        rollouts = generate_rollouts(world_model, policy)
        optimize_policy(policy, rollouts, reward_head)
        
        # 2. OUTER LOOP: Reward refinement
        expert_returns = compute_returns(expert_trajectories, reward_head)
        agent_returns = compute_returns(rollouts, reward_head)
        
        # IRL loss: maximize gap between expert and agent returns
        irl_loss = agent_returns - expert_returns + lambda_reg * regularization(reward_head)
        optimize_reward(reward_head, irl_loss)
        
        # 3. (Optional) Human feedback adaptation
        if new_surgeon_feedback_available():
            adapt_reward_to_feedback(reward_head, surgeon_feedback)
```

## Critical Implementation Notes

1. **Avoid Reward Collapse**: Make sure the reward function doesn't degenerate into trivial solutions by using proper regularization.

2. **Balance Inner/Outer Updates**: Typically 5 policy updates per reward update works well.

3. **Warm-Start Reward Function**: Initialize from global decomposition solution to start from a reasonable baseline.

4. **Monitor the Expert-Agent Margin**: You want a positive but stable margin between expert and agent returns.

5. **Validation**: Ensure expert trajectories still score higher under the learned reward than agent trajectories.

## Approach for Limited Data (CholecT50)

Since we only have 50 videos:

1. **Feature Engineering is Critical**: Extract rich, domain-specific features to make learning more sample-efficient.

2. **Data Augmentation**: Consider perturbations of instrument trajectories or synthetic variations.

3. **Regularize Heavily**: Use strong regularization on all model components.

4. **World Model Simplification**: Consider simpler dynamics models like RSSMs rather than large Transformers.

5. **Transfer Learning**: Pre-train components on related datasets if available.

## Code Structure

```
/src
  /preprocessing
    - segment_instruments.py
    - extract_features.py
    - compute_metrics.py
  /models
    - reward_network.py
    - world_model.py
    - policy.py
  /training
    - bilevel_optimization.py
    - irl.py
    - policy_optimization.py
  /evaluation
    - visualize_rewards.py
    - evaluate_policy.py
/data
  /cholect50
    - videos/
    - annotations/
    - processed_features/
/configs
  - training_config.yaml
  - model_config.yaml
```

## Reference Architecture Choices

For the world model component, consider:
- **RSSM/Dreamer architecture** for sample efficiency
- **MuZero-style dynamics** for planning capabilities
- **Transformer-based models** only if you have sufficient data

For policy optimization, consider:
- **PPO** for stable policy improvements
- **Model predictive control (MPC)** for zero-shot planning with the model

## Dependencies
- PyTorch
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib (for visualization)
