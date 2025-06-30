# RL Mechanics: Episodes, Rewards, and Expert Matching

## 🎯 1. Completion Bonus - How It Works

The completion bonus is a **terminal reward** that encourages the RL agent to complete full episodes rather than getting stuck or taking actions that lead to early termination.

### Implementation:
```python
# In rl_environments.py - step() method
def step(self, action):
    self.current_step += 1
    
    # Check if episode should end
    frames_remaining = len(self.current_video['frame_embeddings']) - self.current_frame_idx - 1
    done = (self.current_step >= self.max_episode_steps) or (frames_remaining <= 0)
    
    if done:
        # Episode completion bonus - only given at the END
        reward = self.reward_weights['completion_bonus']  # Usually +5.0
        next_state = self.current_state.copy()
    else:
        # Normal step rewards (expert matching, etc.)
        reward = self._calculate_meaningful_reward(action, predicted_rewards)
        # Update to next frame...
```

### Why This Matters:
- **Prevents premature termination**: Without this, agents might learn to end episodes early
- **Encourages task completion**: Surgical procedures should be completed, not abandoned
- **Balances exploration**: Agents get a reward boost for seeing full sequences through

### Example Timeline:
```
Step 1: Expert matching reward: +2.3
Step 2: Expert matching reward: +1.8  
Step 3: Expert matching reward: +3.1
...
Step 25: Expert matching reward: +2.5 + completion_bonus: +5.0 = +7.5 total
```

## 📏 2. Episode Length and Termination

Episodes are **flexible** and can end in multiple ways. They're not fixed-length sequences.

### Episode Termination Conditions:

```python
# From rl_environments.py
frames_remaining = len(video['frame_embeddings']) - self.current_frame_idx - 1
step_limit_reached = self.current_step >= self.max_episode_steps

# Episode ends if EITHER condition is true:
terminated = step_limit_reached    # Hit step limit (e.g., 30 steps)
truncated = frames_remaining <= 0  # Ran out of video frames
done = terminated or truncated
```

### 3 Ways Episodes Can End:

#### Option 1: **Step Limit Reached** (Most Common)
```
Video has 200 frames, max_episode_steps = 30
Episode: Frame 50 → 51 → 52 → ... → 80 (30 steps total)
Result: terminated=True, truncated=False
Reason: Reached 30-step limit
```

#### Option 2: **Video Exhausted** (Less Common)
```
Video has 50 frames, start at frame 40, max_episode_steps = 30  
Episode: Frame 40 → 41 → ... → 49 (10 steps, hit end of video)
Result: terminated=False, truncated=True
Reason: No more frames available
```

#### Option 3: **Perfect Timing** (Rare)
```
Both conditions hit simultaneously
Result: terminated=True, truncated=True
```

### Episode Mechanics:
```python
def reset(self):
    # Select random video and starting frame
    self.current_video_idx = random.choice(video_indices)
    
    # Ensure enough frames for full episode
    max_start = len(video_frames) - self.max_episode_steps - 5
    self.current_frame_idx = random.randint(0, max_start)
    
    return initial_state

def step(self, action):
    # Move through video frames sequentially
    self.current_frame_idx += 1  # Next frame in video
    self.current_step += 1       # Next step in episode
    
    # Get real frame or simulate with world model
    next_state = get_next_frame_or_simulate()
    
    return next_state, reward, done, info
```

### Key Insights:
- **Random starting points**: Each episode starts at a different frame in the video
- **Sequential progression**: Moves through video frames in order (like real surgery)
- **Variable length**: Episodes naturally vary in length based on starting position
- **Realistic constraint**: Can't go beyond available video data

## 🎓 3. Expert Matching vs Supervised Learning

This is a crucial conceptual difference! Both use expert demonstrations but in fundamentally different ways.

### Supervised Learning (Method 1 - Autoregressive IL):

```python
# Direct mapping: Input → Expert Output
def supervised_training():
    for batch in dataloader:
        input_frames = batch['current_frames']    # Input
        expert_actions = batch['expert_actions']  # Target
        
        predicted_actions = model(input_frames)
        loss = cross_entropy(predicted_actions, expert_actions)  # Direct loss
        loss.backward()
```

**Characteristics:**
- **Direct supervision**: "Given this state, do exactly this action"
- **No exploration**: Only learns from expert examples
- **Deterministic**: Same input always produces same output
- **Fast convergence**: Direct gradient updates toward expert behavior

### RL with Expert Matching (Methods 2 & 3):

```python
# Reward-based learning: Action → Reward → Policy Update
def rl_training():
    for episode in episodes:
        state = env.reset()
        for step in episode:
            action = policy.predict(state)  # Agent chooses action
            next_state, reward, done = env.step(action)
            
            # Reward based on how well action matches expert
            expert_action = get_expert_action(state)
            reward = calculate_expert_matching(action, expert_action)
            
            # Learn from experience tuple (state, action, reward, next_state)
            policy.update(state, action, reward, next_state)
```

**Characteristics:**
- **Indirect supervision**: "Here's how good your action was" (not what to do)
- **Exploration encouraged**: Agent tries different actions and learns from consequences
- **Stochastic**: Can explore beyond expert demonstrations
- **Slower convergence**: Must learn through trial and error

### Key Differences Illustrated:

| Aspect | Supervised IL | RL + Expert Matching |
|--------|---------------|---------------------|
| **Learning Signal** | "Do this action" | "That action scored X points" |
| **Exploration** | None | Encouraged |
| **Beyond Expert** | Cannot | Can discover better strategies |
| **Training Data** | State-action pairs | State-action-reward-next_state |
| **Convergence** | Fast | Slower but more robust |
| **Generalization** | Limited to expert patterns | Can generalize beyond expert |

### Example Scenario:

**Situation**: Agent sees surgical state where expert used action [1, 0, 1, 0, ...]

#### Supervised Learning Response:
```python
"The expert did [1, 0, 1, 0, ...] so I should do exactly [1, 0, 1, 0, ...]"
# Always outputs the same thing for this state
```

#### RL + Expert Matching Response:
```python
"Let me try [1, 0, 1, 0, ...] → Reward: +8.5 (high expert matching)"
"Let me try [1, 1, 1, 0, ...] → Reward: +3.2 (lower expert matching)"  
"Let me try [1, 0, 0, 0, ...] → Reward: +6.8 (partial expert matching)"

# Learns: "Actions closer to [1, 0, 1, 0, ...] get higher rewards"
# But can still explore variations and potentially find improvements
```

### Why Use RL + Expert Matching Instead of Pure Supervised?

1. **Robustness**: RL can handle situations not seen in expert data
2. **Exploration**: May discover strategies better than expert demonstrations
3. **Sequential decision making**: Better at planning multi-step consequences
4. **Real-world deployment**: More adaptable to variations in real scenarios

### The Reward Function Connection:

```python
def expert_matching_reward(agent_action, expert_action):
    # How similar is the agent's action to the expert's?
    accuracy = np.mean(agent_action == expert_action)
    
    # But also reward partial matches and reasonable variations
    if np.sum(expert_action) > 0:  # If expert took any actions
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * precision * recall / (precision + recall)
        
        return base_reward * accuracy + bonus_reward * f1_score
    
    return base_reward * accuracy
```

This rewards not just perfect matches, but also "reasonable" surgical actions that align with expert intent even if not identical.

The key insight: **Supervised learning teaches "what to do"**, while **RL with expert rewards teaches "how to evaluate what you did"** - which often leads to more robust and adaptable behavior!