# Using Your World Model for RL-Based Policy Learning

Based on your trained world model and goals, I'll guide you through implementing a reinforcement learning approach to improve upon auto-regressive action prediction. Let's break this down into concrete steps.

## Approach Overview

You're in a great position since you already have a trained world model that can predict:
- Next states
- Next actions (imitation learning)
- Next phases
- Various rewards

Your world model can serve as a simulated environment for RL training, which is perfect for offline learning from your surgical videos.

## Step 1: Create a Policy Model

First, let's create a separate policy model that will learn to select actions:

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # For binary actions
        )
        
    def forward(self, state):
        return self.network(state)
    
    def sample_action(self, state, deterministic=False):
        action_probs = self.forward(state)
        if deterministic:
            return (action_probs > 0.5).float()
        else:
            return torch.bernoulli(action_probs)
```

## Step 2: Define a Simple RL Environment Using Your World Model

Create a wrapper class that uses your world model as a simulator:

```python
class WorldModelEnv:
    def __init__(self, world_model, reward_weights=None, max_steps=50):
        self.world_model = world_model
        self.world_model.eval()  # Set to evaluation mode
        self.max_steps = max_steps
        self.current_step = 0
        self.current_state = None
        
        # Default reward weights if not provided
        self.reward_weights = reward_weights or {
            '_r_phase_completion': 1.0,
            '_r_phase_progression': 0.5,
            '_r_risk': -0.7,  # Negative weight for risk (we want to minimize risk)
            '_r_action_probability': 0.2,
            '_r_global_progression': 0.3
        }
    
    def reset(self, initial_state=None):
        """Reset the environment with an optional initial state."""
        if initial_state is None:
            # You'll need a way to sample initial states from your dataset
            raise NotImplementedError("Need to implement initial state sampling")
        
        self.current_state = initial_state
        self.current_step = 0
        return self.current_state
    
    def step(self, action):
        """Take a step in the environment using the world model."""
        with torch.no_grad():
            # Format inputs for the world model
            current_state = self.current_state.unsqueeze(0)  # Add batch dimension
            action = action.unsqueeze(0)  # Add batch dimension
            
            # Use the world model to predict the next state and rewards
            # This assumes your world model has a method to predict next state given current state + action
            outputs = self.world_model(
                current_state=current_state,
                next_actions=action
            )
            
            # Extract next state prediction
            next_state = outputs['_z_hat'].squeeze(0)
            
            # Calculate rewards from the various reward components
            reward = 0
            for reward_key, weight in self.reward_weights.items():
                if reward_key in outputs['head_outputs']:
                    reward_component = outputs['head_outputs'][reward_key].squeeze().item()
                    reward += weight * reward_component
            
            # Update current state
            self.current_state = next_state
            self.current_step += 1
            
            # Check if done
            done = self.current_step >= self.max_steps
            
            return next_state, reward, done, {}
```

## Step 3: Implement a Simple RL Algorithm (PPO)

For a not-too-complicated approach, let's use Proximal Policy Optimization (PPO):

```python
def train_ppo(world_model_env, policy_model, num_episodes=1000, gamma=0.99, 
              clip_epsilon=0.2, policy_lr=0.0003, device='cuda'):
    """Train the policy using PPO algorithm with the world model as the environment."""
    optimizer = optim.Adam(policy_model.parameters(), lr=policy_lr)
    
    for episode in range(num_episodes):
        # Sample an initial state from your dataset
        initial_state = sample_initial_state()  # You'll need to implement this
        state = world_model_env.reset(initial_state)
        
        # Collect trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        done = False
        
        while not done:
            # Convert state to tensor on device
            state_tensor = torch.FloatTensor(state).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action_probs = policy_model(state_tensor)
                action = torch.bernoulli(action_probs)
                log_prob = (action * torch.log(action_probs + 1e-10) + 
                           (1 - action) * torch.log(1 - action_probs + 1e-10)).sum(-1)
            
            # Take step in environment
            next_state, reward, done, _ = world_model_env.step(action)
            
            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            # Update state
            state = next_state
        
        # Calculate returns
        returns = calculate_returns(rewards, gamma)
        
        # Update policy
        update_policy_ppo(policy_model, optimizer, states, actions, log_probs, returns, clip_epsilon)
        
        # Log progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Return: {sum(returns)/len(returns)}")
    
    return policy_model

def calculate_returns(rewards, gamma):
    """Calculate discounted returns."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def update_policy_ppo(policy_model, optimizer, states, actions, old_log_probs, returns, clip_epsilon):
    """Update policy using PPO objective."""
    states = torch.FloatTensor(states).to(next(policy_model.parameters()).device)
    actions = torch.FloatTensor(actions).to(next(policy_model.parameters()).device)
    old_log_probs = torch.stack(old_log_probs).detach()
    returns = torch.FloatTensor(returns).to(next(policy_model.parameters()).device)
    
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Get new action probabilities
    action_probs = policy_model(states)
    new_log_probs = (actions * torch.log(action_probs + 1e-10) + 
                    (1 - actions) * torch.log(1 - action_probs + 1e-10)).sum(-1)
    
    # Calculate ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped objective
    clip_advantage = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * returns
    full_advantage = ratio * returns
    loss = -torch.min(full_advantage, clip_advantage).mean()
    
    # Update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Step 4: Experiment Setup to Compare Methods

Now let's create a function to compare your two approaches:

```python
def compare_methods(world_model, test_data, device='cuda'):
    """Compare auto-regressive action prediction with RL-based policy."""
    # Create policy model and train it
    state_dim = world_model.embedding_dim
    action_dim = world_model.num_action_classes
    
    # 1. Initialize policy network
    policy_model = PolicyNetwork(state_dim, action_dim).to(device)
    
    # 2. Create environment
    world_model_env = WorldModelEnv(world_model)
    
    # 3. Train policy using RL
    trained_policy = train_ppo(world_model_env, policy_model, device=device)
    
    # 4. Evaluate both methods on test data
    autoregressive_results = evaluate_autoregressive(world_model, test_data, device)
    rl_policy_results = evaluate_rl_policy(trained_policy, world_model, test_data, device)
    
    # 5. Compare results
    print("\nResults Comparison:")
    print("===================")
    print(f"Autoregressive Method - Average Reward: {autoregressive_results['avg_reward']:.4f}")
    print(f"RL Policy Method - Average Reward: {rl_policy_results['avg_reward']:.4f}")
    
    return {
        'autoregressive': autoregressive_results,
        'rl_policy': rl_policy_results
    }
```

## Step 5: Additional Implementations Needed

You'll need to implement a few more functions:

1. **Sampling Initial States:** a function to sample starting points from your dataset
2. **Evaluation Functions:** to compare your methods
3. **A Method to Integrate with Your Dataset:** to load real surgical data for testing

## Practical Implementation Steps

Here's a more practical roadmap:

1. **Implement the Policy Network**: Create the separate policy model.

2. **Create the World Model Environment**: Wrap your world model to act as an environment.

3. **Implement a Simple RL Algorithm**: Start with PPO as outlined.

4. **Setup Data Sampling**: Create functions to sample initial states and trajectories from your dataset.

5. **Train-Evaluate Cycle**:
   - Train the policy with RL
   - Compare with auto-regressive baseline
   - Analyze differences in behavior

6. **Visualization**: Create visualizations of:
   - Reward curves during training
   - Action distributions from both methods
   - State trajectories to see qualitative differences

## Advanced Extensions (After the Basic Implementation)

Once you have the basic implementation working, you could explore:

1. **Constraint-based RL**: Add safety constraints from your risk scores

2. **Hierarchical RL**: Learn high-level policies for phase transitions and low-level policies for actions within phases

3. **Model Predictive Control**: Use your world model for planning multiple steps ahead

4. **Offline RL Methods**: Implement conservative Q-learning (CQL) or behavior constrained Q-learning (BCQ) for better offline learning

Would you like me to expand on any specific part of this approach? Or would you prefer more concrete code for a particular component?