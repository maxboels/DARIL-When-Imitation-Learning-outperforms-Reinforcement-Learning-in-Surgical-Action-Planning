### HIGH-LEVEL INTENDED STRATEGY (BASED ON OUR DISCUSSION)

# 1. Use your trained world model to simulate rollouts:
#     (o_t, a_t) -> (o_{t+1}, r_t)
#
# 2. Sample actions from a trainable policy \pi_\phi(a_t | context)
#
# 3. Store simulated transitions in a buffer: (o_t, a_t, r_t, o_{t+1})
#
# 4. Optimise \pi_\phi using policy gradient (e.g., PPO), Q-learning, or other RL methods
#     to maximise predicted cumulative rewards r_t.
#
# 5. Optionally initialise \pi_\phi with behaviour cloning from expert trajectories,
#     then fine-tune via model-based RL to surpass the expert.

# IMPLEMENTATION: WorldModelEnv Wrapper
import torch
import gym
import numpy as np
from gym import spaces

class WorldModelEnv(gym.Env):
    """
    A Gym-style environment wrapper around the trained WorldModel.
    This allows rollout-based interaction using a simulated world.
    """
    def __init__(self, world_model, horizon=50, device='cuda'):
        super().__init__()
        self.model = world_model.eval()
        self.device = device
        self.horizon = horizon

        # Define observation and action spaces (assumes one-hot action encoding)
        embedding_dim = self.model.embedding_dim
        num_actions = self.model.num_action_classes

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(num_actions)

        # Initialise state
        self.reset()

    def reset(self):
        # Randomly initialise from offline dataset or synthetic init
        self.current_embedding = torch.randn(1, 1, self.model.embedding_dim).to(self.device)
        self.current_action = torch.zeros(1, 1, self.model.num_action_classes).bernoulli_(0.5).to(self.device)
        self.step_count = 0
        return self.current_embedding.squeeze(0).squeeze(0).cpu().numpy()

    def step(self, action):
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Forward pass: predict next embedding and rewards
            output = self.model(
                current_state=self.current_embedding,
                next_actions=action_tensor,
                eval_mode='basic'
            )
            next_embedding = output['_z_hat'][:, -1:, :].detach()
            reward = 0.0
            if self.model.reward_learning:
                # Use one or more reward heads to compute reward
                reward += output.get('_r_phase_completion_loss', 0.0).item() * -1.0  # negated since it's a loss
                reward += output.get('_r_phase_progression_loss', 0.0).item() * -1.0

        self.current_embedding = next_embedding.detach()
        self.current_action = action_tensor.detach()
        self.step_count += 1

        done = self.step_count >= self.horizon
        obs = next_embedding.squeeze(0).squeeze(0).cpu().numpy()
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

# Next step: Define PPO-compatible policy, rollout buffer, and training loop.
