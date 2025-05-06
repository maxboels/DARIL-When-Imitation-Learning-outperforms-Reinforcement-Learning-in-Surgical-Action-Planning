import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# -- Simplicial Normalization Layer (SimNorm) --
class SimNorm(nn.Module):
    def __init__(self, dim, tau=0.5, group_size=16):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.group_size = group_size

    def forward(self, z):
        # z: [B, T, D]
        B, T, D = z.shape
        V = self.group_size
        z = z.view(B, T, D // V, V)
        exp_z = torch.exp(z / self.tau)
        g = exp_z / (exp_z.sum(-1, keepdim=True) + 1e-6)
        return g.view(B, T, D)

# -- Q-Ensemble Module --
class QEnsemble(nn.Module):
    def __init__(self, hidden_dim, action_dim, K=5):
        super().__init__()
        self.q_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(K)
        ])
    def forward(self, h):
        # h: [B, T, H]
        return torch.stack([q(h) for q in self.q_nets], dim=0)  # [K, B, T, A]

# -- Policy Prior Head --
class PolicyPrior(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    def forward(self, h):
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = torch.exp(log_std)
        return mu, std

# -- TD-MPC2 World Model --
class TDMPC2Model(nn.Module):
    def __init__(self, embedding_dim, action_dim, hidden_dim=256, n_layer=6,
                 reward_bins=51, value_bins=51, q_ensemble=5, tau=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input Encoder & SimNorm
        self.encoder = nn.Linear(embedding_dim + action_dim, hidden_dim)
        self.simnorm = SimNorm(hidden_dim, tau=tau)

        # Latent dynamics predictor
        self.dynamics = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Reward & Value Heads
        self.reward_head = nn.Linear(hidden_dim, reward_bins)
        self.value_head = nn.Linear(hidden_dim, value_bins)

        # Q-Ensemble
        self.q_ensemble = QEnsemble(hidden_dim, action_dim, K=q_ensemble)

        # Policy Prior
        self.policy_prior = PolicyPrior(hidden_dim, action_dim)

        # EMA Target Networks
        self._build_target_networks()

    def _build_target_networks(self):
        # Copy Q-ensemble for target
        self.q_target = QEnsemble(self.hidden_dim, self.action_dim, K=len(self.q_ensemble.q_nets))
        for tgt, src in zip(self.q_target.q_nets, self.q_ensemble.q_nets):
            tgt.load_state_dict(src.state_dict())
        self.register_buffer('tau_ema', torch.tensor(0.005))

    @torch.no_grad()
    def update_target(self):
        for tgt, src in zip(self.q_target.q_nets, self.q_ensemble.q_nets):
            for p_t, p in zip(tgt.parameters(), src.parameters()):
                p_t.data.mul_(1 - self.tau_ema).add_(self.tau_ema * p.data)

    def forward(self, states, actions):
        # states: [B, T, E], actions: [B, T, A]
        B, T, _ = states.shape
        x = torch.cat([states, actions], dim=-1)
        h = F.relu(self.encoder(x))       # [B, T, H]
        h = self.simnorm(h)
        h_seq, _ = self.dynamics(h)       # [B, T, H]

        # Predictions
        r_logits = self.reward_head(h_seq)
        v_logits = self.value_head(h_seq)
        q_logits = self.q_ensemble(h_seq) # [K, B, T, A]
        mu, std = self.policy_prior(h_seq)

        return {
            'h': h_seq,
            'reward_logits': r_logits,
            'value_logits': v_logits,
            'q_logits': q_logits,
            'policy_mu': mu,
            'policy_std': std
        }

# -- MPPI Planner --
class MPPIPlanner:
    def __init__(self, model, horizon=8, num_traj=256, gamma=0.99):
        self.model = model
        self.horizon = horizon
        self.num_traj = num_traj
        self.gamma = gamma

    def plan(self, z0):
        # z0: [B, 1, E]
        B = z0.size(0)
        device = z0.device

        mu, std = self.model.policy_prior(z0)  # [B, 1, A]
        mu = mu.unsqueeze(2).expand(-1, -1, self.horizon, -1)  # [B,1,H,A]
        std = std.unsqueeze(2).expand(-1, -1, self.horizon, -1)

        # Sample candidate trajectories
        eps = torch.randn(B, self.num_traj, self.horizon, self.model.action_dim, device=device)
        actions = mu.unsqueeze(1) + eps * std.unsqueeze(1)

        # Rollout
        returns = torch.zeros(B, self.num_traj, device=device)
        z = z0.expand(B, self.num_traj, -1).reshape(B*self.num_traj, 1, -1)
        for t in range(self.horizon):
            a_t = actions[:, :, t].reshape(B*self.num_traj, 1, -1)
            out = self.model(z, a_t)
            r_logits = out['reward_logits'][:, -1]
            q_logits = out['q_logits'][:, :, -1]
            r = torch.softmax(r_logits, dim=-1).argmax(dim=-1).float()
            v = torch.softmax(q_logits, dim=-1).argmax(dim=-1).float()
            returns += (self.gamma**t) * r.view(B, self.num_traj)
            z = out['h']  # next latent
        returns += (self.gamma**self.horizon) * v.view(B, self.num_traj)

        # Select top trajectories
        weights = F.softmax(returns / returns.std(dim=-1, keepdim=True), dim=-1)
        best_a0 = (weights.unsqueeze(-1) * actions[:, :, 0]).sum(1)
        return best_a0

# -- Training Loop (Offline Replay) --
def train_tdmpc2(model, dataset, epochs=100, batch_size=32, lr=3e-4, gamma=0.99, lam=0.95):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    planner = MPPIPlanner(model)

    for epoch in range(epochs):
        for batch in loader:
            s = batch['z']       # [B, T, E]
            a = batch['_a']      # [B, T, A]
            s_next = batch['_z'] # [B, T, E]
            r = batch.get('reward', torch.zeros(s.size(0), s.size(1)))
            
            out = model(s, a)
            h = out['h']
            r_logits = out['reward_logits']
            v_logits = out['value_logits']
            q_logits = out['q_logits']   # [K, B, T, A]

            # Compute TD targets
            with torch.no_grad():
                # Next state latent
                out_next = model(s_next, a)
                q_target = model.q_target(out_next['h'])  # [K,B,T,A]
                q_min = q_target.min(0)[0]
                td_target = r + gamma * q_min.mean(-1)
                td_target_bins = td_target.long().clamp(0, v_logits.size(-1)-1)

            # Losses
            dyn_loss   = F.mse_loss(h, s_next)
            rew_loss   = F.cross_entropy(r_logits.view(-1, r_logits.size(-1)), r.long().view(-1))
            val_loss   = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), td_target_bins.view(-1))
            q_loss     = F.mse_loss(q_logits.mean(0), td_target.unsqueeze(-1).expand_as(q_logits.mean(0)))

            # Policy prior loss
            mu, std = out['policy_mu'], out['policy_std']
            pi_dist = torch.distributions.Normal(mu, std)
            a_sample = pi_dist.rsample()
            q_pi = model.q_ensemble(h).mean(0)
            pi_loss = -(q_pi.gather(-1, a_sample.argmax(-1, keepdim=True)).squeeze(-1) - 0.01 * pi_dist.entropy()).mean()

            loss = dyn_loss + rew_loss + val_loss + q_loss + pi_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            model.update_target()

        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

# Example dataset stub
class NextFrameDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Usage:
# dataset = NextFrameDataset(my_offline_data)\# list of dicts with 'z','_a','_z','reward'
# model   = TDMPC2Model(embedding_dim=1024, action_dim=100)
# train_tdmpc2(model, dataset)
