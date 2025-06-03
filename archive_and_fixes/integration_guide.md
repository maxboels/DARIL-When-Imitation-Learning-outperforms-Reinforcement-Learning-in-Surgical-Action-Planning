# 🤖 Surgical World Model RL Integration Guide

This guide shows you how to integrate RL training with your existing surgical world model codebase to compare **Imitation Learning vs Reinforcement Learning** approaches.

## 📋 Overview

Your research will compare:
1. **Baseline Imitation Learning** (using your existing world model)
2. **PPO** (Proximal Policy Optimization)  
3. **SAC** (Soft Actor-Critic)
4. **TD-MPC2** (coming soon)

## 🚀 Quick Start

### Step 1: Setup Environment
```bash
# Install additional RL dependencies
pip install -r requirements_rl.txt

# Setup integration files
python setup_rl_experiment.py
```

### Step 2: Verify Integration
```bash
# Test that everything works
python test_rl_integration.py
```

### Step 3: Run Full Experiment
```bash
# Run the complete RL comparison
python run_integrated_rl_experiment.py
```

### Step 4: Analyze Results
```bash
# Generate plots and analysis
python analyze_rl_results.py
```

## 📁 File Structure

Add these new files to your existing project:

```
your_project/
├── rl_environment.py          # New: Gym environment wrapper
├── rl_trainer.py              # New: RL training pipeline  
├── run_rl_experiments.py      # New: Main RL experiment runner
├── config_rl.yaml             # New: Enhanced config with RL settings
├── analyze_rl_results.py      # New: Results analysis
├── setup_rl_experiment.py     # New: Setup script
├── test_rl_integration.py     # New: Integration test
└── (your existing files...)
```

## ⚙️ Configuration Changes

Update your `config_rl.yaml` to enable RL experiments:

```yaml
# Enable RL experiments
rl_experiments:
  enabled: true
  algorithms: ['imitation_learning', 'ppo', 'sac']
  timesteps: 50000
  
# Ensure reward learning is enabled in world model
models:
  world_model:
    reward_learning: true  # CRITICAL: Must be true for RL
    action_learning: true
    phase_learning: true
```

## 🎯 How It Works

### 1. Environment Integration
The `SurgicalWorldModelEnv` wraps your existing `WorldModel`:

```python
# Your world model becomes the environment dynamics
env = SurgicalWorldModelEnv(your_world_model, config)

# RL agent interacts with the environment
obs, reward, done, info = env.step(action)
```

### 2. Reward Aggregation
Multiple reward signals are combined with configurable weights:

```python
reward_weights = {
    '_r_phase_completion': 1.0,    # Task completion
    '_r_phase_progression': 1.0,   # Progress toward goals
    '_r_risk': -0.5,              # Safety penalty
    # ... other rewards
}
```

### 3. Comparison Framework
The system automatically compares:
- **Imitation Learning**: Uses your world model's action prediction
- **RL Algorithms**: Train policies to maximize cumulative reward

## 🔧 Customization Options

### Reward Weighting
Adjust reward importance in `config_rl.yaml`:
```yaml
reward_weights:
  '_r_phase_completion': 2.0     # Double weight for task completion
  '_r_risk': -1.0               # Stronger safety penalty
```

### Algorithm Parameters
Fine-tune RL algorithms:
```yaml
ppo:
  learning_rate: 1e-4           # Lower learning rate
  n_steps: 4096                 # More steps per update
  
sac:
  buffer_size: 200000           # Larger replay buffer
  batch_size: 512               # Larger batches
```

### Episode Length
Control experiment length:
```yaml
rl_experiments:
  horizon: 100                  # Longer episodes
  timesteps: 100000             # More training steps
```

## 📊 Expected Results

After running experiments, you'll get:

### 1. Quantitative Comparison
```
=== EXPERIMENT SUMMARY ===
Imitation Learning: avg_reward=0.245
PPO: avg_reward=0.312 (+27.3% improvement)  
SAC: avg_reward=0.289 (+18.0% improvement)
```

### 2. Detailed Analysis
- Episode reward trajectories
- Phase-wise performance breakdown  
- Statistical significance tests
- Visualizations and plots

### 3. Research Insights
- Does RL improve over IL for surgical tasks?
- Which RL algorithm works best?
- How important are different reward components?

## 🐛 Troubleshooting

### Common Issues

**1. "World model not found"**
```bash
# Make sure your world model path is correct in config_rl.yaml
world_model:
  best_model_path: "path/to/your/trained/model.pt"
```

**2. "Reward learning not enabled"**
```yaml
# Ensure in your world model config:
world_model:
  reward_learning: true
```

**3. "CUDA out of memory"**
```yaml
# Reduce batch sizes in config:
training:
  batch_size: 8
rl_experiments:
  timesteps: 25000
```

**4. "No improvement over baseline"**
- Try different reward weights
- Increase training timesteps
- Check if world model is properly trained

### Debug Mode
Enable debug logging:
```yaml
debug: true
```

## 📈 Advanced Usage

### Custom Reward Functions
Add your own reward components:
```python
# In rl_environment.py
def custom_reward(self, state, action, info):
    # Your custom reward logic
    return custom_reward_value
```

### Video-Specific Evaluation
Test on specific surgical videos:
```python
# Reset environment with specific video
obs, _ = env.reset(options={'video_id': 'VID01'})
```

### Hyperparameter Tuning
Use Optuna or similar for automated tuning:
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3)
    # ... run experiment with suggested hyperparameters
    return final_reward
```

## 📚 Research Questions

Your experiments will help answer:

1. **Does RL outperform IL for surgical tasks?**
   - Compare final episode rewards
   - Analyze learning curves
   
2. **Which RL algorithm is most suitable?**
   - PPO (on-policy) vs SAC (off-policy)
   - Sample efficiency comparison
   
3. **What reward components matter most?**
   - Ablation studies with different weights
   - Phase-specific reward analysis

4. **How does the world model quality affect RL?**
   - Compare with different world model checkpoints
   - Analyze prediction accuracy vs RL performance

## 🎓 Publication Ready Results

The framework generates publication-ready:
- **Tables**: Quantitative comparison metrics
- **Figures**: Learning curves and performance plots  
- **Statistical Tests**: Significance analysis
- **Ablation Studies**: Component-wise analysis

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_rl_integration.py` to verify setup
3. Enable debug mode in config for detailed logs
4. Check that your world model has `reward_learning=True`

## 🎉 Success Metrics

You'll know it's working when you see:
- ✅ World model loads without errors
- ✅ Environment completes episodes
- ✅ RL algorithms train without crashes  
- ✅ Results show meaningful comparisons
- ✅ Plots and analysis generate successfully

---

**Next Steps**: Run `python setup_rl_experiment.py` to get started! 🚀