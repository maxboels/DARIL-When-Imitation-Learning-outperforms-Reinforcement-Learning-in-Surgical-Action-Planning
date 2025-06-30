# Repository Cleanup Instructions

## 🎯 Goal: Create Clean Structure for IL vs RL Comparison

Your repository has significant duplication and clutter. Follow these instructions to create a clean, focused structure for comparing:
1. **Imitation Learning** (baseline)
2. **RL with World Model** (our main approach)
3. **RL without World Model** (ablation study)

## 📁 Target Clean Structure

```
surgical_action_prediction/
├── README.md
├── requirements.txt
├── config.yaml                    # Single clean config
├── main.py                       # Main experiment runner
│
├── datasets/
│   ├── __init__.py
│   ├── cholect50.py              # Keep this - it's good
│   └── preprocessing/            # Rename from current preprocessing files
│       ├── __init__.py
│       ├── phase_completion.py
│       ├── progression.py
│       └── risk_scores.py
│
├── models/
│   ├── __init__.py
│   └── dual_world_model.py       # Keep - this is correct
│
├── environments/
│   ├── __init__.py
│   ├── world_model_env.py        # From rl/environment.py
│   └── direct_env.py             # New - for non-world-model RL
│
├── training/
│   ├── __init__.py
│   ├── il_trainer.py             # Clean IL trainer
│   ├── rl_trainer.py             # Clean RL trainer
│   └── world_model_trainer.py    # For world model pre-training
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluator.py              # Single clean evaluator
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Keep
│   └── visualization.py
│
├── experiments/
│   ├── __init__.py
│   ├── run_il_baseline.py
│   ├── run_rl_with_world_model.py
│   ├── run_rl_without_world_model.py
│   └── run_comparison.py
│
├── configs/
│   ├── il_config.yaml
│   ├── rl_world_model_config.yaml
│   └── rl_direct_config.yaml
│
└── results/                      # Clean results directory
    ├── il_baseline/
    ├── rl_world_model/
    ├── rl_direct/
    └── comparison/
```

## 🗂️ File Migration Instructions

### Phase 1: Keep These Files (Copy to New Structure)

```bash
# Core Model (KEEP)
cp models/dual_world_model.py NEW_REPO/models/

# Correct Environment (KEEP)
cp rl/environment.py NEW_REPO/environments/world_model_env.py

# Dataset Loader (KEEP)
cp datasets/cholect50.py NEW_REPO/datasets/

# Preprocessing (KEEP - Consolidate)
cp datasets/preprocess_phase_completion.py NEW_REPO/datasets/preprocessing/phase_completion.py
cp datasets/preprocess_progression.py NEW_REPO/datasets/preprocessing/progression.py
cp datasets/preprocess_risk_scores.py NEW_REPO/datasets/preprocessing/risk_scores.py

# Utils (KEEP)
cp utils/logger.py NEW_REPO/utils/
cp utils/metrics.py NEW_REPO/evaluation/metrics.py

# One Good Config (CHOOSE BEST)
cp config.yaml NEW_REPO/config.yaml
```

### Phase 2: Delete These Redundant Files/Folders

```bash
# Delete all backup/copy files
rm -rf *copy*
rm -rf *backup*
rm -rf *old*

# Delete redundant configs (keep only config.yaml)
rm config_dgx_*.yaml
rm config_local_*.yaml
rm config_rl*.yaml
rm config_xps.yaml

# Delete redundant trainers (we'll create clean ones)
rm final_fixed_trainer.py
rm fixed_sb3_trainer.py
rm sb3_rl_trainer*
rm surgical_world_model_rl.py

# Delete old experiments
rm -rf main_experiment*
rm -rf run_main_experiment*
rm -rf updated_main_experiment.py
rm -rf working_experiment.py

# Delete redundant evaluation files
rm -rf evaluation/dual_evaluation_framework*copy*
rm corrected_evaluation_framework.py
rm corrected_rl_vs_il_evaluation.py
rm fixed_dual_evaluation.py

# Delete old results (keep only latest/best)
rm -rf logs/2025-05-*  # Keep only June results
rm -rf results/
rm -rf evaluation_results/
rm -rf clinical_evaluation_results/

# Delete visualization clutter
rm -rf surgical_animations/
rm -rf api/
rm -rf figures/reward_visualizations/
rm -rf figures/risk_visualizations/

# Delete development files
rm -rf __pycache__/
rm -rf test_*.py
rm -rf debug_*.py
rm -rf diagnose_*.py
rm -rf emergency_patch.py
rm -rf patch_fixes.py
```

### Phase 3: Create New Clean Files

#### NEW_REPO/main.py
```python
#!/usr/bin/env python3
"""
Main experiment runner for IL vs RL comparison
"""

import argparse
from experiments.run_il_baseline import run_il_experiment
from experiments.run_rl_with_world_model import run_rl_world_model_experiment  
from experiments.run_rl_without_world_model import run_rl_direct_experiment
from experiments.run_comparison import run_full_comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=[
        'il_baseline', 
        'rl_world_model', 
        'rl_direct', 
        'comparison'
    ], required=True)
    parser.add_argument('--config', default='config.yaml')
    
    args = parser.parse_args()
    
    if args.experiment == 'il_baseline':
        run_il_experiment(args.config)
    elif args.experiment == 'rl_world_model':
        run_rl_world_model_experiment(args.config)
    elif args.experiment == 'rl_direct':
        run_rl_direct_experiment(args.config)
    elif args.experiment == 'comparison':
        run_full_comparison(args.config)

if __name__ == "__main__":
    main()
```

#### NEW_REPO/environments/direct_env.py
```python
"""
Direct RL Environment (without world model)
For ablation study comparison
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List

class DirectSurgicalEnv(gym.Env):
    """
    RL Environment that directly steps through video frames
    (not using world model simulation)
    """
    
    def __init__(self, video_data: List[Dict], config: Dict):
        super().__init__()
        self.video_data = video_data
        self.config = config
        
        # Action and observation spaces
        self.action_space = gym.spaces.MultiBinary(100)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(1024,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to random video and frame"""
        super().reset(seed=seed)
        
        # Select random video and frame
        self.current_video_idx = np.random.randint(len(self.video_data))
        video = self.video_data[self.current_video_idx]
        
        max_start = len(video['frame_embeddings']) - 50
        self.current_frame_idx = np.random.randint(0, max(1, max_start))
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Get initial observation
        observation = video['frame_embeddings'][self.current_frame_idx]
        
        return observation.astype(np.float32), {}
    
    def step(self, action):
        """Step using direct frame progression (NO world model)"""
        video = self.video_data[self.current_video_idx]
        
        # Move to next frame directly
        self.current_frame_idx += 1
        self.step_count += 1
        
        # Check termination
        done = (self.current_frame_idx >= len(video['frame_embeddings']) - 1 or 
                self.step_count >= 50)
        
        if done:
            next_obs = video['frame_embeddings'][self.current_frame_idx - 1]
            reward = 0.0
        else:
            next_obs = video['frame_embeddings'][self.current_frame_idx]
            
            # Simple reward based on action similarity to expert
            expert_action = video['actions_binaries'][self.current_frame_idx]
            action_similarity = np.mean(action == expert_action)
            reward = action_similarity - 0.5  # Center around 0
        
        self.episode_reward += reward
        
        info = {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'uses_world_model': False  # Key difference!
        }
        
        return next_obs.astype(np.float32), reward, done, False, info
```

## 🧪 Experimental Comparison Setup

Create these three experiments for a comprehensive comparison:

### Experiment 1: IL Baseline
- Train world model in supervised mode on expert demonstrations
- Evaluate action prediction accuracy (mAP)

### Experiment 2: RL with World Model (Our Main Approach)
- Use trained world model as environment simulator
- Train RL agent (PPO/SAC) in simulated environment
- True model-based RL

### Experiment 3: RL without World Model (Ablation Study)  
- Train RL agent directly on video frame sequences
- No world model simulation
- Direct policy learning

### Experiment 4: Comprehensive Comparison
- Compare all three approaches
- Analyze when world model helps vs hurts
- Publication-ready results

## 📋 Cleanup Execution Script

```bash
#!/bin/bash
# Repository cleanup script

echo "🧹 Starting repository cleanup..."

# Create new clean directory
mkdir -p surgical_action_prediction_clean
cd surgical_action_prediction_clean

# Create directory structure
mkdir -p datasets/preprocessing
mkdir -p models
mkdir -p environments  
mkdir -p training
mkdir -p evaluation
mkdir -p utils
mkdir -p experiments
mkdir -p configs
mkdir -p results/{il_baseline,rl_world_model,rl_direct,comparison}

# Copy essential files
echo "📁 Copying essential files..."

# Copy core model
cp ../models/dual_world_model.py models/

# Copy correct environment
cp ../rl/environment.py environments/world_model_env.py

# Copy dataset
cp ../datasets/cholect50.py datasets/

# Copy preprocessing
cp ../datasets/preprocess_phase_completion.py datasets/preprocessing/phase_completion.py
cp ../datasets/preprocess_progression.py datasets/preprocessing/progression.py  
cp ../datasets/preprocess_risk_scores.py datasets/preprocessing/risk_scores.py

# Copy utils
cp ../utils/logger.py utils/
cp ../utils/metrics.py evaluation/metrics.py

# Copy best config
cp ../config.yaml .

echo "✅ Repository cleaned and organized!"
echo "📂 New structure created in: surgical_action_prediction_clean/"
echo ""
echo "🎯 Next steps:"
echo "1. Implement the new experiment files"
echo "2. Create clean trainers"  
echo "3. Run three-way comparison"
echo "4. Generate publication results"
```

## 🎯 Benefits of This Structure

1. **Clean Separation**: Each experiment type is isolated
2. **Ablation Study**: Can compare with/without world model
3. **Reproducible**: Clear configs and experiment scripts
4. **Publication Ready**: Organized results structure
5. **Maintainable**: No more duplicate files

## 🚀 Implementation Priority

1. **Phase 1**: Execute cleanup script above
2. **Phase 2**: Implement new experiment files
3. **Phase 3**: Run three-way comparison
4. **Phase 4**: Generate publication results

This will give you a much cleaner codebase and enable a more comprehensive experimental comparison including the ablation study of RL with vs without world model simulation.
