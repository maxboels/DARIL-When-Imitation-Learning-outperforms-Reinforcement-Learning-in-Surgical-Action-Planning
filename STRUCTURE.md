# Repository Structure Guide

This document provides a detailed overview of the DARIL repository organization.

## 📂 Directory Overview

### Root Level

```
DARIL/
├── README.md           # Main project documentation
├── .gitignore          # Git ignore patterns
├── STRUCTURE.md        # This file
├── configs/            # Configuration files
├── scripts/            # Executable scripts
├── src/                # Core source code
├── notebooks/          # Interactive demos
├── docs/               # Documentation
├── outputs/            # Generated outputs (gitignored)
├── data/               # Raw datasets (user-provided)
├── docker/             # Container configurations
└── archive/            # Historical versions
```

## 🔍 Detailed Structure

### `configs/` - Configuration Files
Contains YAML configuration files for different experiments:
- `config_dgx_all_v8.yaml` - Main experimental configuration
- `config_dgx_all.yaml` - Alternative configurations
- Defines hyperparameters, model settings, data paths

**Usage:** Reference these files when running experiments
```bash
python scripts/run_experiment_v8.py --config configs/config_dgx_all_v8.yaml
```

---

### `scripts/` - Executable Scripts
Entry points for training, evaluation, and analysis:

#### Main Scripts:
- `run_experiment_v8.py` - Primary training/evaluation pipeline
- `run_paper_generation.py` - Generate all paper figures
- `runai.sh` - GPU cluster job submission
- `run_file.sh` - Batch processing utilities
- `delete_jobs.sh` - Cluster job management

**Usage:** Run experiments from project root
```bash
python scripts/run_experiment_v8.py --method daril
```

---

### `src/` - Core Source Code
Main implementation modules organized by functionality:

#### `src/training/` - Training Modules
- `autoregressive_il_trainer.py` - **DARIL model trainer** (main IL approach)
- `world_model_trainer.py` - World model learning (Dreamer-inspired)
- `world_model_rl_trainer.py` - RL training in learned world models
- `irl_direct_trainer.py` - Inverse Reinforcement Learning
- `irl_next_action_trainer.py` - IRL for next action prediction
- `archive/` - Previous implementations and experimental variants

#### `src/evaluation/` - Evaluation Framework
- Metrics computation (mAP, precision, recall)
- Multi-horizon evaluation (1s, 2s, 3s, 5s, 10s, 20s)
- Comparison frameworks for IL vs RL
- Performance analysis tools

#### `src/models/` - Model Architectures
- MHA (Multi-Head Attention) encoder for current action recognition
- GPT-2 decoder for autoregressive future prediction
- World model architectures (VAE-based dynamics models)
- RL policy networks (PPO, SAC implementations)
- Baseline models for comparison

#### `src/environment/` - RL Environments
- Surgical action planning environment (Gym-compatible)
- State representations from video features
- Action space definitions (100 IVT triplet classes)
- Reward functions (expert-similarity, outcome-based)
- Episode management and trajectory handling

#### `src/utils/` - Utility Functions
- `metrics.py` - Evaluation metrics
- `logger.py` - Training logging
- `visualization.py` - Plotting utilities
- `optimizer_scheduler.py` - Learning rate schedules
- `rl_optimization.py` - RL-specific utilities
- Various plotting and analysis tools

#### `src/debugging/` - Debug Tools
- `rl_debug_tools.py` - RL training diagnostics
- `rl_diagnostic_script.py` - Troubleshooting utilities

---

### `notebooks/` - Interactive Visualizations
HTML-based interactive demos and Jupyter notebooks:

- `enhanced_interactive_viz.html` - Interactive surgical action visualizer
- `interactive_surgical_grid.html` - Grid-based action timeline
- `updated_visualization.html` - Updated visualization dashboard
- `visualization/` - Visualization module code
  - `surgical_action_visualizer.py` - Visualization generation
  - `map_horizon_plotter.py` - Multi-horizon mAP plots
  - `archive/` - Previous visualization versions

**Usage:** Open HTML files in browser for interactive exploration

---

### `docs/` - Documentation

#### `docs/paper_manuscript/` - Paper Source
- LaTeX source files
- MICCAI 2025 COLAS Workshop submission
- Figures, tables, and supplementary materials

#### `docs/paper_notes/` - Research Notes
- `rl_mechanics_explanation.md` - RL implementation details
- `strategic_improvement_plan.md` - Future directions
- `safety_guardrails_framework.md` - Clinical safety considerations
- `repo_structure_design.md` - This repository's design philosophy
- Various planning and analysis documents

#### `docs/paper_generation/` - Figure Generation
- `paper_generator.py` - Automated figure generation
- Scripts to create publication-ready visualizations

---

### `outputs/` - Generated Outputs (Gitignored)
Contains all experiment outputs - **not tracked by Git**:

#### `outputs/results/` - Evaluation Results
- Experiment logs organized by timestamp
- JSON files with metrics
- Evaluation reports
- Comparative analysis data

#### `outputs/models_saved/` - Model Checkpoints
- Trained model weights (`.pt` files)
- Best model checkpoints
- Training state for resuming

#### `outputs/logs/` - Training Logs
- TensorBoard event files
- Text logs
- Training progress tracking

#### `outputs/figures/` - Generated Figures
- Publication figures
- Training curves
- Evaluation visualizations

#### `outputs/data/` - Processed Data
- `enhanced_data/` - Augmented dataset versions
- `datasets/` - Processed features
- Cached computations

**Note:** These directories are created automatically during training and are excluded from version control due to size.

---

### `data/` - Raw Datasets
User-provided raw data (not included in repository):

```
data/
└── cholect50/          # CholecT50 dataset
    ├── video_01/       # Swin features per video
    ├── video_02/
    └── ...
```

**Setup:** Download CholecT50 from [CAMMA](http://camma.u-strasbg.fr/datasets) and place extracted features here.

---

### `docker/` - Docker Configurations
Container definitions for reproducible environments:
- Dockerfile for training environment
- Docker Compose configurations
- GPU-enabled container setup

---

### `archive/` - Historical Code
Legacy implementations and experimental code:
- Previous experiment configurations
- Deprecated model implementations
- Research explorations
- Backup code

**Note:** Code here is kept for reference but may not be maintained.

---

## 🚀 Quick Navigation

### Want to...

**Train a model?**
→ `scripts/run_experiment_v8.py` + `configs/config_dgx_all_v8.yaml`

**Understand the DARIL model?**
→ `src/training/autoregressive_il_trainer.py` + `src/models/`

**Evaluate results?**
→ `src/evaluation/` + check `outputs/results/`

**Create visualizations?**
→ `notebooks/visualization/` + `docs/paper_generation/`

**Modify RL environments?**
→ `src/environment/`

**Debug training issues?**
→ `src/debugging/` + `outputs/logs/`

---

## 📝 File Naming Conventions

- **Scripts:** Descriptive action-based names (`run_experiment_v8.py`)
- **Modules:** Noun-based names (`trainer.py`, `evaluator.py`)
- **Configs:** Context + version (`config_dgx_all_v8.yaml`)
- **Outputs:** Timestamped directories (`2025-07-07_00-11-00/`)

---

## 🔄 Data Flow

```
data/cholect50 
    ↓
[Feature Extraction]
    ↓
src/training/autoregressive_il_trainer.py
    ↓
outputs/models_saved/best_model.pt
    ↓
src/evaluation/
    ↓
outputs/results/metrics.json
    ↓
docs/paper_generation/paper_generator.py
    ↓
outputs/figures/publication_figures/
```

---

## 🛠️ Development Workflow

1. **Configure:** Edit `configs/config_dgx_all_v8.yaml`
2. **Train:** Run `python scripts/run_experiment_v8.py`
3. **Monitor:** Check `outputs/logs/` and TensorBoard
4. **Evaluate:** Metrics saved to `outputs/results/`
5. **Visualize:** Generate figures with `scripts/run_paper_generation.py`
6. **Analyze:** Review results in `outputs/figures/`

---

## 📊 Size Guidelines

- **Small files** (< 1MB): Track in Git (code, configs, docs)
- **Medium files** (1-100MB): Exclude from Git, store in `outputs/`
- **Large files** (> 100MB): User-provided in `data/`, external storage

---

## 🎯 Best Practices

1. **Never commit to `outputs/`** - Auto-generated content
2. **Version control configs** - Track experiment setups
3. **Document in `docs/paper_notes/`** - Research decisions
4. **Archive old code** - Move to `archive/` rather than delete
5. **Use absolute imports** - From project root: `from src.models import ...`

---

## 🔗 Related Documentation

- [README.md](README.md) - Main project documentation
- [docs/paper_manuscript/](docs/paper_manuscript/) - Paper LaTeX source
- [Paper (arXiv)](https://arxiv.org/abs/2507.05011) - Published work

---

**Last Updated:** October 23, 2025  
**Repository:** [DARIL-When-Imitation-Learning-outperforms-Reinforcement-Learning-in-Surgical-Action-Planning](https://github.com/maxboels/DARIL-When-Imitation-Learning-outperforms-Reinforcement-Learning-in-Surgical-Action-Planning)
