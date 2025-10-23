# DARIL: When Imitation Learning outperforms Reinforcement Learning in Surgical Action Planning

[![arXiv](https://img.shields.io/badge/arXiv-2507.05011-b31b1b.svg)](https://arxiv.org/abs/2507.05011)
[![MICCAI 2025](https://img.shields.io/badge/MICCAI%202025-COLAS%20Workshop-blue)](https://arxiv.org/abs/2507.05011)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** [Maxence Boels](https://github.com/maxenceboels), Harry Robertshaw, Thomas C Booth, Prokar Dasgupta, Alejandro Granados, Sebastien Ourselin  
**Affiliation:** Surgical and Interventional Engineering, King's College London

---

## 📋 Overview

This repository contains the official implementation of **DARIL (Dual-task Autoregressive Imitation Learning)**, presenting the first comprehensive comparison of Imitation Learning (IL) versus Reinforcement Learning (RL) for surgical action planning. Our work challenges conventional assumptions about RL superiority in sequential decision-making tasks.

### Key Contributions

- **First systematic IL vs RL comparison** for surgical action planning on CholecT50 dataset
- **Novel DARIL architecture** combining dual-task learning with autoregressive prediction
- **Surprising findings**: IL consistently outperforms sophisticated RL approaches (world models, direct video RL, inverse RL)
- **Critical insights** on evaluation bias and distribution matching in expert domains

---

## 🎯 Problem Statement

Surgical action planning predicts future **instrument-verb-target (IVT) triplets** from laparoscopic video feeds for real-time surgical assistance. Unlike recognition tasks, planning requires multi-horizon prediction under safety-critical constraints with sparse annotations (100 distinct triplet classes, 0-3 simultaneous actions per frame).

---

## 🏗️ Architecture

DARIL combines three key components:

1. **MHA Encoder** - Temporal processing for current action recognition
2. **GPT-2 Decoder** - Causal autoregressive generation for future action prediction (20-frame context window)
3. **Dual-task Optimization** - Joint training on recognition + prediction with auxiliary losses

```python
L = L_current + L_next + L_embed + L_phase
```

**Input:** 1024-dim Swin Transformer features  
**Output:** Multi-horizon IVT triplet predictions (1s, 2s, 3s, 5s, 10s, 20s)

---

## 📊 Results

### Main Findings (IVT mAP %)

| Method | Current | 1s | 5s | 10s |
|--------|---------|----|----|-----|
| **DARIL (Ours)** | **34.6** | **33.6** | **31.2** | **29.2** |
| DARIL + IRL | 33.1 | 32.1 | 29.6 | 28.1 |
| Direct Video RL | 33.2 | 22.6 | 19.3 | 15.9 |
| World Model RL | 33.1 | 14.0 | 9.1 | **3.1** |

### Component-wise Performance

| Component | Current | Next |
|-----------|---------|------|
| Instrument (I) | 91.4 | 88.2 |
| Verb (V) | 69.4 | 68.1 |
| Target (T) | 52.7 | 52.5 |
| **IVT** | **34.6** | **33.6** |

**Key Insight:** DARIL maintains robust temporal consistency with only 13.1% relative performance decrease from 1s to 10s planning horizons, while world model RL catastrophically degrades to 3.1% mAP.

---

## 🔍 Why RL Underperformed

Our analysis identifies critical factors:

1. **Expert-Optimal Demonstrations** - CholecT50 contains near-optimal expert data; RL explores valid alternatives penalized by expert-similarity metrics
2. **Evaluation Metric Alignment** - Test metrics directly reward expert-like behavior, systematically favoring IL
3. **State-Action Representation Challenges** - Frame embeddings + discrete action triplets + sparse rewards limit RL learning
4. **Distribution Mismatch** - RL policies optimized for different objectives produce behaviors misaligned with test distributions
5. **Limited Exploration Benefits** - Safety constraints and expert optimality reduce advantages from exploration

---

## 🚀 Implications for Surgical AI

- **Method Selection:** Well-optimized IL may outperform sophisticated RL in expert domains with high-quality demonstrations
- **Hybrid Approaches:** Bootstrap RL with IL-learned skills, explore safely in simulation/world models
- **Safety Advantages:** IL inherently stays closer to expert behavior for clinical deployment
- **Evaluation Frameworks:** Alternative metrics focusing on patient outcomes (beyond expert similarity) may favor RL

---

## 📁 Repository Structure

The repository is organized for easy navigation and reproducibility. **See [STRUCTURE.md](STRUCTURE.md) for detailed documentation.**

```
DARIL/
├── README.md                    # This file
├── STRUCTURE.md                 # Detailed structure guide
├── .gitignore                   # Git ignore rules
│
├── 📂 configs/                  # Configuration files
│   ├── config_dgx_all_v8.yaml  # Main experiment config
│   └── config_dgx_all.yaml     # Alternative configs
│
├── 📂 scripts/                  # Executable scripts
│   ├── run_experiment_v8.py    # Main experiment runner
│   ├── run_paper_generation.py # Paper figure generator
│   ├── runai.sh                # GPU cluster scripts
│   └── *.sh                    # Shell scripts
│
├── 📂 src/                      # Core source code
│   ├── training/               # Training modules
│   │   ├── autoregressive_il_trainer.py      # DARIL trainer
│   │   ├── world_model_trainer.py            # World model RL
│   │   ├── world_model_rl_trainer.py         # RL in world models
│   │   ├── irl_direct_trainer.py             # Inverse RL
│   │   └── irl_next_action_trainer.py        # IRL variants
│   ├── evaluation/             # Evaluation framework
│   ├── models/                 # Model architectures
│   ├── environment/            # RL environments
│   ├── utils/                  # Utility functions
│   └── debugging/              # Debug tools
│
├── 📂 notebooks/                # Interactive visualizations
│   ├── visualization/          # Visualization modules
│   └── *.html                  # Interactive HTML demos
│
├── 📂 docs/                     # Documentation
│   ├── paper_manuscript/       # LaTeX paper source
│   ├── paper_notes/            # Research notes
│   └── paper_generation/       # Figure generation
│
├── 📂 outputs/                  # Experiment outputs (gitignored)
│   ├── results/                # Evaluation results
│   ├── models_saved/           # Trained model checkpoints
│   ├── logs/                   # Training logs
│   ├── figures/                # Generated figures
│   └── data/                   # Processed datasets
│
├── 📂 data/                     # Raw dataset (user-provided)
│   └── cholect50/              # CholecT50 video features
│
├── 📂 docker/                   # Docker configurations
│
└── 📂 archive/                  # Historical code versions
```

### Key Files

- **Main Scripts:**
  - `scripts/run_experiment_v8.py` - Primary training/evaluation pipeline
  - `scripts/run_paper_generation.py` - Generate paper figures
  
- **Core Implementations:**
  - `src/training/autoregressive_il_trainer.py` - DARIL model
  - `src/models/` - Model architectures (MHA encoder, GPT-2 decoder)
  - `src/evaluation/` - Evaluation metrics and pipelines
  
- **Configuration:**
  - `configs/config_dgx_all_v8.yaml` - Main experimental setup

---

## �🛠️ Installation

```bash
git clone https://github.com/yourusername/DARIL.git
cd DARIL
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.12+
- Transformers (GPT-2)
- timm (Swin Transformer)
- Standard ML libraries (numpy, pandas, scikit-learn)

---

## 📦 Dataset

**CholecT50:** 50 laparoscopic cholecystectomy videos with frame-level annotations
- Training: 40 videos (78,968 frames)
- Testing: 10 videos (21,895 frames)
- Sampling: 1 FPS
- Classes: 100 distinct IVT triplets

[Download CholecT50](http://camma.u-strasbg.fr/datasets)

---

## 🏃 Usage

### Quick Start

```bash
# Train DARIL model
python scripts/run_experiment_v8.py --config configs/config_dgx_all_v8.yaml --data_path /path/to/cholect50

# Evaluate on test set
python scripts/evaluate.py --checkpoint outputs/models_saved/daril_best.pth --horizon 10

# Run multi-horizon planning evaluation
python scripts/evaluate_planning.py --checkpoint outputs/models_saved/daril_best.pth
```

### Training from Scratch

```bash
# DARIL baseline (Imitation Learning)
python scripts/run_experiment_v8.py \
    --method daril \
    --config configs/config_dgx_all_v8.yaml \
    --epochs 100 \
    --lr 1e-4

# Direct Video RL
python scripts/run_experiment_v8.py \
    --method direct_video_rl \
    --config configs/config_rl.yaml

# World Model RL (Dreamer-based)
python scripts/run_experiment_v8.py \
    --method world_model_rl \
    --config configs/config_world_model.yaml

# Inverse RL
python scripts/run_experiment_v8.py \
    --method inverse_rl \
    --config configs/config_irl.yaml
```

### Generating Paper Figures

```bash
# Generate all figures for the paper
python scripts/run_paper_generation.py \
    --results_dir outputs/results \
    --output_dir outputs/figures
```

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{boels2025daril,
  title={DARIL: When Imitation Learning outperforms Reinforcement Learning in Surgical Action Planning},
  author={Boels, Maxence and Robertshaw, Harry and Booth, Thomas C and Dasgupta, Prokar and Granados, Alejandro and Ourselin, Sebastien},
  journal={arXiv preprint arXiv:2507.05011},
  year={2025},
  note={Accepted at MICCAI 2025 COLAS Workshop}
}
```

---

## 🔗 Links

- **Paper:** [arXiv:2507.05011](https://arxiv.org/abs/2507.05011)
- **Conference:** [MICCAI 2025 COLAS Workshop](https://colas-workshop.github.io/)
- **Dataset:** [CholecT50](http://camma.u-strasbg.fr/datasets)
- **Lab:** [Surgical & Interventional Engineering, KCL](https://www.kcl.ac.uk/bmeis/our-research/surgical-interventional-engineering)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for:
- Bug fixes
- Feature enhancements
- Improved RL implementations
- Extensions to other surgical datasets

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- CholecT50 dataset by CAMMA, University of Strasbourg
- Pre-trained Swin Transformer models
- OpenAI GPT-2 architecture
- Dreamer world model implementation

---

## ⚠️ Limitations & Future Work

1. Single dataset evaluation (CholecT50) - generalization to other procedures needed
2. Expert test data may favor IL - sub-expert scenarios unexplored
3. Evaluation metrics reward expert-like behavior - outcome-focused metrics needed
4. RL state-action representations require further optimization
5. Limited dataset size may cause overfitting - larger datasets and simulators needed

**Future Directions:** Cross-dataset evaluation, outcome-based metrics, physics simulators, comprehensive state-action-reward modeling

---

## 📧 Contact

For questions or collaboration:
- Maxence Boels: maxence.boels@kcl.ac.uk
- GitHub Issues: [Open an issue](https://github.com/yourusername/DARIL/issues)