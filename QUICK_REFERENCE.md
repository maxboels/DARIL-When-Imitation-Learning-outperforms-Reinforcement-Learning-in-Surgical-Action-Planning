# 🚀 DARIL Quick Reference

**One-page guide to navigating the DARIL repository**

---

## 📂 Where is...?

| Looking for... | Go to... |
|----------------|----------|
| 🔧 **Configuration files** | `configs/` |
| ▶️ **Scripts to run** | `scripts/` |
| 💻 **Source code** | `src/` |
| 📊 **Visualizations** | `notebooks/` |
| 📖 **Documentation** | `docs/` |
| 📈 **Results & outputs** | `outputs/` |
| 🗃️ **Raw data** | `data/` |
| 🐳 **Docker files** | `docker/` |
| 🗄️ **Old code** | `archive/` |

---

## 🎯 Common Tasks

### Train DARIL model
```bash
python scripts/run_experiment_v8.py \
    --config configs/config_dgx_all_v8.yaml \
    --data_path data/cholect50
```

### Evaluate trained model
```bash
python scripts/evaluate.py \
    --checkpoint outputs/models_saved/best_model.pt \
    --horizon 10
```

### Generate paper figures
```bash
python scripts/run_paper_generation.py \
    --results_dir outputs/results \
    --output_dir outputs/figures
```

### Run on GPU cluster
```bash
bash scripts/runai.sh
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `README.md` | Main documentation |
| `STRUCTURE.md` | Detailed structure guide |
| `CHANGELOG.md` | Version history |
| `configs/config_dgx_all_v8.yaml` | Main experiment config |
| `scripts/run_experiment_v8.py` | Training entry point |
| `src/training/autoregressive_il_trainer.py` | DARIL implementation |
| `src/evaluation/` | Evaluation metrics |
| `outputs/results/` | Experiment results |

---

## 🔍 Code Organization

```
src/
├── training/           # How models learn
│   └── autoregressive_il_trainer.py ← DARIL is here
├── evaluation/         # How models are tested
├── models/            # Model architectures
├── environment/       # RL environments
├── utils/             # Helper functions
└── debugging/         # Debug tools
```

---

## 📊 Outputs

All generated files go to `outputs/`:

```
outputs/
├── results/           # Metrics, logs, analysis
├── models_saved/      # Trained checkpoints (.pt)
├── logs/              # Training logs, TensorBoard
├── figures/           # Generated plots
└── data/              # Processed datasets
```

⚠️ **Note:** `outputs/` is gitignored (too large)

---

## 🎓 Learning Path

1. **Start here:** `README.md` - Overview
2. **Understand structure:** `STRUCTURE.md` - Detailed guide
3. **See the code:** `src/training/autoregressive_il_trainer.py` - DARIL
4. **Run an experiment:** `scripts/run_experiment_v8.py`
5. **Check results:** `outputs/results/`
6. **Read the paper:** `docs/paper_manuscript/` or [arXiv](https://arxiv.org/abs/2507.05011)

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't find a file | Check `STRUCTURE.md` |
| Import errors | Use `from src.module import ...` |
| Path not found | Update paths to new structure (see `CHANGELOG.md`) |
| Training fails | Check `outputs/logs/` |
| Need old code | Look in `archive/` |

---

## 🔗 Quick Links

- **Paper:** [arXiv:2507.05011](https://arxiv.org/abs/2507.05011)
- **Dataset:** [CholecT50](http://camma.u-strasbg.fr/datasets)
- **GitHub:** [DARIL Repository](https://github.com/maxboels/DARIL-When-Imitation-Learning-outperforms-Reinforcement-Learning-in-Surgical-Action-Planning)

---

## 📝 File Naming

- **Config:** `config_<context>_v<version>.yaml`
- **Script:** `run_<action>.py`
- **Module:** `<noun>.py` or `<noun>_<description>.py`
- **Output:** `<timestamp>_<description>/`

---

## ⚡ Pro Tips

1. **Use tab completion** - Folder names are descriptive
2. **Read STRUCTURE.md** - Comprehensive guide
3. **Check outputs/** - All results go here
4. **Don't commit outputs/** - It's gitignored
5. **Archive old code** - Don't delete, move to `archive/`

---

**Last updated:** October 23, 2025  
**Version:** 2.0.0

For more details, see [STRUCTURE.md](STRUCTURE.md)
