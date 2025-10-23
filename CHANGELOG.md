# Changelog

All notable changes to the DARIL repository structure and codebase.

## [2.0.0] - 2025-10-23

### 🎉 Major Repository Reorganization

Complete restructuring of the repository for improved navigation and maintainability.

#### Changed
- **Reduced root-level items from 33 to 12** for cleaner organization
- Consolidated scattered files into logical directory grouping

#### Added
- `configs/` - Centralized configuration files (moved from root)
- `scripts/` - All executable scripts in one place
- `src/` - Core source code with clear module separation
  - `src/training/` - Training implementations
  - `src/evaluation/` - Evaluation framework
  - `src/models/` - Model architectures
  - `src/environment/` - RL environments
  - `src/utils/` - Utility functions
  - `src/debugging/` - Debug tools
- `notebooks/` - Interactive visualizations and demos
- `docs/` - All documentation in one place
  - `docs/paper_manuscript/` - LaTeX paper source
  - `docs/paper_notes/` - Research notes
  - `docs/paper_generation/` - Figure generation scripts
- `outputs/` - Unified output directory
  - `outputs/results/` - Evaluation results
  - `outputs/models_saved/` - Model checkpoints
  - `outputs/logs/` - Training logs
  - `outputs/figures/` - Generated figures
  - `outputs/data/` - Processed datasets
- `STRUCTURE.md` - Comprehensive repository documentation
- `CHANGELOG.md` - This file

#### Removed
- `api/` folder (HTML files moved to `notebooks/`)
- `web_app/` folder (HTML files moved to `notebooks/`)
- `__pycache__/` directories (added to .gitignore)
- Redundant root-level `.yaml`, `.py`, `.sh`, `.txt` files (moved to appropriate folders)

#### Migrated
- All config files (`*.yaml`) → `configs/`
- All scripts (`*.py`, `*.sh`, `*.txt`) → `scripts/`
- Source folders (`training/`, `evaluation/`, `models/`, `environment/`, `utils/`, `debugging/`) → `src/`
- Visualization files and folders → `notebooks/`
- Documentation folders (`paper_manuscript/`, `paper_notes/`, `paper_generation/`) → `docs/`
- Output folders (`results/`, `logs/`, `figures/`, `models_saved/`, `enhanced_data/`, `datasets/`) → `outputs/`

#### Updated
- `README.md` - Added repository structure section with clear navigation
- `.gitignore` - Updated patterns to reflect new structure
- All documentation now references correct paths

### Benefits
✅ **Clearer navigation** - Logical grouping makes it easy to find files  
✅ **Better maintainability** - Related code is co-located  
✅ **Improved onboarding** - New contributors can understand structure quickly  
✅ **Cleaner root** - Only essential files at top level  
✅ **Standard conventions** - Follows common Python project patterns  

---

## [1.0.0] - 2025-07-07

### Initial Release
- DARIL (Dual-task Autoregressive Imitation Learning) implementation
- Comprehensive IL vs RL comparison framework
- World model RL implementation (Dreamer-inspired)
- Direct video RL with PPO/SAC
- Inverse Reinforcement Learning variants
- Multi-horizon evaluation (1s-20s)
- CholecT50 surgical action planning pipeline
- Paper accepted at MICCAI 2025 COLAS Workshop

---

## Version Format

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes or major restructuring
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

---

**Repository:** [DARIL on GitHub](https://github.com/maxboels/DARIL-When-Imitation-Learning-outperforms-Reinforcement-Learning-in-Surgical-Action-Planning)
