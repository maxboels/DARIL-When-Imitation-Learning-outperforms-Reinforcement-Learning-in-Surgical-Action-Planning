# MICCAI 2025 Paper: Surgical Action Triplet Prediction

**Beyond Recognition: Comparing Imitation Learning and Reinforcement Learning for Surgical Action Triplet Prediction in Planning and Control**

## Paper Overview

This paper presents the first comprehensive comparison of Imitation Learning (IL) versus Reinforcement Learning (RL) approaches for surgical action triplet prediction, with emphasis on next action prediction for planning applications.

### Key Contributions

1. **Autoregressive IL Baseline**: State-of-the-art performance (0.979 mAP) on CholecT50 next action prediction
2. **Comprehensive RL Evaluation**: Multiple RL variants including world model-based RL, direct video RL, and inverse RL enhancement
3. **Novel Evaluation Framework**: Comparing recognition accuracy versus planning capability across different prediction horizons
4. **Clinical Insights**: When RL provides advantages over IL for surgical assistance applications

### Main Results

- **IL Excellence**: Superior single-step accuracy and computational efficiency for routine sequences
- **RL Advantages**: Better multi-step consistency and scenario-specific adaptation for complex situations
- **Hybrid Approach**: IRL enhancement achieves best of both paradigms
- **Clinical Impact**: IL for primary prediction with selective RL enhancement for complex scenarios

## Repository Structure

```
paper_manuscript/
├── surgical_action_prediction_miccai2025.tex    # Main paper LaTeX source
├── references.bib                               # Bibliography database
├── Makefile                                    # Build system
├── submission_checklist.md                    # Pre-submission checklist
├── README.md                                   # This file
├── figures/                                    # Generated figures
├── tables/                                     # Generated tables
├── generate_paper_figures.py                  # Figure generation script
├── create_results_table.py                    # Table generation script
└── MICCAI2025-LaTeX-Template/                 # Official template
```

## Quick Start

### Prerequisites

- LaTeX installation with LLNCS class
- Python 3.8+ with matplotlib, seaborn, pandas
- Your experimental results in `../results/`

### Building the Paper

```bash
# Generate all figures and tables
make assets

# Compile the paper
make paper

# Quick build (skip bibliography)
make quick

# Final preparation
make final
```

### Development Workflow

```bash
# Watch for changes and auto-compile
make watch

# Generate only figures
make figures

# Generate only tables  
make tables

# Check word count
make wordcount

# Validate paper structure
make validate
```

## File Descriptions

### Core Paper Files

- **`surgical_action_prediction_miccai2025.tex`**: Main LaTeX source with complete paper content
- **`references.bib`**: Bibliography with all relevant citations for surgical AI, IL/RL comparison
- **`Makefile`**: Automated build system for figures, tables, and paper compilation

### Generation Scripts

- **`generate_paper_figures.py`**: Creates publication-quality figures from experimental results
  - Recognition performance comparison
  - Planning performance across horizons
  - Scenario-specific analysis
  - Architecture comparison
  
- **`create_results_table.py`**: Generates LaTeX tables from results
  - Main performance comparison
  - Planning analysis across horizons
  - Scenario-specific performance
  - Computational efficiency analysis

### Quality Assurance

- **`submission_checklist.md`**: Comprehensive pre-submission checklist ensuring MICCAI compliance
- **`README.md`**: Complete documentation and usage guide

## Figure Generation

The paper includes four main figures generated automatically from your experimental results:

1. **Figure 1**: Recognition vs Next Action Performance Comparison
2. **Figure 2**: Planning Performance Across Prediction Horizons  
3. **Figure 3**: Scenario-Specific Performance Analysis
4. **Figure 4**: Architecture Comparison (Accuracy vs Efficiency)

Run `make figures` to generate all figures in PDF and PNG formats.

## Table Generation

Four main tables are generated automatically:

1. **Table 1**: Main Results Comparison (IL vs RL approaches)
2. **Table 2**: Planning Performance Analysis (multi-horizon)
3. **Table 3**: Scenario-Specific Comparison (routine vs complex)
4. **Table 4**: Computational Efficiency Analysis

Run `make tables` to generate all LaTeX table files.

## Experimental Data

The scripts automatically locate and process results from your latest successful experiment in:
- `../results/YYYY-MM-DD_HH-MM-SS/fold0/complete_results.json`

### Expected Result Structure

Your experiment results should contain:
- `method_1_autoregressive_il`: IL baseline results
- `method_2_conditional_world_model`: World model RL results
- `method_3_direct_video_rl`: Direct video RL results  
- `method_4_irl_enhancement`: IRL enhancement results

## Submission Preparation

### MICCAI 2025 Requirements

- **Page limit**: 8 pages main content + 2 pages references
- **Template**: LLNCS format (already configured)
- **Anonymization**: Author information anonymized
- **Format**: PDF with embedded fonts

### Final Submission Steps

1. **Complete content**: `make final`
2. **Check formatting**: Review PDF for MICCAI compliance
3. **Validate anonymization**: Ensure all author info removed
4. **Create package**: `make submission`
5. **Upload**: Submit through MICCAI system

## Quality Metrics

### Target Performance
- Recognition mAP: >0.95 (achieved: 0.979)
- Planning degradation: <20% at 5-step horizon
- RL improvement: Demonstrable in complex scenarios
- Real-time inference: <50ms per prediction

### Technical Contributions
- Novel autoregressive IL architecture for surgical prediction
- First systematic IL vs RL comparison for surgical tasks
- Multi-horizon planning evaluation framework
- Clinical deployment insights

## Troubleshooting

### Common Issues

**LaTeX compilation fails**:
```bash
make clean && make paper
```

**Figures not generating**:
```bash
# Check if results exist
ls ../results/*/fold*/complete_results.json

# Generate figures manually
python generate_paper_figures.py --results_dir ../results --output_dir figures
```

**Missing dependencies**:
```bash
pip install matplotlib seaborn pandas numpy
```

### Build System Help

```bash
make help  # Show all available commands
```

## Timeline for Submission

### 1 Week Before Deadline
- [ ] All experiments completed
- [ ] Figures and tables generated
- [ ] First complete draft ready

### 3 Days Before Deadline  
- [ ] Final results incorporated
- [ ] Paper content finalized
- [ ] Internal review completed

### Day of Submission
- [ ] Final PDF generated
- [ ] Submission package created
- [ ] Upload completed

## Key Paper Sections

### Abstract (150-250 words)
Highlights IL vs RL comparison for surgical prediction with clinical implications.

### Introduction
- Motivation for next action prediction in surgery
- IL vs RL trade-offs in sequential decision making
- Contributions and paper organization

### Methods
- Autoregressive IL baseline architecture
- World model-based RL approach
- Direct video RL implementation
- IRL enhancement methodology
- Dual evaluation framework (recognition + planning)

### Results
- Performance comparison tables
- Multi-horizon planning analysis
- Scenario-specific performance breakdown
- Computational efficiency analysis

### Discussion
- Clinical deployment considerations
- When to use IL vs RL
- Limitations and future work

### Conclusion
- Summary of key findings
- Impact on surgical AI field
- Future research directions

## Contact and Support

For questions about the paper structure or build system:
1. Check this README
2. Review the submission checklist
3. Use `make help` for build commands
4. Consult MICCAI guidelines for format requirements

## Related Work Integration

The paper builds on your extensive experimental framework while focusing on the novel IL vs RL comparison for surgical applications. All figures and tables are automatically generated from your latest experimental results, ensuring consistency between the paper and your actual findings.

## Final Notes

- Keep the submission anonymous until acceptance
- Focus on clinical relevance and practical implications
- Emphasize the novel comparison methodology
- Highlight the strong IL baseline performance (0.979 mAP)
- Position RL as scenario-specific enhancement rather than replacement