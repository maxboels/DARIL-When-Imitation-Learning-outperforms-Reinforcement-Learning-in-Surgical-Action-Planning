
# Surgical Action Prediction: RL vs IL Evaluation Results

## Summary

**Best Performing Method**: Imitation Learning (mAP: 0.358)

## Method Performance:
- **Imitation Learning**: 0.358 mAP
- **Sac**: 0.283 mAP
- **Ppo**: 0.276 mAP

## RL vs IL Comparison:
- **SAC** underperforms IL by 0.074 mAP (-20.8%)
- **PPO** underperforms IL by 0.081 mAP (-22.8%)

## Files Generated:
- `main_results_figure.pdf` - Main publication figure
- `results_table.tex` - LaTeX table for paper
- `trajectory_results.csv` - Detailed trajectory data
- `summary_statistics.csv` - Summary statistics

## Next Steps:
1. Include the LaTeX table in your paper
2. Use the main figure as Figure 1
3. Reference the trajectory analysis in your results section
4. Extend evaluation with more videos if needed
