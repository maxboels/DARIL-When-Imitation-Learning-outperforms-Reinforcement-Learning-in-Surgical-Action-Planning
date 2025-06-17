# RL Debugging System - Complete Instructions

This comprehensive debugging system helps you understand why your RL training can't reach supervised learning performance levels. The system focuses on expert action matching and provides detailed analysis of training dynamics, world model quality, and action space handling.

## 🎯 System Overview

**Goal**: Understand why RL performance is poor compared to supervised learning baseline (target: reach >10% mAP vs supervised learning)

**Key Features**:
- ✅ Comprehensive RL debugging and monitoring
- ✅ Simplified expert matching environment  
- ✅ World model quality evaluation
- ✅ Action space analysis and threshold optimization
- ✅ Training visualizations and analysis
- ✅ Performance gap analysis with actionable recommendations

## 📁 File Structure

```
rl_debugging_system/
├── run_complete_debugging.py          # Main script to run everything
├── debug_experiment_runner.py         # Experiment runner with debugging
├── rl_debug_system.py                # Core debugging and visualization
├── simplified_rl_trainer.py          # Simplified RL trainer with expert focus
├── threshold_optimizer.py            # Action threshold optimization
├── debug_config_rl.yaml             # Optimized config for debugging
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Prepare Your Environment

```bash
# Make sure you have the required dependencies
pip install torch stable-baselines3 plotly matplotlib seaborn pandas scikit-learn tqdm

# Ensure your data is properly formatted
# - Frame embeddings: shape [num_frames, 1024]
# - Action binaries: shape [num_frames, 100] 
# - Expert actions: binary arrays with 1-3 positive actions per frame
```

### 2. Configure for Debugging

Create or modify `debug_config_rl.yaml`:

```yaml
# Key settings for debugging
experiment:
  train:
    max_videos: 40  # Good balance for debugging
  test:
    max_videos: 10  # Sufficient for evaluation

rl_training:
  timesteps: 30000  # Increased for better convergence
  reward_mode: 'simplified_expert_matching'  # Focus only on expert matching
  
  # Simplified reward weights
  reward_weights:
    expert_f1_score: 100.0        # PRIMARY: F1-like reward
    action_sparsity_matching: 5.0 # Match expert action density
    completion_bonus: 2.0         # Small episode completion
    
    # DISABLED complex rewards
    world_model_rewards: 0.0
    phase_completion: 0.0
    risk_penalty: 0.0

# Enhanced debugging
rl_debugging:
  enabled: true
  threshold_optimization: true
  world_model_quality_evaluation: true
  action_space_analysis: true
  compare_with_supervised: true
```

### 3. Run Complete Debugging Pipeline

```bash
# Basic run with debugging config
python run_complete_debugging.py --config debug_config_rl.yaml

# Skip supervised baseline if you already have it
python run_complete_debugging.py --config debug_config_rl.yaml --skip-supervised

# Use fewer timesteps for faster debugging
python run_complete_debugging.py --config debug_config_rl.yaml --timesteps 10000

# Verbose output for detailed debugging
python run_complete_debugging.py --config debug_config_rl.yaml --verbose
```

## 📊 What the System Does

### Step 1: Supervised Baseline
- Trains or loads supervised IL model as performance baseline
- Establishes target mAP for RL to reach
- **Expected**: >10% mAP for good baseline

### Step 2: World Model Quality Analysis
- Evaluates world model prediction accuracy
- Tests action conditioning and temporal consistency
- **Key Insight**: Poor world model quality (< 0.3) indicates RL issues

### Step 3: Simplified RL Training
- Focuses ONLY on expert action matching (F1-like rewards)
- Removes complex reward components that don't align with mAP
- **Key Feature**: Expert demonstration matching optimization

### Step 4: Action Threshold Optimization
- Tests different thresholds for converting continuous RL actions to binary
- Finds optimal threshold for maximizing mAP
- **Critical**: Often reveals if poor performance is due to thresholding

### Step 5: Comprehensive Analysis
- Compares RL vs supervised performance
- Identifies main issues and generates recommendations
- Creates detailed visualizations and reports

## 🔍 Understanding the Results

### Performance Metrics

```
📊 PERFORMANCE SUMMARY:
   Supervised Baseline: 0.1234 mAP     # Target to reach
   Best RL Performance: 0.0567 mAP     # Current RL performance  
   Performance Gap: 0.0667              # Gap to close
   RL vs Supervised Ratio: 45.9%       # Relative performance
```

### Key Indicators

| Metric | Good | Concerning | Action Needed |
|--------|------|------------|---------------|
| Supervised mAP | >10% | 5-10% | <5% |
| RL vs Supervised Ratio | >50% | 20-50% | <20% |
| World Model Quality | >0.5 | 0.3-0.5 | <0.3 |
| Threshold Improvement | >2% | 0.5-2% | <0.5% |

### Common Issues and Solutions

#### 1. "RL Not Learning" (mAP < 2%)
**Symptoms**: Very low RL performance, no learning progress
**Solutions**:
- Implement behavioral cloning warm-start
- Simplify reward function further
- Check action space conversion
- Validate expert demonstration consistency

#### 2. "Large RL vs Supervised Gap" (Gap > 5%)
**Symptoms**: RL learning but far from supervised performance
**Solutions**:
- Optimize action thresholds
- Increase training timesteps
- Adjust reward weights
- Consider hybrid IL+RL approach

#### 3. "Poor World Model Quality" (Quality < 0.3)
**Symptoms**: World model not predicting well
**Solutions**:
- Retrain world model with more data
- Use direct video RL instead
- Check world model architecture

#### 4. "Threshold Issues" (>2% improvement with optimal threshold)
**Symptoms**: Performance significantly improves with different threshold
**Solutions**:
- Use optimal threshold instead of 0.5
- Retrain with threshold-aware rewards
- Modify action space representation

## 📈 Interpreting Visualizations

### 1. Training Progress Plots
- **Reward Progression**: Should show upward trend
- **Expert Matching**: Should increase over time (target >50%)
- **Action Density**: Should match expert patterns (1-3 actions/frame)
- **mAP During Training**: Should gradually improve

### 2. Threshold Analysis Plots
- **Threshold vs mAP**: Shows optimal threshold
- **Action Density Comparison**: RL vs expert action patterns
- **Prediction Distribution**: Shows RL output distribution
- **Per-Action Performance**: Identifies problematic actions

### 3. World Model Quality Plots
- **State Prediction Accuracy**: MSE between predicted and actual states
- **Action Conditioning**: How much actions affect predictions
- **Temporal Consistency**: Smoothness of predictions over time

## 💡 Debugging Workflow

### Phase 1: Initial Assessment
1. Run complete debugging pipeline
2. Check if supervised baseline is reasonable (>5% mAP)
3. Identify main performance gaps

### Phase 2: Issue Identification  
1. Analyze world model quality if using world model RL
2. Check action threshold optimization results
3. Review training dynamics and convergence

### Phase 3: Targeted Fixes
1. Apply top recommendations from analysis
2. Re-run with optimized settings
3. Monitor improvement in key metrics

### Phase 4: Iterative Improvement
1. Continue optimization based on results
2. Consider ensemble or hybrid approaches
3. Scale up successful configurations

## 🛠️ Advanced Usage

### Custom Reward Functions
```python
# Modify simplified_rl_trainer.py for custom rewards
def _calculate_simplified_reward(self, action, predicted_rewards):
    reward = 0.0
    
    # Custom expert matching logic
    if self.current_frame_idx < len(self.current_expert_actions):
        expert_actions = self.current_expert_actions[self.current_frame_idx]
        # Add your custom reward calculation here
        
    return reward
```

### Custom Threshold Analysis
```python
# Use threshold_optimizer.py independently
from threshold_optimizer import ActionThresholdOptimizer

optimizer = ActionThresholdOptimizer(logger, "custom_analysis")
results = optimizer.analyze_rl_model_comprehensive(
    rl_model, test_data, "MyModel"
)
```

### Integration with Existing Code
```python
# Add debugging to existing RL training
from rl_debug_system import RLDebugger

debugger = RLDebugger(save_dir="debug", logger=logger, config=config)
world_model_analysis = debugger.evaluate_world_model_quality(world_model, test_data)
```

## 📋 Troubleshooting

### Common Errors

**"No predictions collected"**
- Check that RL model is properly loaded
- Verify test data format and shape
- Ensure action prediction works

**"World model evaluation failed"**  
- Check world model device compatibility
- Verify world model can perform simulation
- Ensure proper state/action tensor shapes

**"Threshold optimization crashed"**
- Check that expert actions are binary (0/1)
- Verify prediction shapes match expert actions
- Ensure sufficient data for analysis

### Performance Issues

**Slow training/evaluation**
- Reduce number of test videos for debugging
- Use CPU for small models to avoid GPU memory issues
- Sample fewer frames per video for analysis

**Memory issues**
- Reduce batch sizes in config
- Use fewer videos for initial debugging
- Enable gradient checkpointing if available

## 📞 Support

If you encounter issues:

1. **Check logs**: Detailed logs are saved in the results directory
2. **Review config**: Ensure all paths and parameters are correct  
3. **Verify data**: Check that data shapes and formats are as expected
4. **Start simple**: Use fewer videos and timesteps for initial debugging

## 🎯 Expected Outcomes

After running the complete debugging pipeline, you should have:

✅ **Clear performance baseline** from supervised learning
✅ **Understanding of RL learning dynamics** through visualizations  
✅ **Identification of main performance bottlenecks**
✅ **Optimal action threshold** for RL predictions
✅ **Actionable recommendations** for improvement
✅ **Comprehensive analysis reports** for further investigation

The goal is to either:
1. **Fix RL training** to reach competitive performance with supervised learning
2. **Understand fundamental limitations** of RL for your specific task
3. **Identify optimal hybrid approaches** combining IL and RL

This system provides the tools and insights needed to systematically debug and improve RL performance in sparse action prediction tasks like surgical action recognition.
