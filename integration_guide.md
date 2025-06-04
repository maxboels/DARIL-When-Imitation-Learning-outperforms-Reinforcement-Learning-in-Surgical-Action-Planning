# Complete Integration Guide: Enhanced Evaluation with Rollout Saving

## 🎯 Overview

This guide shows how to integrate the enhanced evaluation framework into your surgical RL comparison project. The enhanced evaluation provides:

- **Unified mAP metrics** across all methods (IL, RL+WorldModel, RL+OfflineVideos)
- **Detailed rollout saving** at every timestep for visualization
- **Planning horizon analysis** showing AI decision-making process
- **Statistical significance testing** between all methods
- **Interactive visualization** of model thinking process

## 📁 Files Needed

### 1. Core Integration Files
- `integrated_evaluation_framework.py` - Main evaluation framework
- `standalone_integrated_eval.py` - Standalone runner script
- `updated_visualization.html` - Interactive visualization tool

### 2. Integration Code for `run_experiment_v2.py`
- Updated `_run_comprehensive_evaluation()` method
- Updated `_generate_paper_results()` method
- Additional helper methods for result processing

## 🚀 Step-by-Step Integration

### Step 1: Add Core Files to Your Project

```bash
# In your project directory
mkdir evaluation
mv integrated_evaluation_framework.py evaluation/
mv standalone_integrated_eval.py .
mv updated_visualization.html visualization/
```

### Step 2: Update run_experiment_v2.py

Replace the existing `_run_comprehensive_evaluation` method in your `SurgicalRLComparison` class with the updated version:

```python
def _run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
    """
    UPDATED: Run integrated evaluation with rollout saving and unified mAP metrics
    """
    
    self.logger.info("📊 Running Integrated Evaluation with Rollout Saving...")
    
    try:
        # Import the integrated evaluation framework
        from evaluation.integrated_evaluation_framework import run_integrated_evaluation
        
        # Run integrated evaluation
        horizon = self.config.get('evaluation', {}).get('horizon', 15)
        integrated_results = run_integrated_evaluation(
            experiment_results=self.results,
            test_data=test_data,
            results_dir=str(self.results_dir),
            logger=self.logger,
            horizon=horizon
        )
        
        if integrated_results:
            self.logger.info("✅ Integrated evaluation completed successfully!")
            
            # Extract key results for backward compatibility
            evaluator = integrated_results['evaluator']
            results = integrated_results['results']
            file_paths = integrated_results['file_paths']
            
            # Print method comparison
            self._print_method_comparison(results['aggregate_results'])
            
            # Print statistical significance
            self._print_statistical_significance(results['statistical_tests'])
            
            return {
                'integrated_evaluation': {
                    'status': 'success',
                    'results': results,
                    'file_paths': file_paths,
                    'visualization_data_path': str(file_paths['visualization_json'])
                },
                'evaluation_type': 'integrated_with_rollouts',
                'summary': self._create_evaluation_summary(results)
            }
        else:
            return {'error': 'Integrated evaluation failed', 'status': 'failed'}
            
    except Exception as e:
        self.logger.error(f"❌ Integrated evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'status': 'failed'}
```

### Step 3: Add Helper Methods

Add these helper methods to your `SurgicalRLComparison` class:

```python
def _print_method_comparison(self, aggregate_results: Dict):
    """Print comparison of all methods"""
    
    self.logger.info("\n📊 METHOD COMPARISON (Unified mAP Metrics)")
    self.logger.info("=" * 60)
    
    # Sort methods by performance
    methods_sorted = sorted(aggregate_results.items(), 
                          key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
    
    for rank, (method, stats) in enumerate(methods_sorted, 1):
        method_display = method.replace('_', ' ')
        final_map = stats['final_mAP']['mean']
        std_map = stats['final_mAP']['std']
        degradation = stats['mAP_degradation']['mean']
        
        self.logger.info(f"{rank}. {method_display}:")
        self.logger.info(f"   📈 mAP: {final_map:.4f} ± {std_map:.4f}")
        self.logger.info(f"   📉 Degradation: {degradation:.4f}")
        self.logger.info(f"   🎯 Stability: {-degradation:.4f}")
        self.logger.info("")

def _print_statistical_significance(self, statistical_tests: Dict):
    """Print statistical significance results"""
    
    self.logger.info("🔬 STATISTICAL SIGNIFICANCE TESTS")
    self.logger.info("=" * 40)
    
    significant_comparisons = [
        (comparison, results) for comparison, results in statistical_tests.items()
        if results['significant']
    ]
    
    if significant_comparisons:
        self.logger.info(f"Found {len(significant_comparisons)} significant differences:")
        for comparison, results in significant_comparisons:
            method1, method2 = comparison.split('_vs_')
            method1_display = method1.replace('_', ' ')
            method2_display = method2.replace('_', ' ')
            
            self.logger.info(f"  • {method1_display} vs {method2_display}:")
            self.logger.info(f"    p-value: {results['p_value']:.4f}")
            self.logger.info(f"    Effect size: {results['effect_size_interpretation']}")
            self.logger.info(f"    Mean difference: {results['mean_diff']:.4f}")
    else:
        self.logger.info("No statistically significant differences found")

def _create_evaluation_summary(self, results: Dict) -> Dict:
    """Create summary for backward compatibility"""
    
    aggregate_results = results['aggregate_results']
    
    # Find best method
    best_method = max(aggregate_results.items(), 
                     key=lambda x: x[1]['final_mAP']['mean'])
    
    # Count significant comparisons
    significant_count = sum(1 for test in results['statistical_tests'].values() 
                          if test['significant'])
    
    return {
        'best_method': {
            'name': best_method[0],
            'mAP': best_method[1]['final_mAP']['mean'],
            'std': best_method[1]['final_mAP']['std']
        },
        'total_methods': len(aggregate_results),
        'significant_comparisons': significant_count,
        'evaluation_horizon': results['evaluation_config']['horizon'],
        'videos_evaluated': results['evaluation_config']['num_videos']
    }
```

### Step 4: Update Configuration

Add these settings to your `config.yaml` file:

```yaml
evaluation:
  # Enhanced evaluation settings
  horizon: 15                    # Prediction horizon for trajectory analysis
  max_videos: 10                 # Maximum test videos to evaluate
  
  # Rollout saving options
  save_detailed_rollouts: true   # Save AI thinking process
  save_planning_horizon: true    # Save future planning steps
  max_planning_steps: 5          # How many steps ahead to plan
  
  # Visualization settings
  create_visualization_data: true # Generate data for HTML tool
  save_confidence_scores: true   # Save prediction confidence
  
  # Statistical analysis
  significance_level: 0.05       # p-value threshold
  effect_size_threshold: 0.2     # Cohen's d threshold
```

## 🏃‍♂️ Running the Integrated Evaluation

### Option 1: With Your Main Experiment

Run your main experiment as usual. The integrated evaluation will automatically run at the end:

```bash
python run_experiment_v2.py
```

### Option 2: Standalone on Existing Results

If you already have experimental results, run the standalone evaluator:

```bash
# Analyze existing results first
python standalone_integrated_eval.py --analyze-only --results logs/latest/surgical_rl_results/complete_surgical_rl_results.json

# Run integrated evaluation
python standalone_integrated_eval.py --results logs/latest/surgical_rl_results/complete_surgical_rl_results.json --output integrated_evaluation_results --horizon 15 --max-videos 10
```

## 📊 Generated Output Files

The integrated evaluation will generate:

### 1. CSV Files (for data analysis)
- `integrated_video_results.csv` - Per-video performance metrics
- `integrated_aggregate_results.csv` - Summary statistics across all videos
- `trajectory_data.csv` - Performance degradation over prediction horizon

### 2. JSON Files (for programmatic access)
- `complete_integrated_results.json` - Full evaluation results
- `detailed_rollouts.json` - AI thinking process at each timestep
- `visualization_data.json` - Data for interactive visualization

### 3. Visualization Files
- Use `visualization_data.json` with the HTML visualization tool
- Contains rollout data, thinking process, and planning horizon information

## 🎨 Using the Interactive Visualization

### Step 1: Open the HTML Tool
1. Open `updated_visualization.html` in a web browser
2. Click "Choose visualization_data.json"
3. Select the `visualization_data.json` file generated by the evaluation

### Step 2: Explore the Results
- **Method Tabs**: Switch between IL, RL+WorldModel, RL+OfflineVideos
- **Video Selection**: Choose different surgical videos
- **Timestep Slider**: Move through time to see AI decisions
- **Planning Horizon**: Adjust how far ahead to show planning
- **Auto-play**: Watch AI decision-making over time

### Step 3: Analyze the Thinking Process
- **Right Panel**: Shows AI thinking steps, action candidates, and planning horizon
- **Grid Visualization**: Past actions (gray), current timestep (red), future planning (blue/green), ground truth (gold dashed)
- **Tooltips**: Hover over cells for detailed information

## 🎯 Key Features of the Integrated Evaluation

### 1. Unified mAP Metrics
- All methods evaluated on identical action prediction metrics
- No more comparing mAP vs rewards (apples vs oranges)
- Fair comparison across IL and RL approaches

### 2. Rollout Saving
- Captures AI "thinking process" at every timestep
- Shows how models plan future actions
- Enables visualization of decision-making

### 3. Statistical Analysis
- Pairwise significance testing between all methods
- Effect size calculation (Cohen's d)
- Confidence intervals and uncertainty quantification

### 4. Trajectory Analysis
- Shows how performance degrades over prediction horizon
- Identifies which methods maintain performance longer
- Useful for understanding planning capabilities

## 🔬 Research Paper Benefits

### What You Can Now Report:

1. **Fair Comparison**: "All methods evaluated using unified mAP metrics on identical test conditions"

2. **Statistical Rigor**: "Pairwise significance testing with effect size analysis shows Method X significantly outperforms Method Y (p=0.023, Cohen's d=0.78, large effect)"

3. **Planning Analysis**: "Trajectory analysis reveals Method A maintains 89% of initial performance at 15-step horizon while Method B degrades to 56%"

4. **Interpretability**: "Rollout visualization shows IL focuses on immediate action mimicry while RL demonstrates longer-term planning behavior"

5. **Reproducibility**: "Complete evaluation framework and visualization tools provided for reproducible research"

## 🚨 Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure `integrated_evaluation_framework.py` is in the `evaluation/` directory
2. **Model Loading Errors**: Check that model paths in your results JSON are correct and files exist
3. **Memory Issues**: Reduce `max_videos` or `horizon` for large evaluations
4. **Visualization Not Loading**: Ensure the JSON file is valid and browser allows local file access

### Performance Tips:

1. **Start Small**: Test with `max_videos=2` and `horizon=5` first
2. **GPU Memory**: The evaluation loads multiple models, consider using CPU for some models
3. **Parallel Processing**: For large evaluations, consider running videos in batches

## 📞 Support

If you encounter issues:

1. Check the error messages in the console output
2. Verify all model files exist and are loadable
3. Test with a smaller subset of data first
4. Check that your config file has the required evaluation settings

The integrated evaluation provides a comprehensive, fair, and visualizable comparison of your surgical RL methods. This addresses the key limitation of comparing different metrics (mAP vs rewards) and provides the unified evaluation needed for strong research contributions.
