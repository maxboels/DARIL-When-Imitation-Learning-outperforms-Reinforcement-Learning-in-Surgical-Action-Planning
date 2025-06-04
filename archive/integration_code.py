# Integration code for run_experiment_v2.py
# Replace the existing _run_comprehensive_evaluation method with this

def _run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
    """
    UPDATED: Run integrated evaluation with rollout saving and unified mAP metrics
    """
    
    self.logger.info("📊 Running Integrated Evaluation with Rollout Saving...")
    
    try:
        # Import the integrated evaluation framework
        from integrated_evaluation_framework import run_integrated_evaluation
        
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

# Also update the _generate_paper_results method to include integrated evaluation results

def _generate_paper_results(self):
    """Generate research paper ready results with integrated evaluation"""
    
    self.logger.info("📝 Generating enhanced research paper results...")
    
    paper_results = {
        'title': 'Comprehensive Evaluation: IL vs RL+WorldModel vs RL+OfflineVideos for Surgical Action Prediction',
        'experiment_summary': {
            'method_1': 'Imitation Learning (Baseline) - Supervised learning on expert demonstrations',
            'method_2': 'RL with World Model Simulation - Uses learned dynamics for exploration',
            'method_3': 'RL with Offline Videos - Direct RL on video sequences',
            'evaluation': 'Integrated evaluation with unified mAP metrics, rollout saving, and trajectory analysis'
        },
        'key_findings': [],
        'method_performance': {},
        'integrated_evaluation_results': {},
        'research_contributions': []
    }
    
    # Extract integrated evaluation results
    integrated_eval = self.results.get('comparative_analysis', {}).get('integrated_evaluation', {})
    
    if integrated_eval.get('status') == 'success':
        eval_results = integrated_eval.get('results', {})
        
        if 'aggregate_results' in eval_results:
            aggregate_stats = eval_results['aggregate_results']
            
            # Sort methods by performance
            methods_sorted = sorted(aggregate_stats.items(), 
                                  key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
            
            paper_results['integrated_evaluation_results'] = {
                'ranking': [
                    {
                        'rank': i+1,
                        'method': method,
                        'final_mAP': stats['final_mAP']['mean'],
                        'mAP_std': stats['final_mAP']['std'],
                        'degradation': stats['mAP_degradation']['mean'],
                        'stability': stats['trajectory_stability'],
                        'confidence': stats.get('confidence', {}).get('mean', 0.0)
                    }
                    for i, (method, stats) in enumerate(methods_sorted)
                ],
                'best_method': methods_sorted[0][0] if methods_sorted else 'Unknown',
                'best_performance': methods_sorted[0][1]['final_mAP']['mean'] if methods_sorted else 0.0,
                'evaluation_features': [
                    'Unified mAP metrics across all methods',
                    'Rollout saving at every timestep',
                    'Planning horizon visualization',
                    'Thinking process capture',
                    'Statistical significance testing'
                ]
            }
            
            # Statistical significance results
            if 'statistical_tests' in eval_results:
                paper_results['integrated_evaluation_results']['statistical_tests'] = {
                    'significant_comparisons': [
                        {
                            'comparison': comp,
                            'p_value': results['p_value'],
                            'mean_difference': results['mean_diff'],
                            'effect_size': results['cohens_d'],
                            'interpretation': results['effect_size_interpretation']
                        }
                        for comp, results in eval_results['statistical_tests'].items()
                        if results['significant']
                    ],
                    'total_comparisons': len(eval_results['statistical_tests']),
                    'significant_count': sum(1 for r in eval_results['statistical_tests'].values() if r['significant'])
                }
            
            # Add visualization data path
            viz_path = integrated_eval.get('visualization_data_path')
            if viz_path:
                paper_results['integrated_evaluation_results']['visualization_data_path'] = viz_path
    
    # Traditional method results (for comparison)
    method1 = self.results.get('method_1_il_baseline', {})
    method2 = self.results.get('method_2_rl_world_model', {})
    method3 = self.results.get('method_3_rl_offline_videos', {})
    
    # Extract performance metrics
    if method1.get('status') == 'success':
        il_performance = method1.get('evaluation', {})
        paper_results['method_performance']['IL_Baseline'] = {
            'traditional_mAP': il_performance.get('mAP', 0),
            'exact_match': il_performance.get('exact_match_accuracy', 0),
            'status': 'success',
            'strength': 'Action mimicry via supervised learning'
        }
    
    if method2.get('status') == 'success':
        rl_models = method2.get('rl_models', {})
        paper_results['method_performance']['RL_WorldModel'] = {
            'algorithms': list(rl_models.keys()),
            'status': 'success',
            'strength': 'Exploration via world model simulation',
            'reward_performance': {
                alg: res.get('mean_reward', 0) 
                for alg, res in rl_models.items() 
                if res.get('status') == 'success'
            }
        }
    
    if method3.get('status') == 'success':
        rl_models = method3.get('rl_models', {})
        paper_results['method_performance']['RL_OfflineVideos'] = {
            'algorithms': list(rl_models.keys()),
            'status': 'success',
            'strength': 'Direct interaction with real video data',
            'reward_performance': {
                alg: res.get('mean_reward', 0) 
                for alg, res in rl_models.items() 
                if res.get('status') == 'success'
            }
        }
    
    # Key findings based on integrated evaluation
    findings = []
    if integrated_eval.get('status') == 'success':
        findings.append("✅ Integrated evaluation completed with unified mAP metrics")
        findings.append("✅ All methods evaluated on identical action prediction metrics")
        findings.append("✅ Detailed rollout saving enables visualization of thinking process")
        findings.append("✅ Statistical significance testing performed between all method pairs")
        findings.append("✅ Planning horizon analysis shows performance degradation patterns")
        
        # Add specific performance findings
        if 'integrated_evaluation_results' in paper_results and 'ranking' in paper_results['integrated_evaluation_results']:
            ranking = paper_results['integrated_evaluation_results']['ranking']
            if ranking:
                best = ranking[0]
                findings.append(f"✅ Best method: {best['method']} with {best['final_mAP']:.3f} mAP")
                
                if len(ranking) > 1:
                    performance_gap = best['final_mAP'] - ranking[-1]['final_mAP']
                    findings.append(f"✅ Performance gap between best and worst: {performance_gap:.3f} mAP")
    else:
        findings.append("⚠️ Integrated evaluation encountered issues")
    
    # Add method-specific findings
    if method1.get('status') == 'success':
        findings.append("✅ Method 1 (IL): Successfully trained and evaluated")
    if method2.get('status') == 'success':
        findings.append("✅ Method 2 (RL + World Model): Successfully demonstrates model-based RL")
    if method3.get('status') == 'success':
        findings.append("✅ Method 3 (RL + Offline Videos): Successfully demonstrates model-free RL")
    
    paper_results['key_findings'] = findings
    
    # Research contributions (updated)
    paper_results['research_contributions'] = [
        "First systematic three-way comparison: IL vs model-based RL vs model-free RL in surgery",
        "Integrated evaluation framework with unified mAP metrics for fair comparison",
        "Rollout saving and visualization of AI decision-making process",
        "Trajectory analysis showing performance degradation over prediction horizons",
        "Statistical significance testing with effect size analysis",
        "Comprehensive visualization suite for surgical AI method comparison",
        "Open-source implementation for reproducible surgical RL research"
    ]
    
    # Save enhanced paper results
    paper_results_path = self.results_dir / 'integrated_paper_results.json'
    with open(paper_results_path, 'w') as f:
        json.dump(paper_results, f, indent=2, default=str)
    
    # Generate enhanced paper summary
    self._generate_enhanced_paper_summary(paper_results)
    
    self.logger.info(f"📄 Integrated paper results saved to: {paper_results_path}")

def _generate_enhanced_paper_summary(self, paper_results: Dict):
    """Generate a markdown summary with integrated evaluation results"""
    
    summary_lines = []
    summary_lines.append("# Integrated Three-Way Experimental Comparison Results")
    summary_lines.append("## Surgical Action Prediction: IL vs RL Approaches with Rollout Analysis")
    summary_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Integrated evaluation results
    if 'integrated_evaluation_results' in paper_results:
        eval_results = paper_results['integrated_evaluation_results']
        
        summary_lines.append("## 🎯 Integrated Evaluation Results (Unified mAP Metrics)")
        summary_lines.append("")
        
        if 'ranking' in eval_results:
            for i, method_result in enumerate(eval_results['ranking'], 1):
                method_name = method_result['method'].replace('_', ' ')
                final_map = method_result['final_mAP']
                std_map = method_result['mAP_std']
                degradation = method_result['degradation']
                
                summary_lines.append(f"### {i}. {method_name}")
                summary_lines.append(f"- **Final mAP**: {final_map:.4f} ± {std_map:.4f}")
                summary_lines.append(f"- **mAP Degradation**: {degradation:.4f}")
                summary_lines.append(f"- **Stability Score**: {method_result['stability']:.4f}")
                summary_lines.append(f"- **Avg Confidence**: {method_result['confidence']:.4f}")
                summary_lines.append("")
        
        # Statistical significance
        if 'statistical_tests' in eval_results:
            stat_tests = eval_results['statistical_tests']
            summary_lines.append("## 🔬 Statistical Analysis")
            summary_lines.append("")
            summary_lines.append(f"- **Total Comparisons**: {stat_tests['total_comparisons']}")
            summary_lines.append(f"- **Significant Differences**: {stat_tests['significant_count']}")
            summary_lines.append("")
            
            if stat_tests['significant_comparisons']:
                summary_lines.append("### Significant Comparisons (p < 0.05)")
                for comp in stat_tests['significant_comparisons']:
                    comparison_name = comp['comparison'].replace('_vs_', ' vs ').replace('_', ' ')
                    summary_lines.append(f"- **{comparison_name}**: p={comp['p_value']:.4f}, effect size={comp['interpretation']}")
                summary_lines.append("")
        
        # Evaluation features
        if 'evaluation_features' in eval_results:
            summary_lines.append("## 🚀 Evaluation Features")
            summary_lines.append("")
            for feature in eval_results['evaluation_features']:
                summary_lines.append(f"- {feature}")
            summary_lines.append("")
    
    # Traditional method comparison
    summary_lines.append("## 📊 Traditional Method Performance")
    summary_lines.append("")
    
    for method, performance in paper_results['method_performance'].items():
        summary_lines.append(f"### {method.replace('_', ' ')}")
        if performance['status'] == 'success':
            summary_lines.append(f"- **Status**: ✅ Successful")
            summary_lines.append(f"- **Strength**: {performance['strength']}")
            if 'traditional_mAP' in performance:
                summary_lines.append(f"- **Traditional mAP**: {performance['traditional_mAP']:.4f}")
            if 'algorithms' in performance:
                summary_lines.append(f"- **Algorithms**: {', '.join(performance['algorithms'])}")
                if 'reward_performance' in performance:
                    for alg, perf in performance['reward_performance'].items():
                        summary_lines.append(f"  - **{alg.upper()}**: Mean Reward = {perf:.3f}")
        else:
            summary_lines.append(f"- **Status**: {performance['status']}")
        summary_lines.append("")
    
    summary_lines.append("## 🔍 Key Findings")
    summary_lines.append("")
    for finding in paper_results['key_findings']:
        summary_lines.append(f"- {finding}")
    summary_lines.append("")
    
    summary_lines.append("## 🏆 Research Contributions")
    summary_lines.append("")
    for contribution in paper_results['research_contributions']:
        summary_lines.append(f"- {contribution}")
    summary_lines.append("")
    
    # Add visualization note
    if 'integrated_evaluation_results' in paper_results and 'visualization_data_path' in paper_results['integrated_evaluation_results']:
        viz_path = paper_results['integrated_evaluation_results']['visualization_data_path']
        summary_lines.append("## 📊 Visualization")
        summary_lines.append("")
        summary_lines.append(f"Interactive visualization data available at: `{viz_path}`")
        summary_lines.append("Load this file in the HTML visualization tool to explore:")
        summary_lines.append("- Model thinking process at each timestep")
        summary_lines.append("- Planning horizon rollouts")
        summary_lines.append("- Ground truth vs predictions comparison")
        summary_lines.append("- Confidence and uncertainty analysis")
        summary_lines.append("")
    
    # Save summary
    summary_path = self.results_dir / 'integrated_experiment_summary.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    self.logger.info(f"📄 Integrated experiment summary saved to: {summary_path}")
