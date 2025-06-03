import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from collections import defaultdict, Counter

class ModelAnalyzer:
    """
    Utility class for analyzing the DualWorldModel and its predictions.
    """
    
    def __init__(self, model, device='cuda', logger=None):
        """
        Initialize the analyzer.
        
        Args:
            model: DualWorldModel instance
            device: Device to run analysis on
            logger: Optional logger
        """
        self.model = model
        self.device = device
        self.logger = logger or self._create_dummy_logger()
    
    def _create_dummy_logger(self):
        """Create a dummy logger if none provided."""
        class DummyLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
        return DummyLogger()
    
    def analyze_model_architecture(self):
        """Analyze and report model architecture details."""
        self.logger.info("=== MODEL ARCHITECTURE ANALYSIS ===")
        
        # Count parameters by component
        param_counts = {}
        total_params = 0
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    param_counts[name] = module_params
                    total_params += module_params
        
        # Group by major components
        component_counts = defaultdict(int)
        for name, count in param_counts.items():
            if 'gpt2' in name:
                component_counts['GPT-2 Backbone'] += count
            elif 'heads' in name:
                component_counts['Prediction Heads'] += count
            elif 'embedding' in name or 'projection' in name:
                component_counts['Embedding Layers'] += count
            else:
                component_counts['Other'] += count
        
        # Report
        self.logger.info(f"Total Parameters: {total_params:,}")
        for component, count in component_counts.items():
            percentage = (count / total_params) * 100
            self.logger.info(f"{component}: {count:,} ({percentage:.1f}%)")
        
        # Analyze model capabilities
        capabilities = []
        if self.model.autoregressive_action_prediction:
            capabilities.append("Autoregressive Action Prediction")
        if self.model.rl_state_prediction:
            capabilities.append("RL State Prediction")
        if self.model.reward_prediction:
            capabilities.append("Reward Prediction")
        
        self.logger.info(f"Model Capabilities: {', '.join(capabilities)}")
        
        return {
            'total_params': total_params,
            'component_counts': dict(component_counts),
            'capabilities': capabilities
        }
    
    def analyze_attention_patterns(self, sample_data, max_samples=5):
        """
        Analyze attention patterns in the GPT-2 backbone.
        
        Args:
            sample_data: Sample input data
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary of attention analysis results
        """
        self.logger.info("Analyzing attention patterns...")
        
        self.model.eval()
        attention_results = []
        
        with torch.no_grad():
            for i, batch in enumerate(sample_data):
                if i >= max_samples:
                    break
                
                current_states = batch['current_states'][:1].to(self.device)  # Single sample
                
                # Forward pass with attention output
                outputs = self.model(
                    current_states=current_states,
                    mode='supervised',
                    return_hidden_states=True
                )
                
                # Extract attention weights (if available)
                if 'all_hidden_states' in outputs:
                    hidden_states = outputs['all_hidden_states']
                    
                    # Analyze attention diversity
                    attention_analysis = self._analyze_attention_diversity(hidden_states)
                    attention_results.append(attention_analysis)
        
        if attention_results:
            # Aggregate results
            avg_attention_entropy = np.mean([r['attention_entropy'] for r in attention_results])
            avg_attention_spread = np.mean([r['attention_spread'] for r in attention_results])
            
            return {
                'average_attention_entropy': avg_attention_entropy,
                'average_attention_spread': avg_attention_spread,
                'num_samples_analyzed': len(attention_results)
            }
        
        return {'status': 'no_attention_data_available'}
    
    def _analyze_attention_diversity(self, hidden_states):
        """Analyze diversity in attention patterns."""
        # Simple attention diversity analysis
        # This is a placeholder - actual implementation would need access to attention weights
        
        layer_entropies = []
        layer_spreads = []
        
        for layer_hidden in hidden_states:
            # Compute simple entropy-like measure based on hidden state variance
            hidden_var = torch.var(layer_hidden, dim=-1)
            entropy_proxy = -torch.sum(hidden_var * torch.log(hidden_var + 1e-8))
            layer_entropies.append(entropy_proxy.item())
            
            # Compute spread
            spread = torch.std(layer_hidden).item()
            layer_spreads.append(spread)
        
        return {
            'attention_entropy': np.mean(layer_entropies),
            'attention_spread': np.mean(layer_spreads)
        }
    
    def analyze_prediction_patterns(self, test_data, num_samples=100):
        """
        Analyze patterns in model predictions.
        
        Args:
            test_data: Test data loader
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary of prediction analysis results
        """
        self.logger.info("Analyzing prediction patterns...")
        
        self.model.eval()
        
        # Collect predictions
        state_predictions = []
        action_predictions = []
        actual_states = []
        actual_actions = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in test_data:
                if sample_count >= num_samples:
                    break
                
                current_states = batch['current_states'].to(self.device)
                next_states = batch['next_states'].to(self.device)
                next_actions = batch['next_actions'].to(self.device)
                
                # Get predictions
                outputs = self.model(
                    current_states=current_states,
                    mode='supervised'
                )
                
                # Store predictions and actuals
                if 'state_pred' in outputs:
                    state_predictions.append(outputs['state_pred'].cpu().numpy())
                    actual_states.append(next_states.cpu().numpy())
                
                if 'action_pred' in outputs:
                    action_probs = torch.sigmoid(outputs['action_pred'])
                    action_predictions.append(action_probs.cpu().numpy())
                    actual_actions.append(next_actions.cpu().numpy())
                
                sample_count += current_states.size(0)
        
        results = {}
        
        # Analyze state predictions
        if state_predictions:
            state_pred_array = np.concatenate(state_predictions, axis=0)
            actual_state_array = np.concatenate(actual_states, axis=0)
            
            results['state_analysis'] = self._analyze_state_predictions(
                state_pred_array, actual_state_array
            )
        
        # Analyze action predictions
        if action_predictions:
            action_pred_array = np.concatenate(action_predictions, axis=0)
            actual_action_array = np.concatenate(actual_actions, axis=0)
            
            results['action_analysis'] = self._analyze_action_predictions(
                action_pred_array, actual_action_array
            )
        
        return results
    
    def _analyze_state_predictions(self, predictions, actuals):
        """Analyze state prediction patterns."""
        # Flatten for analysis
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        actual_flat = actuals.reshape(-1, actuals.shape[-1])
        
        # Basic statistics
        pred_mean = np.mean(pred_flat, axis=0)
        actual_mean = np.mean(actual_flat, axis=0)
        
        # Correlation analysis
        correlations = []
        for i in range(pred_flat.shape[1]):
            corr = np.corrcoef(pred_flat[:, i], actual_flat[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        # PCA analysis
        pca = PCA(n_components=min(10, pred_flat.shape[1]))
        pred_pca = pca.fit_transform(pred_flat)
        actual_pca = pca.transform(actual_flat)
        
        return {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'prediction_mean_norm': np.linalg.norm(pred_mean),
            'actual_mean_norm': np.linalg.norm(actual_mean),
            'pca_explained_variance': pca.explained_variance_ratio_[:5].tolist(),
            'dimensionality_reduction_quality': np.mean(correlations) if correlations else 0
        }
    
    def _analyze_action_predictions(self, predictions, actuals):
        """Analyze action prediction patterns."""
        # Action class frequency analysis
        pred_binary = (predictions > 0.5).astype(int)
        
        # Class distribution
        pred_class_dist = np.mean(pred_binary, axis=0)
        actual_class_dist = np.mean(actuals, axis=0)
        
        # Class-wise accuracy
        class_accuracies = []
        for i in range(predictions.shape[-1]):
            acc = np.mean(pred_binary[:, :, i] == actuals[:, :, i])
            class_accuracies.append(acc)
        
        # Prediction confidence analysis
        confidence_scores = np.max(predictions, axis=-1)  # Max probability per prediction
        avg_confidence = np.mean(confidence_scores)
        
        return {
            'class_distribution_similarity': np.corrcoef(pred_class_dist, actual_class_dist)[0, 1],
            'mean_class_accuracy': np.mean(class_accuracies),
            'std_class_accuracy': np.std(class_accuracies),
            'average_prediction_confidence': avg_confidence,
            'active_classes_predicted': np.sum(pred_class_dist > 0.1),
            'active_classes_actual': np.sum(actual_class_dist > 0.1)
        }
    
    def visualize_embedding_space(self, test_data, save_path=None, max_samples=500):
        """
        Visualize the learned embedding space using t-SNE.
        
        Args:
            test_data: Test data loader
            save_path: Path to save visualization
            max_samples: Maximum samples for visualization
            
        Returns:
            Dictionary with visualization results
        """
        self.logger.info("Creating embedding space visualization...")
        
        self.model.eval()
        
        # Collect embeddings and metadata
        embeddings = []
        phases = []
        video_ids = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in test_data:
                if sample_count >= max_samples:
                    break
                
                current_states = batch['current_states'].to(self.device)
                next_phases = batch.get('next_phases', None)
                
                # Get hidden representations
                outputs = self.model(
                    current_states=current_states,
                    mode='supervised',
                    return_hidden_states=True
                )
                
                if 'hidden_states' in outputs:
                    # Use the last timestep hidden state
                    hidden = outputs['hidden_states'][:, -1, :].cpu().numpy()
                    embeddings.append(hidden)
                    
                    # Collect metadata
                    if next_phases is not None:
                        phase_labels = torch.argmax(next_phases, dim=-1)[:, -1].cpu().numpy()
                        phases.extend(phase_labels)
                    else:
                        phases.extend([0] * hidden.shape[0])  # Default phase
                    
                    # Video IDs (if available in batch)
                    if 'video_id' in batch:
                        video_ids.extend(batch['video_id'])
                    else:
                        video_ids.extend(['unknown'] * hidden.shape[0])
                
                sample_count += current_states.size(0)
        
        if not embeddings:
            return {'status': 'no_embeddings_collected'}
        
        # Concatenate embeddings
        embeddings_array = np.concatenate(embeddings, axis=0)
        
        # Apply t-SNE
        self.logger.info("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)//4))
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Color by phase
        unique_phases = list(set(phases))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_phases)))
        
        for i, phase in enumerate(unique_phases):
            mask = np.array(phases) == phase
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f'Phase {phase}', alpha=0.6)
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Embedding Space Visualization (colored by surgical phase)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Embedding visualization saved to {save_path}")
        
        plt.show()
        
        # Analysis of embedding space
        results = {
            'num_samples': len(embeddings_array),
            'embedding_dim': embeddings_array.shape[1],
            'num_phases': len(unique_phases),
            'tsne_coordinates': embeddings_2d.tolist(),
            'phase_labels': phases
        }
        
        # Compute clustering metrics
        if len(unique_phases) > 1:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(embeddings_2d, phases)
            results['silhouette_score'] = silhouette
            
            self.logger.info(f"Embedding space silhouette score: {silhouette:.3f}")
        
        return results

class DataAnalyzer:
    """
    Utility class for analyzing the surgical video dataset.
    """
    
    def __init__(self, logger=None):
        """Initialize the data analyzer."""
        self.logger = logger or self._create_dummy_logger()
    
    def _create_dummy_logger(self):
        """Create a dummy logger if none provided."""
        class DummyLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
        return DummyLogger()
    
    def analyze_dataset_statistics(self, data_list):
        """
        Analyze basic statistics of the dataset.
        
        Args:
            data_list: List of video data dictionaries
            
        Returns:
            Dictionary of dataset statistics
        """
        self.logger.info("Analyzing dataset statistics...")
        
        stats = {
            'num_videos': len(data_list),
            'total_frames': 0,
            'video_lengths': [],
            'action_statistics': defaultdict(int),
            'phase_statistics': defaultdict(int),
            'instrument_statistics': defaultdict(int)
        }
        
        for video in data_list:
            num_frames = video['num_frames']
            stats['total_frames'] += num_frames
            stats['video_lengths'].append(num_frames)
            
            # Analyze actions
            actions = video['actions_binaries']
            for frame_actions in actions:
                active_actions = np.where(frame_actions > 0)[0]
                for action_idx in active_actions:
                    stats['action_statistics'][action_idx] += 1
            
            # Analyze phases
            phases = video['phase_binaries']
            for frame_phases in phases:
                active_phase = np.argmax(frame_phases)
                stats['phase_statistics'][active_phase] += 1
            
            # Analyze instruments
            instruments = video['instruments_binaries']
            for frame_instruments in instruments:
                active_instruments = np.where(frame_instruments > 0)[0]
                for inst_idx in active_instruments:
                    stats['instrument_statistics'][inst_idx] += 1
        
        # Compute summary statistics
        stats['avg_video_length'] = np.mean(stats['video_lengths'])
        stats['std_video_length'] = np.std(stats['video_lengths'])
        stats['min_video_length'] = np.min(stats['video_lengths'])
        stats['max_video_length'] = np.max(stats['video_lengths'])
        
        # Most common actions/phases/instruments
        stats['most_common_actions'] = sorted(
            stats['action_statistics'].items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        stats['most_common_phases'] = sorted(
            stats['phase_statistics'].items(), key=lambda x: x[1], reverse=True
        )
        
        stats['most_common_instruments'] = sorted(
            stats['instrument_statistics'].items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Log key statistics
        self.logger.info(f"Dataset: {stats['num_videos']} videos, {stats['total_frames']} total frames")
        self.logger.info(f"Average video length: {stats['avg_video_length']:.1f} ± {stats['std_video_length']:.1f} frames")
        self.logger.info(f"Most common actions: {[f'Action {a}' for a, _ in stats['most_common_actions'][:3]]}")
        self.logger.info(f"Phase distribution: {dict(stats['most_common_phases'])}")
        
        return stats
    
    def analyze_temporal_patterns(self, data_list):
        """
        Analyze temporal patterns in the surgical videos.
        
        Args:
            data_list: List of video data dictionaries
            
        Returns:
            Dictionary of temporal analysis results
        """
        self.logger.info("Analyzing temporal patterns...")
        
        # Phase transition analysis
        phase_transitions = []
        action_sequences = []
        
        for video in data_list:
            phases = video['phase_binaries']
            actions = video['actions_binaries']
            
            # Track phase transitions
            current_phase = -1
            transitions = []
            
            for i, frame_phases in enumerate(phases):
                frame_phase = np.argmax(frame_phases)
                if frame_phase != current_phase:
                    if current_phase != -1:
                        transitions.append((current_phase, frame_phase, i))
                    current_phase = frame_phase
            
            phase_transitions.extend(transitions)
            
            # Track action sequences (simplified)
            action_sequence = []
            for frame_actions in actions:
                active_actions = tuple(np.where(frame_actions > 0)[0])
                action_sequence.append(active_actions)
            
            action_sequences.append(action_sequence)
        
        # Analyze phase transitions
        transition_counts = Counter([(t[0], t[1]) for t in phase_transitions])
        
        # Analyze action pattern diversity
        all_action_patterns = []
        for seq in action_sequences:
            all_action_patterns.extend(seq)
        
        pattern_counts = Counter(all_action_patterns)
        
        results = {
            'total_phase_transitions': len(phase_transitions),
            'most_common_transitions': list(transition_counts.most_common(10)),
            'unique_action_patterns': len(pattern_counts),
            'most_common_action_patterns': list(pattern_counts.most_common(10)),
            'avg_transitions_per_video': len(phase_transitions) / len(data_list)
        }
        
        self.logger.info(f"Phase transitions: {results['total_phase_transitions']} total")
        self.logger.info(f"Unique action patterns: {results['unique_action_patterns']}")
        
        return results
    
    def create_dataset_visualization(self, data_list, save_path=None):
        """
        Create comprehensive dataset visualizations.
        
        Args:
            data_list: List of video data dictionaries  
            save_path: Directory to save visualizations
        """
        self.logger.info("Creating dataset visualizations...")
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        # Video length distribution
        video_lengths = [video['num_frames'] for video in data_list]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(video_lengths, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Video Length (frames)')
        plt.ylabel('Number of Videos')
        plt.title('Video Length Distribution')
        plt.grid(True, alpha=0.3)
        
        # Phase distribution
        all_phases = []
        for video in data_list:
            phases = video['phase_binaries']
            for frame_phases in phases:
                phase = np.argmax(frame_phases)
                all_phases.append(phase)
        
        phase_counts = Counter(all_phases)
        
        plt.subplot(1, 3, 2)
        phases = list(phase_counts.keys())
        counts = list(phase_counts.values())
        plt.bar(phases, counts, alpha=0.7)
        plt.xlabel('Surgical Phase')
        plt.ylabel('Number of Frames')
        plt.title('Phase Distribution')
        plt.grid(True, alpha=0.3)
        
        # Action frequency (top 10)
        all_actions = defaultdict(int)
        for video in data_list:
            actions = video['actions_binaries']
            for frame_actions in actions:
                active_actions = np.where(frame_actions > 0)[0]
                for action_idx in active_actions:
                    all_actions[action_idx] += 1
        
        top_actions = sorted(all_actions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        plt.subplot(1, 3, 3)
        action_ids = [f'A{a}' for a, _ in top_actions]
        action_counts = [c for _, c in top_actions]
        plt.bar(action_ids, action_counts, alpha=0.7)
        plt.xlabel('Action ID')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Frequent Actions')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
            self.logger.info(f"Dataset visualization saved to {save_path}")
        
        plt.show()

def create_comprehensive_analysis_report(model, test_data, train_data, save_dir="analysis_report"):
    """
    Create a comprehensive analysis report of the model and dataset.
    
    Args:
        model: Trained DualWorldModel
        test_data: Test data loader
        train_data: Training data list
        save_dir: Directory to save the report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize analyzers
    model_analyzer = ModelAnalyzer(model)
    data_analyzer = DataAnalyzer()
    
    print("Creating comprehensive analysis report...")
    
    # 1. Model Architecture Analysis
    print("Analyzing model architecture...")
    arch_analysis = model_analyzer.analyze_model_architecture()
    
    # 2. Dataset Analysis
    print("Analyzing dataset...")
    dataset_stats = data_analyzer.analyze_dataset_statistics(train_data)
    temporal_patterns = data_analyzer.analyze_temporal_patterns(train_data)
    
    # 3. Prediction Pattern Analysis
    print("Analyzing prediction patterns...")
    prediction_analysis = model_analyzer.analyze_prediction_patterns(test_data)
    
    # 4. Embedding Space Visualization
    print("Creating embedding visualization...")
    embedding_viz = model_analyzer.visualize_embedding_space(
        test_data, 
        save_path=os.path.join(save_dir, 'embedding_space.png')
    )
    
    # 5. Dataset Visualizations
    print("Creating dataset visualizations...")
    data_analyzer.create_dataset_visualization(train_data, save_dir)
    
    # 6. Compile Report
    report = {
        'model_architecture': arch_analysis,
        'dataset_statistics': dataset_stats,
        'temporal_patterns': temporal_patterns,
        'prediction_analysis': prediction_analysis,
        'embedding_analysis': embedding_viz,
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    # Save report
    report_path = os.path.join(save_dir, 'comprehensive_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Comprehensive analysis report saved to {save_dir}")
    return report