#!/usr/bin/env python3
"""
Prior Data Extraction for Safety Guardrails Negative Generation
Analyzes training data to inform targeted negative generation for high-opportunity components

Run this BEFORE training to extract:
1. Action-phase co-occurrence statistics (for validation)
2. Component difficulty analysis (T: 52%, IVT: 33% focus)
3. Anatomical safety patterns
4. Combination appropriateness rules
5. Workflow safety constraints
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from datetime import datetime

# Import your dataset loader
from datasets.cholect50 import load_cholect50_data

class SurgicalPriorExtractor:
    """
    Extract prior knowledge from training data to inform safety guardrails negative generation
    """
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        
        # Load label configuration
        with open('data/labels.json', 'r') as f:
            self.labels_config = json.load(f)
        
        self.actions = self.labels_config['action']
        self.phases = self.labels_config['phase']
        self.instruments = self.labels_config['instrument'] 
        self.verbs = self.labels_config['verb']
        self.targets = self.labels_config['target']
        
        # Parse action structure
        self._parse_action_structure()
        
        # Initialize storage for extracted priors
        self.priors = {
            'action_phase_cooccurrence': defaultdict(lambda: defaultdict(int)),
            'phase_frame_counts': defaultdict(int),
            'component_difficulty': {},
            'anatomical_relationships': {},
            'combination_patterns': {},
            'workflow_constraints': {},
            'safety_violations': []
        }
        
        self.logger.info("🔬 Surgical Prior Extractor initialized")
        self.logger.info(f"   Actions: {len(self.actions)}")
        self.logger.info(f"   Phases: {len(self.phases)}")
        self.logger.info(f"   Focus: High-opportunity components (T: 52%, IVT: 33%)")
    
    def _parse_action_structure(self):
        """Parse actions into instrument-verb-target structure"""
        
        self.action_triplets = {}
        self.actions_by_component = {
            'instrument': defaultdict(list),
            'verb': defaultdict(list), 
            'target': defaultdict(list)
        }
        
        for action_id, action_str in self.actions.items():
            action_id = int(action_id)
            
            if 'null_verb' in action_str:
                # Handle null actions
                parts = action_str.split(',')
                instrument = parts[0]
                self.action_triplets[action_id] = {
                    'instrument': instrument,
                    'verb': 'null',
                    'target': 'null',
                    'is_null': True
                }
            else:
                # Normal triplets
                parts = action_str.split(',')
                if len(parts) == 3:
                    instrument, verb, target = parts
                    self.action_triplets[action_id] = {
                        'instrument': instrument,
                        'verb': verb,
                        'target': target,
                        'is_null': False
                    }
                    
                    # Group by components
                    self.actions_by_component['instrument'][instrument].append(action_id)
                    self.actions_by_component['verb'][verb].append(action_id)
                    self.actions_by_component['target'][target].append(action_id)
    
    def extract_all_priors(self, train_data: List[Dict]) -> Dict[str, Any]:
        """
        Extract all prior knowledge from training data
        
        Args:
            train_data: Training videos from load_cholect50_data
            
        Returns:
            Dictionary containing all extracted priors
        """
        
        self.logger.info("🔬 EXTRACTING SURGICAL PRIORS FROM TRAINING DATA")
        self.logger.info("=" * 60)
        
        # 1. Action-Phase Co-occurrence Analysis
        self.logger.info("📊 Analyzing action-phase co-occurrence patterns...")
        self._extract_action_phase_cooccurrence(train_data)
        
        # 2. Component Difficulty Analysis 
        self.logger.info("📈 Analyzing component difficulty patterns...")
        self._extract_component_difficulty(train_data)
        
        # 3. Anatomical Safety Analysis
        self.logger.info("🩺 Extracting anatomical safety patterns...")
        self._extract_anatomical_patterns(train_data)
        
        # 4. Combination Safety Analysis
        self.logger.info("⚕️ Analyzing surgical combination patterns...")
        self._extract_combination_patterns(train_data)
        
        # 5. Workflow Constraint Analysis
        self.logger.info("🕒 Extracting workflow constraint patterns...")
        self._extract_workflow_constraints(train_data)
        
        # 6. Safety Violation Detection
        self.logger.info("🚨 Detecting potential safety violation patterns...")
        self._detect_safety_violations(train_data)
        
        # Generate summary
        summary = self._generate_extraction_summary()
        self.priors['extraction_summary'] = summary
        
        self.logger.info("✅ Prior extraction completed")
        self.logger.info(f"📋 Summary: {summary['total_patterns']} patterns extracted")
        
        return self.priors
    
    def _extract_action_phase_cooccurrence(self, train_data: List[Dict]):
        """Extract action-phase co-occurrence for negative validation"""
        
        for video in train_data:
            video_id = video['video_id']
            actions_binaries = video['actions_binaries']
            phase_binaries = video['phase_binaries']

            self.logger.info(f"   Processing video: {video_id} with {len(actions_binaries)} frames")
            
            for frame_idx in range(len(actions_binaries)):
                # Get current phase
                phase_id = np.argmax(phase_binaries[frame_idx])
                phase_name = self.phases.get(str(phase_id), 'unknown')
                
                # Count total frames per phase
                self.priors['phase_frame_counts'][phase_name] += 1
                
                # Get active actions
                active_actions = np.where(actions_binaries[frame_idx] > 0.5)[0]
                
                # Count co-occurrences
                for action_id in active_actions:
                    action_str = self.actions.get(str(action_id), f'action_{action_id}')
                    self.priors['action_phase_cooccurrence'][phase_name][action_str] += 1
        
        self.logger.info(f"   ✅ Processed {sum(self.priors['phase_frame_counts'].values())} frames")
        self.logger.info(f"   ✅ Found {len(self.priors['action_phase_cooccurrence'])} phase patterns")
    
    def _extract_component_difficulty(self, train_data: List[Dict]):
        """Analyze component difficulty to focus on high-opportunity areas"""
        
        component_stats = {
            'target_frequency': defaultdict(int),
            'target_diversity': defaultdict(set),
            'combination_frequency': defaultdict(int),
            'rare_targets': [],
            'rare_combinations': []
        }
        
        total_frames = 0
        
        for video in train_data:
            actions_binaries = video['actions_binaries']
            total_frames += len(actions_binaries)

            self.logger.info(f"   Processing video: {video['video_id']} with {len(actions_binaries)} frames")
            
            for frame_idx in range(len(actions_binaries)):
                active_actions = np.where(actions_binaries[frame_idx] > 0.5)[0]
                
                for action_id in active_actions:
                    if action_id in self.action_triplets:
                        triplet = self.action_triplets[action_id]
                        if not triplet['is_null']:
                            target = triplet['target']
                            combination = f"{triplet['instrument']},{triplet['verb']},{triplet['target']}"
                            
                            # Count target frequency
                            component_stats['target_frequency'][target] += 1
                            
                            # Track target diversity (which instruments/verbs used with each target)
                            component_stats['target_diversity'][target].add(f"{triplet['instrument']},{triplet['verb']}")
                            
                            # Count combination frequency
                            component_stats['combination_frequency'][combination] += 1
        
        # Identify rare targets (potential high-difficulty)
        target_threshold = total_frames * 0.01  # Targets appearing in <1% of frames
        for target, count in component_stats['target_frequency'].items():
            if count < target_threshold:
                component_stats['rare_targets'].append((target, count, count/total_frames))
        
        # Identify rare combinations (potential high-difficulty)
        combination_threshold = total_frames * 0.005  # Combinations appearing in <0.5% of frames
        for combination, count in component_stats['combination_frequency'].items():
            if count < combination_threshold:
                component_stats['rare_combinations'].append((combination, count, count/total_frames))
        
        # Sort by rarity
        component_stats['rare_targets'].sort(key=lambda x: x[1])
        component_stats['rare_combinations'].sort(key=lambda x: x[1])
        
        self.priors['component_difficulty'] = component_stats
        
        self.logger.info(f"   ✅ Found {len(component_stats['rare_targets'])} rare targets")
        self.logger.info(f"   ✅ Found {len(component_stats['rare_combinations'])} rare combinations")
    
    def _extract_anatomical_patterns(self, train_data: List[Dict]):
        """Extract anatomical safety patterns for Target component improvement"""
        
        anatomical_patterns = {
            'target_cooccurrence': defaultdict(lambda: defaultdict(int)),
            'dangerous_proximity': [],
            'safe_alternatives': {},
            'specificity_preferences': {}
        }
        
        # Define anatomical groupings for safety analysis
        anatomical_groups = {
            'vessels': ['cystic_artery', 'blood_vessel'],
            'ducts': ['cystic_duct'],
            'organs': ['gallbladder', 'liver'],
            'tissues': ['cystic_plate', 'cystic_pedicle', 'omentum', 'peritoneum'],
            'spaces': ['abdominal_wall_cavity'],
            'other': ['gut', 'specimen_bag', 'fluid', 'adhesion']
        }
        
        # Analyze target co-occurrence in same frames
        for video in train_data:
            actions_binaries = video['actions_binaries']

            self.logger.info(f"   Processing video: {video['video_id']} with {len(actions_binaries)} frames")
            
            for frame_idx in range(len(actions_binaries)):
                active_actions = np.where(actions_binaries[frame_idx] > 0.5)[0]
                frame_targets = []
                
                for action_id in active_actions:
                    if action_id in self.action_triplets and not self.action_triplets[action_id]['is_null']:
                        target = self.action_triplets[action_id]['target']
                        frame_targets.append(target)
                
                # Count co-occurrences
                for i, target1 in enumerate(frame_targets):
                    for j, target2 in enumerate(frame_targets):
                        if i != j:
                            anatomical_patterns['target_cooccurrence'][target1][target2] += 1
        
        # Identify dangerous proximity patterns
        dangerous_pairs = [
            ('cystic_artery', 'blood_vessel'),  # Specific vs generic vessel
            ('gallbladder', 'liver'),           # Adjacent organs
            ('cystic_duct', 'blood_vessel'),    # Structure confusion
        ]
        
        for target1, target2 in dangerous_pairs:
            cooccur_count = anatomical_patterns['target_cooccurrence'][target1].get(target2, 0)
            if cooccur_count > 0:
                anatomical_patterns['dangerous_proximity'].append({
                    'target_pair': (target1, target2),
                    'cooccurrence_count': cooccur_count,
                    'risk_level': 'high' if 'blood_vessel' in (target1, target2) else 'medium'
                })
        
        # Analyze specificity preferences (specific vs generic)
        specificity_pairs = {
            'cystic_artery': 'blood_vessel',
            'cystic_duct': 'blood_vessel',
        }
        
        for specific, generic in specificity_pairs.items():
            specific_count = sum(anatomical_patterns['target_cooccurrence'][specific].values())
            generic_count = sum(anatomical_patterns['target_cooccurrence'][generic].values())
            
            if specific_count > 0 or generic_count > 0:
                anatomical_patterns['specificity_preferences'][specific] = {
                    'specific_frequency': specific_count,
                    'generic_frequency': generic_count,
                    'specificity_ratio': specific_count / (specific_count + generic_count) if (specific_count + generic_count) > 0 else 0
                }
        
        self.priors['anatomical_relationships'] = anatomical_patterns
        
        self.logger.info(f"   ✅ Found {len(anatomical_patterns['dangerous_proximity'])} dangerous proximity patterns")
        self.logger.info(f"   ✅ Analyzed {len(anatomical_patterns['specificity_preferences'])} specificity preferences")
    
    def _extract_combination_patterns(self, train_data: List[Dict]):
        """Extract combination safety patterns for IVT component improvement"""
        
        combination_patterns = {
            'instrument_target_safety': defaultdict(lambda: defaultdict(int)),
            'verb_target_safety': defaultdict(lambda: defaultdict(int)),
            'dangerous_combinations': [],
            'safe_combinations': [],
            'technique_preferences': {}
        }
        
        # Define known dangerous combinations
        known_dangerous = [
            ('grasper', 'grasp', 'liver'),      # Fragile organ
            ('scissors', 'cut', 'cystic_artery'), # Bleeding risk
            ('scissors', 'cut', 'blood_vessel'),   # Bleeding risk
        ]
        
        # Analyze actual combinations in training data
        for video in train_data:
            actions_binaries = video['actions_binaries']

            self.logger.info(f"   Processing video: {video['video_id']} with {len(actions_binaries)} frames")
            
            for frame_idx in range(len(actions_binaries)):
                active_actions = np.where(actions_binaries[frame_idx] > 0.5)[0]
                
                for action_id in active_actions:
                    if action_id in self.action_triplets and not self.action_triplets[action_id]['is_null']:
                        triplet = self.action_triplets[action_id]
                        instrument = triplet['instrument']
                        verb = triplet['verb']
                        target = triplet['target']
                        
                        # Count instrument-target combinations
                        combination_patterns['instrument_target_safety'][instrument][target] += 1
                        
                        # Count verb-target combinations
                        combination_patterns['verb_target_safety'][verb][target] += 1
                        
                        # Check if this is a known dangerous combination
                        if (instrument, verb, target) in known_dangerous:
                            combination_patterns['dangerous_combinations'].append({
                                'combination': (instrument, verb, target),
                                'action_id': action_id,
                                'frequency': 1,
                                'risk_reason': self._get_combination_risk_reason(instrument, verb, target)
                            })
        
        # Analyze technique preferences (e.g., clip vs cut for arteries)
        technique_analysis = {
            'cystic_artery': {'clip': 0, 'cut': 0, 'coagulate': 0},
            'blood_vessel': {'clip': 0, 'cut': 0, 'coagulate': 0},
            'cystic_duct': {'clip': 0, 'cut': 0, 'coagulate': 0}
        }
        
        for target in technique_analysis.keys():
            for verb in technique_analysis[target].keys():
                count = combination_patterns['verb_target_safety'][verb].get(target, 0)
                technique_analysis[target][verb] = count
        
        combination_patterns['technique_preferences'] = technique_analysis
        
        self.priors['combination_patterns'] = combination_patterns
        
        self.logger.info(f"   ✅ Analyzed {len(combination_patterns['instrument_target_safety'])} instrument-target patterns")
        self.logger.info(f"   ✅ Found {len(combination_patterns['dangerous_combinations'])} dangerous combination instances")
    
    def _extract_workflow_constraints(self, train_data: List[Dict]):
        """Extract workflow constraint patterns for phase safety"""
        
        workflow_patterns = {
            'phase_action_frequency': defaultdict(lambda: defaultdict(int)),
            'phase_transitions': defaultdict(lambda: defaultdict(int)),
            'inappropriate_actions': defaultdict(list),
            'phase_duration_stats': defaultdict(list)
        }
        
        for video in train_data:
            actions_binaries = video['actions_binaries']
            phase_binaries = video['phase_binaries']

            self.logger.info(f"   Processing video: {video['video_id']} with {len(actions_binaries)} frames")
            
            prev_phase = None
            phase_start_frame = 0
            
            for frame_idx in range(len(actions_binaries)):
                # Get current phase
                phase_id = np.argmax(phase_binaries[frame_idx])
                phase_name = self.phases.get(str(phase_id), 'unknown')
                
                # Track phase transitions
                if prev_phase is not None and prev_phase != phase_name:
                    workflow_patterns['phase_transitions'][prev_phase][phase_name] += 1
                    
                    # Calculate phase duration
                    phase_duration = frame_idx - phase_start_frame
                    workflow_patterns['phase_duration_stats'][prev_phase].append(phase_duration)
                    phase_start_frame = frame_idx
                
                # Get active actions in this phase
                active_actions = np.where(actions_binaries[frame_idx] > 0.5)[0]
                
                for action_id in active_actions:
                    action_str = self.actions.get(str(action_id), f'action_{action_id}')
                    workflow_patterns['phase_action_frequency'][phase_name][action_str] += 1
                    
                    # Check for potentially inappropriate actions
                    if self._is_potentially_inappropriate_action(action_id, phase_name):
                        workflow_patterns['inappropriate_actions'][phase_name].append({
                            'action_id': action_id,
                            'action_str': action_str,
                            'frame_idx': frame_idx,
                            'video_id': video['video_id']
                        })
                
                prev_phase = phase_name
        
        # Calculate phase duration statistics
        for phase_name, durations in workflow_patterns['phase_duration_stats'].items():
            if durations:
                workflow_patterns['phase_duration_stats'][phase_name] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'count': len(durations)
                }
        
        self.priors['workflow_constraints'] = workflow_patterns
        
        self.logger.info(f"   ✅ Analyzed {len(workflow_patterns['phase_action_frequency'])} phase-action patterns")
        self.logger.info(f"   ✅ Found {sum(len(actions) for actions in workflow_patterns['inappropriate_actions'].values())} potentially inappropriate actions")
    
    def _detect_safety_violations(self, train_data: List[Dict]):
        """Detect potential safety violations in training data"""
        
        violations = []
        
        # Define violation patterns to look for
        violation_patterns = [
            {
                'name': 'generic_vessel_clipping',
                'pattern': ('clipper', 'clip', 'blood_vessel'),
                'risk': 'Wrong vessel targeting - should be specific'
            },
            {
                'name': 'liver_grasping',
                'pattern': ('grasper', 'grasp', 'liver'),
                'risk': 'Fragile organ damage risk'
            },
            {
                'name': 'artery_cutting',
                'pattern': ('scissors', 'cut', 'cystic_artery'),
                'risk': 'Bleeding risk - should clip arteries'
            }
        ]
        
        for video in train_data:
            actions_binaries = video['actions_binaries']
            phase_binaries = video['phase_binaries']

            self.logger.info(f"   Processing video: {video['video_id']} with {len(actions_binaries)} frames")
            
            for frame_idx in range(len(actions_binaries)):
                active_actions = np.where(actions_binaries[frame_idx] > 0.5)[0]
                phase_id = np.argmax(phase_binaries[frame_idx])
                phase_name = self.phases.get(str(phase_id), 'unknown')
                
                for action_id in active_actions:
                    if action_id in self.action_triplets and not self.action_triplets[action_id]['is_null']:
                        triplet = self.action_triplets[action_id]
                        combination = (triplet['instrument'], triplet['verb'], triplet['target'])
                        
                        # Check against violation patterns
                        for violation in violation_patterns:
                            if combination == violation['pattern']:
                                violations.append({
                                    'violation_type': violation['name'],
                                    'action_id': action_id,
                                    'combination': combination,
                                    'risk_description': violation['risk'],
                                    'video_id': video['video_id'],
                                    'frame_idx': frame_idx,
                                    'phase': phase_name
                                })
        
        self.priors['safety_violations'] = violations
        
        self.logger.info(f"   ✅ Detected {len(violations)} potential safety violations")
    
    def _is_potentially_inappropriate_action(self, action_id: int, phase_name: str) -> bool:
        """Check if action is potentially inappropriate for phase"""
        
        if action_id not in self.action_triplets:
            return False
        
        triplet = self.action_triplets[action_id]
        if triplet['is_null']:
            return False
        
        # Define phase-inappropriate patterns
        inappropriate_patterns = {
            'preparation': [('clipper', 'clip')],  # No clipping in preparation
            'gallbladder-packaging': [('scissors', 'cut'), ('bipolar', 'dissect')],  # No cutting/dissecting in packaging
            'gallbladder-extraction': [('clipper', 'clip'), ('scissors', 'cut')]  # No major surgery in extraction
        }
        
        if phase_name in inappropriate_patterns:
            for instrument, verb in inappropriate_patterns[phase_name]:
                if triplet['instrument'] == instrument and triplet['verb'] == verb:
                    return True
        
        return False
    
    def _get_combination_risk_reason(self, instrument: str, verb: str, target: str) -> str:
        """Get risk reason for combination"""
        
        risk_patterns = {
            ('grasper', 'grasp', 'liver'): 'Liver is fragile - high bleeding risk',
            ('scissors', 'cut', 'cystic_artery'): 'Cutting arteries causes bleeding - should clip',
            ('scissors', 'cut', 'blood_vessel'): 'Uncontrolled bleeding risk',
        }
        
        return risk_patterns.get((instrument, verb, target), 'Unknown risk pattern')
    
    def _generate_extraction_summary(self) -> Dict[str, Any]:
        """Generate summary of extracted priors"""
        
        summary = {
            'total_patterns': 0,
            'action_phase_patterns': len(self.priors['action_phase_cooccurrence']),
            'rare_targets': len(self.priors['component_difficulty'].get('rare_targets', [])),
            'rare_combinations': len(self.priors['component_difficulty'].get('rare_combinations', [])),
            'dangerous_proximity_patterns': len(self.priors['anatomical_relationships'].get('dangerous_proximity', [])),
            'safety_violations': len(self.priors['safety_violations']),
            'workflow_constraints': len(self.priors['workflow_constraints'].get('phase_action_frequency', {})),
            'extraction_timestamp': datetime.now().isoformat(),
            'target_components': ['anatomical_targets', 'surgical_combinations'],
            'focus_areas': ['T_component_52_percent', 'IVT_component_33_percent']
        }
        
        summary['total_patterns'] = sum([
            summary['action_phase_patterns'],
            summary['rare_targets'],
            summary['rare_combinations'],
            summary['dangerous_proximity_patterns'],
            summary['safety_violations'],
            summary['workflow_constraints']
        ])
        
        return summary
    
    def save_priors(self, output_dir: str):
        """Save extracted priors to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main priors file
        priors_file = output_path / 'surgical_priors.json'
        with open(priors_file, 'w') as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            json_priors = self._convert_for_json(self.priors)
            json.dump(json_priors, f, indent=2, default=str)
        
        # Save component-specific files
        self._save_component_analysis(output_path)
        self._save_safety_analysis(output_path)
        self._save_workflow_analysis(output_path)
        
        # Generate visualizations
        self._generate_visualizations(output_path)
        
        self.logger.info(f"✅ Priors saved to: {output_path}")
        self.logger.info(f"   Main file: {priors_file}")
    
    def _save_component_analysis(self, output_path: Path):
        """Save component-specific analysis"""
        
        component_file = output_path / 'component_difficulty_analysis.json'
        with open(component_file, 'w') as f:
            json.dump(self._convert_for_json(self.priors['component_difficulty']), f, indent=2, default=str)
    
    def _save_safety_analysis(self, output_path: Path):
        """Save safety analysis"""
        
        safety_file = output_path / 'safety_patterns.json'
        safety_data = {
            'anatomical_relationships': self.priors['anatomical_relationships'],
            'combination_patterns': self.priors['combination_patterns'],
            'safety_violations': self.priors['safety_violations']
        }
        with open(safety_file, 'w') as f:
            json.dump(self._convert_for_json(safety_data), f, indent=2, default=str)
    
    def _save_workflow_analysis(self, output_path: Path):
        """Save workflow analysis"""
        
        workflow_file = output_path / 'workflow_constraints.json'
        with open(workflow_file, 'w') as f:
            json.dump(self._convert_for_json(self.priors['workflow_constraints']), f, indent=2, default=str)
    
    def _generate_visualizations(self, output_path: Path):
        """Generate visualization plots"""
        
        try:
            # Plot 1: Component difficulty
            self._plot_component_difficulty(output_path)
            
            # Plot 2: Phase-action frequency
            self._plot_phase_action_frequency(output_path)
            
            # Plot 3: Safety violations
            self._plot_safety_violations(output_path)
            
            self.logger.info("✅ Visualizations generated")
        except Exception as e:
            self.logger.warning(f"⚠️ Visualization generation failed: {e}")
    
    def _plot_component_difficulty(self, output_path: Path):
        """Plot component difficulty analysis"""
        
        rare_targets = self.priors['component_difficulty'].get('rare_targets', [])
        if not rare_targets:
            return
        
        # Take top 10 rarest targets
        top_rare = rare_targets[:10]
        targets = [item[0] for item in top_rare]
        frequencies = [item[2] for item in top_rare]  # Use percentage
        
        plt.figure(figsize=(12, 6))
        plt.bar(targets, frequencies)
        plt.title('Rarest Surgical Targets (High-Opportunity for IRL)')
        plt.xlabel('Surgical Targets')
        plt.ylabel('Frequency in Training Data')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'component_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_phase_action_frequency(self, output_path: Path):
        """Plot phase-action frequency heatmap"""
        
        phase_action_freq = self.priors['action_phase_cooccurrence']
        if not phase_action_freq:
            return
        
        # Create frequency matrix for top actions
        phases = list(phase_action_freq.keys())
        all_actions = set()
        for phase_actions in phase_action_freq.values():
            all_actions.update(phase_actions.keys())
        
        # Limit to top 20 most frequent actions for readability
        action_counts = defaultdict(int)
        for phase_actions in phase_action_freq.values():
            for action, count in phase_actions.items():
                action_counts[action] += count
        
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        top_action_names = [item[0] for item in top_actions]
        
        # Create matrix
        matrix = np.zeros((len(phases), len(top_action_names)))
        for i, phase in enumerate(phases):
            for j, action in enumerate(top_action_names):
                matrix[i, j] = phase_action_freq[phase].get(action, 0)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(matrix, xticklabels=top_action_names, yticklabels=phases, cmap='YlOrRd')
        plt.title('Phase-Action Co-occurrence Frequency')
        plt.xlabel('Actions')
        plt.ylabel('Surgical Phases')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'phase_action_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_safety_violations(self, output_path: Path):
        """Plot safety violations distribution"""
        
        violations = self.priors['safety_violations']
        if not violations:
            return
        
        # Count violations by type
        violation_counts = defaultdict(int)
        for violation in violations:
            violation_counts[violation['violation_type']] += 1
        
        if violation_counts:
            plt.figure(figsize=(10, 6))
            types = list(violation_counts.keys())
            counts = list(violation_counts.values())
            
            plt.bar(types, counts, color='red', alpha=0.7)
            plt.title('Detected Safety Violations in Training Data')
            plt.xlabel('Violation Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'safety_violations.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _convert_for_json(self, obj):
        """Convert object to JSON-serializable format"""
        
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def extract_surgical_priors(config_path: str = 'config_dgx_all_v8.yaml', 
                           output_dir: str = 'data/surgical_priors',
                           max_videos: int = None):
    """
    Main function to extract surgical priors for negative generation
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save extracted priors
        max_videos: Maximum number of videos to process (None for all)
    """
    
    print("🔬 SURGICAL PRIOR EXTRACTION FOR SAFETY GUARDRAILS")
    print("=" * 60)
    print("Extracting prior knowledge to inform negative generation")
    print("Focus: High-opportunity components (T: 52%, IVT: 33%)")
    print()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create simple logger
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    logger = SimpleLogger()
    
    # Load training data
    logger.info("📂 Loading training data...")
    train_data = load_cholect50_data(
        config, logger, 
        split='train', 
        max_videos=max_videos
    )
    
    if not train_data:
        logger.error("❌ No training data loaded")
        return False
    
    logger.info(f"✅ Loaded {len(train_data)} training videos")
    
    # Extract priors
    extractor = SurgicalPriorExtractor(config, logger)
    priors = extractor.extract_all_priors(train_data)
    
    # Save results
    extractor.save_priors(output_dir)
    
    # Print summary
    summary = priors['extraction_summary']
    print("\n🎯 EXTRACTION SUMMARY:")
    print(f"   Total Patterns: {summary['total_patterns']}")
    print(f"   Rare Targets: {summary['rare_targets']}")
    print(f"   Rare Combinations: {summary['rare_combinations']}")
    print(f"   Safety Violations: {summary['safety_violations']}")
    print(f"   Workflow Constraints: {summary['workflow_constraints']}")
    print(f"\n📁 Results saved to: {output_dir}/")
    print("✅ Prior extraction completed successfully!")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract surgical priors for safety guardrails")
    parser.add_argument('--config', type=str, default='config_dgx_all_v8.yaml', help="Configuration file")
    parser.add_argument('--output', type=str, default='surgical_priors', help="Output directory")
    parser.add_argument('--max_videos', type=int, default=None, help="Maximum videos to process")
    
    args = parser.parse_args()
    
    success = extract_surgical_priors(
        config_path=args.config,
        output_dir=args.output,
        max_videos=args.max_videos
    )
    
    if success:
        print("\n🎉 Ready to proceed with IRL training!")
        print("Use the extracted priors to shape your negative generation strategy.")
    else:
        print("\n❌ Prior extraction failed. Check your data and configuration.")
