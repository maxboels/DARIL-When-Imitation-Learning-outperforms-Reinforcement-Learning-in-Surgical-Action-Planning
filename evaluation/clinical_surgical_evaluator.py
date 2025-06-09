#!/usr/bin/env python3
"""
Clinically-Informed Surgical Action Evaluation Framework
Uses the CholecT50 action taxonomy to provide meaningful, interpretable evaluation
"""
import json
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class ClinicalSurgicalEvaluator:
    """
    Evaluation framework that uses surgical domain knowledge for meaningful assessment.
    """
    
    def __init__(self, labels_path: str = './data/labels.json'):
        """
        Args:
            labels_path: Path to the labels.json file with surgical taxonomy
        """
        # Load surgical taxonomy
        with open(labels_path, 'r') as f:
            self.taxonomy = json.load(f)
        
        # Parse action structure and create clinical groupings
        self.action_info = self._parse_action_taxonomy()
        self.clinical_weights = self._define_clinical_weights()
        self.phase_mappings = self._map_actions_to_phases()
        self.complexity_tiers = self._define_complexity_tiers()
        
    def _parse_action_taxonomy(self) -> Dict:
        """Parse the action taxonomy into structured information."""
        
        action_info = {}
        
        for action_id_str, action_desc in self.taxonomy['action'].items():
            action_id = int(action_id_str)
            
            if action_desc.count(',') == 2:
                instrument, verb, target = action_desc.split(',')
                
                action_info[action_id] = {
                    'instrument': instrument,
                    'verb': verb,
                    'target': target,
                    'is_null': verb == 'null_verb',
                    'full_description': action_desc
                }
            else:
                # Handle any edge cases
                action_info[action_id] = {
                    'instrument': 'unknown',
                    'verb': 'unknown', 
                    'target': 'unknown',
                    'is_null': False,
                    'full_description': action_desc
                }
        
        return action_info
    
    def _define_clinical_weights(self) -> Dict[int, float]:
        """Define clinical importance weights based on surgical criticality."""
        
        weights = {}
        
        # Critical anatomical targets (highest risk if mishandled)
        critical_targets = {'cystic_artery', 'cystic_duct', 'blood_vessel'}
        important_targets = {'gallbladder', 'cystic_plate', 'cystic_pedicle'}
        supportive_targets = {'liver', 'omentum', 'peritoneum', 'gut'}
        
        # Critical verbs (irreversible actions)
        critical_verbs = {'cut', 'clip', 'coagulate'}
        important_verbs = {'dissect', 'pack'}
        supportive_verbs = {'grasp', 'retract', 'aspirate', 'irrigate'}
        
        for action_id, info in self.action_info.items():
            if info['is_null']:
                weights[action_id] = 1.0  # Null actions - baseline importance
                continue
            
            target = info['target']
            verb = info['verb']
            
            # Base weight calculation
            target_weight = 3.0 if target in critical_targets else \
                           2.0 if target in important_targets else 1.0
                           
            verb_weight = 3.0 if verb in critical_verbs else \
                         2.0 if verb in important_verbs else 1.0
            
            # Combined weight (max 9.0 for critical target + critical verb)
            combined_weight = (target_weight + verb_weight) / 2.0
            
            # Special cases for extremely critical actions
            if verb == 'cut' and target in {'cystic_artery', 'cystic_duct'}:
                combined_weight = 4.0  # Maximum criticality
            elif verb == 'clip' and target in {'cystic_artery', 'cystic_duct'}:
                combined_weight = 4.0  # Maximum criticality
            
            weights[action_id] = combined_weight
        
        return weights
    
    def _map_actions_to_phases(self) -> Dict[str, List[int]]:
        """Map actions to typical surgical phases based on clinical knowledge."""
        
        phase_mappings = {
            'preparation': [],  # Actions 94-99 (null actions) + initial grasping
            'calot_triangle_dissection': [],  # Dissection of critical view
            'clipping_and_cutting': [],  # Clipping arteries/ducts, cutting
            'gallbladder_dissection': [],  # Main dissection work
            'gallbladder_packaging': [],  # Specimen bag related
            'cleaning_and_coagulation': [],  # Irrigation, coagulation
            'gallbladder_extraction': []  # Final extraction
        }
        
        for action_id, info in self.action_info.items():
            if info['is_null']:
                phase_mappings['preparation'].append(action_id)
            elif info['verb'] in {'dissect'} and info['target'] in {'cystic_plate', 'cystic_artery', 'cystic_duct'}:
                phase_mappings['calot_triangle_dissection'].append(action_id)
            elif info['verb'] in {'clip', 'cut'} and info['target'] in {'cystic_artery', 'cystic_duct', 'blood_vessel'}:
                phase_mappings['clipping_and_cutting'].append(action_id)
            elif info['verb'] in {'dissect'} and info['target'] in {'gallbladder'}:
                phase_mappings['gallbladder_dissection'].append(action_id)
            elif info['target'] in {'specimen_bag'} or info['verb'] in {'pack'}:
                phase_mappings['gallbladder_packaging'].append(action_id)
            elif info['verb'] in {'coagulate', 'irrigate', 'aspirate'}:
                phase_mappings['cleaning_and_coagulation'].append(action_id)
            elif info['verb'] in {'retract'} and info['target'] in {'gallbladder'}:
                phase_mappings['gallbladder_extraction'].append(action_id)
            else:
                # Default assignment based on target
                if info['target'] in {'gallbladder'}:
                    phase_mappings['gallbladder_dissection'].append(action_id)
                else:
                    phase_mappings['calot_triangle_dissection'].append(action_id)
        
        return phase_mappings
    
    def _define_complexity_tiers(self) -> Dict[str, List[int]]:
        """Define complexity tiers based on surgical skill requirements."""
        
        tiers = {
            'basic': [],      # Simple grasping, retraction
            'intermediate': [], # Dissection, coagulation
            'advanced': [],   # Cutting, clipping critical structures
            'expert': []      # Complex procedures on critical anatomy
        }
        
        for action_id, info in self.action_info.items():
            if info['is_null']:
                tiers['basic'].append(action_id)
            elif info['verb'] in {'grasp', 'retract'} and info['target'] not in {'cystic_artery', 'cystic_duct'}:
                tiers['basic'].append(action_id)
            elif info['verb'] in {'dissect', 'coagulate', 'irrigate', 'aspirate'}:
                if info['target'] in {'cystic_artery', 'cystic_duct'}:
                    tiers['advanced'].append(action_id)
                else:
                    tiers['intermediate'].append(action_id)
            elif info['verb'] in {'cut', 'clip'}:
                if info['target'] in {'cystic_artery', 'cystic_duct', 'blood_vessel'}:
                    tiers['expert'].append(action_id)
                else:
                    tiers['advanced'].append(action_id)
            else:
                tiers['intermediate'].append(action_id)
        
        return tiers
    
    def evaluate_clinical_performance(self, predictions: np.ndarray, ground_truth: np.ndarray,
                                    occurring_actions: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive clinical evaluation of surgical action predictions.
        
        Args:
            predictions: [n_samples, n_actions] prediction probabilities
            ground_truth: [n_samples, n_actions] binary ground truth
            occurring_actions: Boolean mask for actions that occur in dataset
            
        Returns:
            Comprehensive clinical evaluation results
        """
        
        if occurring_actions is None:
            occurring_actions = np.sum(ground_truth, axis=0) > 0
        
        results = {}
        
        # 1. Overall clinical metrics
        results['clinical_overview'] = self._compute_clinical_overview(
            predictions, ground_truth, occurring_actions)
        
        # 2. Instrument-based analysis
        results['instrument_analysis'] = self._evaluate_by_instrument(
            predictions, ground_truth, occurring_actions)
        
        # 3. Anatomical target analysis
        results['anatomical_analysis'] = self._evaluate_by_target(
            predictions, ground_truth, occurring_actions)
        
        # 4. Surgical verb analysis  
        results['procedural_analysis'] = self._evaluate_by_verb(
            predictions, ground_truth, occurring_actions)
        
        # 5. Phase-based analysis
        results['phase_analysis'] = self._evaluate_by_phase(
            predictions, ground_truth, occurring_actions)
        
        # 6. Complexity tier analysis
        results['complexity_analysis'] = self._evaluate_by_complexity(
            predictions, ground_truth, occurring_actions)
        
        # 7. Clinical criticality analysis
        results['criticality_analysis'] = self._evaluate_by_criticality(
            predictions, ground_truth, occurring_actions)
        
        return results
    
    def _compute_clinical_overview(self, predictions: np.ndarray, ground_truth: np.ndarray,
                                  occurring_actions: np.ndarray) -> Dict:
        """Compute overall clinical performance metrics."""
        
        # Standard mAP vs Fair mAP
        all_ap_scores = []
        occurring_ap_scores = []
        
        for i in range(predictions.shape[1]):
            if np.sum(ground_truth[:, i]) > 0:
                ap = average_precision_score(ground_truth[:, i], predictions[:, i])
                all_ap_scores.append(ap)
                if occurring_actions[i]:
                    occurring_ap_scores.append(ap)
            else:
                # Non-occurring action
                ap = 1.0 if np.max(predictions[:, i]) < 0.5 else 0.0
                all_ap_scores.append(ap)
        
        # Clinical importance weighted mAP
        clinical_ap_scores = []
        clinical_weights_used = []
        
        for i in range(predictions.shape[1]):
            if occurring_actions[i] and np.sum(ground_truth[:, i]) > 0:
                ap = average_precision_score(ground_truth[:, i], predictions[:, i])
                weight = self.clinical_weights.get(i, 1.0)
                clinical_ap_scores.append(ap)
                clinical_weights_used.append(weight)
        
        # Weighted average
        if clinical_ap_scores and clinical_weights_used:
            weights_array = np.array(clinical_weights_used)
            weights_normalized = weights_array / np.sum(weights_array)
            clinical_weighted_map = np.sum(np.array(clinical_ap_scores) * weights_normalized)
        else:
            clinical_weighted_map = 0.0
        
        return {
            'standard_map': np.mean(all_ap_scores),
            'fair_map': np.mean(occurring_ap_scores) if occurring_ap_scores else 0.0,
            'clinical_weighted_map': clinical_weighted_map,
            'total_actions': len(all_ap_scores),
            'occurring_actions': len(occurring_ap_scores),
            'evaluated_actions': len(clinical_ap_scores)
        }
    
    def _evaluate_by_instrument(self, predictions: np.ndarray, ground_truth: np.ndarray,
                               occurring_actions: np.ndarray) -> Dict:
        """Evaluate performance by surgical instrument."""
        
        instrument_groups = {}
        
        # Group actions by instrument
        for action_id, info in self.action_info.items():
            instrument = info['instrument']
            if instrument not in instrument_groups:
                instrument_groups[instrument] = []
            instrument_groups[instrument].append(action_id)
        
        results = {}
        
        for instrument, action_ids in instrument_groups.items():
            occurring_in_group = [aid for aid in action_ids 
                                if aid < len(occurring_actions) and occurring_actions[aid]]
            
            if not occurring_in_group:
                results[instrument] = {'map': 0.0, 'count': 0, 'actions': []}
                continue
            
            ap_scores = []
            for action_id in occurring_in_group:
                if np.sum(ground_truth[:, action_id]) > 0:
                    ap = average_precision_score(ground_truth[:, action_id], predictions[:, action_id])
                    ap_scores.append(ap)
            
            results[instrument] = {
                'map': np.mean(ap_scores) if ap_scores else 0.0,
                'count': len(occurring_in_group),
                'actions': occurring_in_group,
                'ap_scores': ap_scores
            }
        
        return results
    
    def _evaluate_by_target(self, predictions: np.ndarray, ground_truth: np.ndarray,
                           occurring_actions: np.ndarray) -> Dict:
        """Evaluate performance by anatomical target."""
        
        target_groups = {}
        
        # Group actions by target
        for action_id, info in self.action_info.items():
            target = info['target']
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(action_id)
        
        results = {}
        
        for target, action_ids in target_groups.items():
            occurring_in_group = [aid for aid in action_ids 
                                if aid < len(occurring_actions) and occurring_actions[aid]]
            
            if not occurring_in_group:
                results[target] = {'map': 0.0, 'count': 0, 'actions': []}
                continue
            
            ap_scores = []
            for action_id in occurring_in_group:
                if np.sum(ground_truth[:, action_id]) > 0:
                    ap = average_precision_score(ground_truth[:, action_id], predictions[:, action_id])
                    ap_scores.append(ap)
            
            results[target] = {
                'map': np.mean(ap_scores) if ap_scores else 0.0,
                'count': len(occurring_in_group),
                'actions': occurring_in_group,
                'ap_scores': ap_scores
            }
        
        return results
    
    def _evaluate_by_verb(self, predictions: np.ndarray, ground_truth: np.ndarray,
                         occurring_actions: np.ndarray) -> Dict:
        """Evaluate performance by surgical procedure/verb."""
        
        verb_groups = {}
        
        # Group actions by verb
        for action_id, info in self.action_info.items():
            verb = info['verb']
            if verb not in verb_groups:
                verb_groups[verb] = []
            verb_groups[verb].append(action_id)
        
        results = {}
        
        for verb, action_ids in verb_groups.items():
            occurring_in_group = [aid for aid in action_ids 
                                if aid < len(occurring_actions) and occurring_actions[aid]]
            
            if not occurring_in_group:
                results[verb] = {'map': 0.0, 'count': 0, 'actions': []}
                continue
            
            ap_scores = []
            for action_id in occurring_in_group:
                if np.sum(ground_truth[:, action_id]) > 0:
                    ap = average_precision_score(ground_truth[:, action_id], predictions[:, action_id])
                    ap_scores.append(ap)
            
            results[verb] = {
                'map': np.mean(ap_scores) if ap_scores else 0.0,
                'count': len(occurring_in_group),
                'actions': occurring_in_group,
                'ap_scores': ap_scores
            }
        
        return results
    
    def _evaluate_by_phase(self, predictions: np.ndarray, ground_truth: np.ndarray,
                          occurring_actions: np.ndarray) -> Dict:
        """Evaluate performance by surgical phase."""
        
        results = {}
        
        for phase, action_ids in self.phase_mappings.items():
            occurring_in_phase = [aid for aid in action_ids 
                                if aid < len(occurring_actions) and occurring_actions[aid]]
            
            if not occurring_in_phase:
                results[phase] = {'map': 0.0, 'count': 0, 'actions': []}
                continue
            
            ap_scores = []
            for action_id in occurring_in_phase:
                if np.sum(ground_truth[:, action_id]) > 0:
                    ap = average_precision_score(ground_truth[:, action_id], predictions[:, action_id])
                    ap_scores.append(ap)
            
            results[phase] = {
                'map': np.mean(ap_scores) if ap_scores else 0.0,
                'count': len(occurring_in_phase),
                'actions': occurring_in_phase,
                'ap_scores': ap_scores
            }
        
        return results
    
    def _evaluate_by_complexity(self, predictions: np.ndarray, ground_truth: np.ndarray,
                               occurring_actions: np.ndarray) -> Dict:
        """Evaluate performance by procedure complexity."""
        
        results = {}
        
        for complexity, action_ids in self.complexity_tiers.items():
            occurring_in_tier = [aid for aid in action_ids 
                               if aid < len(occurring_actions) and occurring_actions[aid]]
            
            if not occurring_in_tier:
                results[complexity] = {'map': 0.0, 'count': 0, 'actions': []}
                continue
            
            ap_scores = []
            for action_id in occurring_in_tier:
                if np.sum(ground_truth[:, action_id]) > 0:
                    ap = average_precision_score(ground_truth[:, action_id], predictions[:, action_id])
                    ap_scores.append(ap)
            
            results[complexity] = {
                'map': np.mean(ap_scores) if ap_scores else 0.0,
                'count': len(occurring_in_tier),
                'actions': occurring_in_tier,
                'ap_scores': ap_scores
            }
        
        return results
    
    def _evaluate_by_criticality(self, predictions: np.ndarray, ground_truth: np.ndarray,
                                occurring_actions: np.ndarray) -> Dict:
        """Evaluate performance by clinical criticality levels."""
        
        # Group actions by clinical weight (criticality)
        criticality_groups = {
            'routine': [],      # Weight 1.0-1.5
            'important': [],    # Weight 1.5-2.5  
            'critical': [],     # Weight 2.5-3.5
            'extremely_critical': [] # Weight 3.5+
        }
        
        for action_id, weight in self.clinical_weights.items():
            if weight <= 1.5:
                criticality_groups['routine'].append(action_id)
            elif weight <= 2.5:
                criticality_groups['important'].append(action_id)
            elif weight <= 3.5:
                criticality_groups['critical'].append(action_id)
            else:
                criticality_groups['extremely_critical'].append(action_id)
        
        results = {}
        
        for criticality, action_ids in criticality_groups.items():
            occurring_in_group = [aid for aid in action_ids 
                                if aid < len(occurring_actions) and occurring_actions[aid]]
            
            if not occurring_in_group:
                results[criticality] = {'map': 0.0, 'count': 0, 'actions': []}
                continue
            
            ap_scores = []
            weights = []
            for action_id in occurring_in_group:
                if np.sum(ground_truth[:, action_id]) > 0:
                    ap = average_precision_score(ground_truth[:, action_id], predictions[:, action_id])
                    weight = self.clinical_weights[action_id]
                    ap_scores.append(ap)
                    weights.append(weight)
            
            results[criticality] = {
                'map': np.mean(ap_scores) if ap_scores else 0.0,
                'weighted_map': np.average(ap_scores, weights=weights) if ap_scores else 0.0,
                'count': len(occurring_in_group),
                'actions': occurring_in_group,
                'ap_scores': ap_scores,
                'avg_weight': np.mean(weights) if weights else 0.0
            }
        
        return results

def generate_clinical_evaluation_report(results: Dict, evaluator: ClinicalSurgicalEvaluator) -> str:
    """Generate a comprehensive clinical evaluation report."""
    
    report = []
    report.append("🏥 CLINICAL SURGICAL ACTION EVALUATION REPORT")
    report.append("=" * 60)
    
    # Clinical Overview
    overview = results['clinical_overview']
    report.append("📊 Clinical Performance Overview:")
    report.append(f"   Standard mAP (inflated): {overview['standard_map']:.4f}")
    report.append(f"   Fair mAP (occurring only): {overview['fair_map']:.4f}")
    report.append(f"   Clinical Weighted mAP: {overview['clinical_weighted_map']:.4f}")
    report.append(f"   Actions evaluated: {overview['evaluated_actions']}/{overview['occurring_actions']}")
    report.append("")
    
    # Instrument Analysis
    instrument_analysis = results['instrument_analysis']
    report.append("🔧 Performance by Surgical Instrument:")
    for instrument, perf in instrument_analysis.items():
        if perf['count'] > 0:
            report.append(f"   {instrument.title()}: {perf['map']:.4f} mAP ({perf['count']} actions)")
    report.append("")
    
    # Anatomical Target Analysis
    anatomical_analysis = results['anatomical_analysis']
    report.append("🎯 Performance by Anatomical Target:")
    # Sort by clinical importance
    critical_targets = ['cystic_artery', 'cystic_duct', 'gallbladder', 'blood_vessel']
    for target in critical_targets:
        if target in anatomical_analysis and anatomical_analysis[target]['count'] > 0:
            perf = anatomical_analysis[target]
            report.append(f"   {target.replace('_', ' ').title()}: {perf['map']:.4f} mAP ({perf['count']} actions)")
    report.append("")
    
    # Procedural Analysis
    procedural_analysis = results['procedural_analysis']
    report.append("⚕️ Performance by Surgical Procedure:")
    critical_verbs = ['cut', 'clip', 'dissect', 'coagulate']
    for verb in critical_verbs:
        if verb in procedural_analysis and procedural_analysis[verb]['count'] > 0:
            perf = procedural_analysis[verb]
            report.append(f"   {verb.title()}: {perf['map']:.4f} mAP ({perf['count']} actions)")
    report.append("")
    
    # Complexity Analysis
    complexity_analysis = results['complexity_analysis']
    report.append("📈 Performance by Surgical Complexity:")
    complexity_order = ['basic', 'intermediate', 'advanced', 'expert']
    for complexity in complexity_order:
        if complexity in complexity_analysis and complexity_analysis[complexity]['count'] > 0:
            perf = complexity_analysis[complexity]
            report.append(f"   {complexity.title()}: {perf['map']:.4f} mAP ({perf['count']} actions)")
    report.append("")
    
    # Criticality Analysis
    criticality_analysis = results['criticality_analysis']
    report.append("🚨 Performance by Clinical Criticality:")
    criticality_order = ['routine', 'important', 'critical', 'extremely_critical']
    for criticality in criticality_order:
        if criticality in criticality_analysis and criticality_analysis[criticality]['count'] > 0:
            perf = criticality_analysis[criticality]
            report.append(f"   {criticality.replace('_', ' ').title()}: {perf['map']:.4f} mAP ({perf['count']} actions)")
    report.append("")
    
    # Phase Analysis
    phase_analysis = results['phase_analysis']
    report.append("🔄 Performance by Surgical Phase:")
    for phase, perf in phase_analysis.items():
        if perf['count'] > 0:
            phase_name = phase.replace('_', ' ').title()
            report.append(f"   {phase_name}: {perf['map']:.4f} mAP ({perf['count']} actions)")
    
    return "\n".join(report)

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ClinicalSurgicalEvaluator('./data/labels.json')
    
    print("🏥 Clinical Surgical Action Evaluator Initialized")
    print(f"📊 Clinical weights defined for {len(evaluator.clinical_weights)} actions")
    print(f"🔧 Instruments: {list(evaluator.taxonomy['instrument'].values())}")
    print(f"⚕️ Procedures: {list(evaluator.taxonomy['verb'].values())}")
    print(f"🎯 Targets: {list(evaluator.taxonomy['target'].values())}")
    
    # Example of clinical weights for most critical actions
    print("\n🚨 Most Critical Actions (Weight ≥ 3.0):")
    critical_actions = [(aid, weight) for aid, weight in evaluator.clinical_weights.items() if weight >= 3.0]
    critical_actions.sort(key=lambda x: x[1], reverse=True)
    
    for action_id, weight in critical_actions[:10]:
        if action_id in evaluator.action_info:
            desc = evaluator.action_info[action_id]['full_description']
            print(f"   Action {action_id}: {desc} (Weight: {weight:.1f})")
