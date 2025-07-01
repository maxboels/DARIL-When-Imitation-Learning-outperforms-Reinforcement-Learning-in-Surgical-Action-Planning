#!/usr/bin/env python3
"""
Targeted IRL Implementation: Safety Guardrails for High-Opportunity Components
Focus on Targets (T: 52%) and Combinations (IVT: 33%) where IRL can provide maximum clinical value
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import json

class SurgicalSafetyGuardrails:
    """
    IRL-based safety guardrails targeting the highest-opportunity components:
    - Target Component (T): 52% → Focus on anatomical safety
    - Combination Component (IVT): 33% → Focus on contextual appropriateness
    """
    
    def __init__(self, labels_config: Dict):
        self.labels_config = labels_config
        self.actions = labels_config['action']
        self.phases = labels_config['phase']
        
        # Build targeted safety knowledge
        self._build_anatomical_safety_rules()  # For Target component
        self._build_combination_safety_rules()  # For IVT component
        
        print("🛡️ Surgical Safety Guardrails Initialized")
        print(f"   Target: Anatomical safety (T component improvement)")
        print(f"   Target: Combination safety (IVT component improvement)")
    
    def _build_anatomical_safety_rules(self):
        """
        Build safety rules specifically for Target (T) component improvement
        Focus: Anatomical specificity and safety
        """
        
        # High-risk target confusions that cause actual clinical harm
        self.dangerous_target_alternatives = {
            # Specific vessel → Generic vessel (precision loss, wrong targeting)
            'cystic_artery': ['blood_vessel'],  # Must clip specific artery, not generic
            'cystic_duct': ['blood_vessel'],    # Ducts vs vessels confusion
            
            # Critical organ confusions
            'gallbladder': ['liver'],           # Grasping liver can cause bleeding/damage
            'cystic_plate': ['liver'],          # Adjacent anatomy confusion
            
            # Specific vs generic targeting errors
            'cystic_pedicle': ['blood_vessel'], # Pedicle contains multiple structures
        }
        
        # Safe targets that should be preferred in specific contexts
        self.preferred_safe_targets = {
            'preparation_phase': ['peritoneum', 'omentum'],  # Safe initial targets
            'clipping_phase': ['cystic_artery', 'cystic_duct'],  # Precise targets
            'dissection_phase': ['cystic_plate', 'gallbladder'],  # Appropriate structures
        }
    
    def _build_combination_safety_rules(self):
        """
        Build safety rules specifically for Combination (IVT) component improvement
        Focus: Contextually appropriate instrument-verb-target combinations
        """
        
        # Dangerous combinations that should trigger guardrails
        self.dangerous_combinations = {
            # Grasping fragile/critical organs
            ('grasper', 'grasp', 'liver'): 'High bleeding risk - liver is fragile',
            ('grasper', 'grasp', 'blood_vessel'): 'Vessel damage risk - use appropriate tool',
            
            # Wrong technique for anatomy
            ('scissors', 'cut', 'cystic_artery'): 'Bleeding risk - should clip arteries',
            ('scissors', 'cut', 'blood_vessel'): 'Uncontrolled bleeding - should clip',
            
            # Inappropriate tool-target combinations
            ('irrigator', 'clip', 'cystic_artery'): 'Impossible - irrigator cannot clip',
            ('grasper', 'clip', 'cystic_duct'): 'Wrong tool - grasper cannot clip',
            
            # Phase-inappropriate combinations
            ('clipper', 'clip', 'cystic_artery'): 'Context-dependent - wrong if in preparation',
        }
        
        # Safe combination patterns to reinforce
        self.safe_combinations = {
            # Appropriate instrument-target pairs
            ('clipper', 'clip', 'cystic_artery'): 'Standard safe technique',
            ('grasper', 'grasp', 'gallbladder'): 'Appropriate organ handling',
            ('bipolar', 'coagulate', 'blood_vessel'): 'Safe hemostasis technique',
        }
    
    def generate_targeted_safety_negatives(self, expert_actions: torch.Tensor, 
                                         current_phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate safety-focused negatives targeting high-opportunity components
        
        Returns:
            Dictionary with component-specific negatives for targeted improvement
        """
        
        batch_size = expert_actions.shape[0]
        
        negatives = {
            'anatomical_safety_negatives': torch.zeros_like(expert_actions),    # Target T component
            'combination_safety_negatives': torch.zeros_like(expert_actions),   # Target IVT component
            'workflow_safety_negatives': torch.zeros_like(expert_actions),      # Context safety
        }
        
        for i in range(batch_size):
            expert_action_ids = torch.where(expert_actions[i] > 0.5)[0].cpu().numpy()
            phase_id = torch.argmax(current_phase[i]).item()
            phase_name = self.phases.get(str(phase_id), 'unknown')
            
            # Generate anatomical safety negatives (TARGET T COMPONENT)
            anatomical_negatives = self._generate_anatomical_safety_negatives(
                expert_action_ids, phase_name
            )
            for neg_id in anatomical_negatives:
                if 0 <= neg_id < 100:
                    negatives['anatomical_safety_negatives'][i, neg_id] = 1.0
            
            # Generate combination safety negatives (TARGET IVT COMPONENT)
            combination_negatives = self._generate_combination_safety_negatives(
                expert_action_ids, phase_name
            )
            for neg_id in combination_negatives:
                if 0 <= neg_id < 100:
                    negatives['combination_safety_negatives'][i, neg_id] = 1.0
            
            # Generate workflow safety negatives (CONTEXT SAFETY)
            workflow_negatives = self._generate_workflow_safety_negatives(
                expert_action_ids, phase_name
            )
            for neg_id in workflow_negatives:
                if 0 <= neg_id < 100:
                    negatives['workflow_safety_negatives'][i, neg_id] = 1.0
        
        return negatives
    
    def _generate_anatomical_safety_negatives(self, expert_action_ids: List[int], 
                                            phase_name: str) -> List[int]:
        """
        Generate negatives specifically targeting anatomical safety (T component improvement)
        """
        
        negatives = []
        
        # Parse expert actions to find target components
        for action_id in expert_action_ids:
            if action_id not in self.action_triplets:
                continue
            
            expert_action = self.action_triplets[action_id]
            expert_target = expert_action.get('target', 'unknown')
            expert_instrument = expert_action.get('instrument', 'unknown')
            expert_verb = expert_action.get('verb', 'unknown')
            
            # Generate dangerous target alternatives
            if expert_target in self.dangerous_target_alternatives:
                for dangerous_target in self.dangerous_target_alternatives[expert_target]:
                    # Create action with same instrument+verb but dangerous target
                    dangerous_combination = (expert_instrument, expert_verb, dangerous_target)
                    dangerous_action_id = self._find_action_id(dangerous_combination)
                    if dangerous_action_id is not None:
                        negatives.append(dangerous_action_id)
        
        return negatives[:2]  # Limit to 2 negatives per frame
    
    def _generate_combination_safety_negatives(self, expert_action_ids: List[int], 
                                             phase_name: str) -> List[int]:
        """
        Generate negatives specifically targeting combination safety (IVT component improvement)
        """
        
        negatives = []
        
        # Check expert actions against dangerous combination patterns
        for action_id in expert_action_ids:
            if action_id not in self.action_triplets:
                continue
            
            expert_action = self.action_triplets[action_id]
            expert_instrument = expert_action.get('instrument', 'unknown')
            expert_verb = expert_action.get('verb', 'unknown')
            expert_target = expert_action.get('target', 'unknown')
            
            # Generate dangerous combinations by modifying one component
            
            # 1. Keep instrument+verb, change to dangerous target
            for dangerous_combo, reason in self.dangerous_combinations.items():
                danger_instrument, danger_verb, danger_target = dangerous_combo
                
                if (expert_instrument == danger_instrument and 
                    expert_verb == danger_verb and 
                    expert_target != danger_target):
                    # This expert action could be replaced with dangerous target
                    dangerous_action_id = self._find_action_id(dangerous_combo)
                    if dangerous_action_id is not None:
                        negatives.append(dangerous_action_id)
            
            # 2. Keep instrument+target, change to dangerous verb
            dangerous_verbs = ['cut'] if expert_target in ['cystic_artery', 'blood_vessel'] else []
            for dangerous_verb in dangerous_verbs:
                if dangerous_verb != expert_verb:
                    dangerous_combo = (expert_instrument, dangerous_verb, expert_target)
                    dangerous_action_id = self._find_action_id(dangerous_combo)
                    if dangerous_action_id is not None:
                        negatives.append(dangerous_action_id)
        
        return negatives[:2]  # Limit to 2 negatives per frame
    
    def _generate_workflow_safety_negatives(self, expert_action_ids: List[int], 
                                          phase_name: str) -> List[int]:
        """
        Generate negatives for workflow safety (wrong phase timing)
        """
        
        negatives = []
        
        # Phase-inappropriate actions
        inappropriate_for_phase = {
            'preparation': [78, 79, 80, 81],  # Clipping actions (too early)
            'gallbladder-packaging': [78, 79, 32, 33, 34],  # Dissection/clipping (too late)
            'gallbladder-extraction': [78, 79, 1, 2, 32],  # Active surgery (should be finishing)
        }
        
        if phase_name in inappropriate_for_phase:
            phase_negatives = inappropriate_for_phase[phase_name]
            # Only add if not in expert actions
            for neg_id in phase_negatives:
                if neg_id not in expert_action_ids:
                    negatives.append(neg_id)
        
        return negatives[:2]  # Limit to 2 negatives per frame
    
    def evaluate_safety_guardrails(self, irl_trainer, test_loaders, logger) -> Dict[str, float]:
        """
        Evaluate IRL safety guardrails effectiveness on targeted components
        """
        
        logger.info("🛡️ EVALUATING SAFETY GUARDRAILS EFFECTIVENESS")
        logger.info("   Target Components: T (anatomical safety), IVT (combination safety)")
        
        results = {
            'anatomical_safety_score': 0.0,
            'combination_safety_score': 0.0,
            'workflow_safety_score': 0.0,
            'overall_safety_score': 0.0
        }
        
        total_tests = 0
        safety_preferences = {'anatomical': [], 'combination': [], 'workflow': []}
        
        for video_id, test_loader in test_loaders.items():
            for batch in test_loader:
                states = batch['current_state'].to(irl_trainer.device)
                expert_actions = batch['target_next_action'].to(irl_trainer.device)
                phases = batch['current_phase'].to(irl_trainer.device)
                
                for i in range(min(20, states.shape[0])):  # Sample for efficiency
                    state = states[i:i+1]
                    expert_action = expert_actions[i:i+1]
                    phase = phases[i:i+1]
                    
                    # Generate safety negatives
                    safety_negatives = self.generate_targeted_safety_negatives(
                        expert_action, phase
                    )
                    
                    # Get expert reward
                    expert_reward = irl_trainer.irl_system.compute_reward(
                        state.squeeze(), expert_action.squeeze(), phase.squeeze()
                    ).item()
                    
                    # Test each safety category
                    for safety_type, neg_tensor in safety_negatives.items():
                        neg_ids = torch.where(neg_tensor.squeeze() > 0.5)[0]
                        for neg_id in neg_ids:
                            neg_vector = torch.zeros(100, device=irl_trainer.device)
                            neg_vector[neg_id] = 1.0
                            
                            neg_reward = irl_trainer.irl_system.compute_reward(
                                state.squeeze(), neg_vector, phase.squeeze()
                            ).item()
                            
                            expert_preferred = expert_reward > neg_reward
                            
                            if 'anatomical' in safety_type:
                                safety_preferences['anatomical'].append(expert_preferred)
                            elif 'combination' in safety_type:
                                safety_preferences['combination'].append(expert_preferred)
                            elif 'workflow' in safety_type:
                                safety_preferences['workflow'].append(expert_preferred)
                            
                            total_tests += 1
        
        # Calculate safety scores
        for safety_type, preferences in safety_preferences.items():
            if preferences:
                score = np.mean(preferences)
                results[f'{safety_type}_safety_score'] = score
                
                logger.info(f"   🛡️ {safety_type.title()} Safety: {score:.1%} expert preference")
        
        # Overall safety score
        all_preferences = []
        for prefs in safety_preferences.values():
            all_preferences.extend(prefs)
        
        if all_preferences:
            overall_score = np.mean(all_preferences)
            results['overall_safety_score'] = overall_score
            
            logger.info(f"")
            logger.info(f"🏆 OVERALL SAFETY GUARDRAILS SCORE: {overall_score:.1%}")
            logger.info(f"   Total Safety Tests: {total_tests}")
            logger.info(f"   Safety Effectiveness: {self._interpret_safety_score(overall_score)}")
        
        return results
    
    def _interpret_safety_score(self, score: float) -> str:
        """Interpret safety guardrails effectiveness"""
        if score > 0.85:
            return "Excellent - Strong safety guardrails"
        elif score > 0.75:
            return "Good - Effective safety protection"
        elif score > 0.65:
            return "Acceptable - Basic safety awareness"
        else:
            return "Poor - Safety guardrails need improvement"
    
    def _find_action_id(self, combination: Tuple[str, str, str]) -> int:
        """Find action ID for instrument-verb-target combination"""
        instrument, verb, target = combination
        
        for action_id, action_str in self.actions.items():
            if 'null_verb' not in action_str:
                parts = action_str.split(',')
                if len(parts) == 3:
                    if parts[0] == instrument and parts[1] == verb and parts[2] == target:
                        return int(action_id)
        return None

def implement_safety_guardrails_irl(config, train_data, test_data, logger, il_model):
    """
    Main implementation function for safety guardrails IRL
    Focuses on high-opportunity components: T (52%) and IVT (33%)
    """
    
    logger.info("🛡️ IMPLEMENTING SAFETY GUARDRAILS IRL")
    logger.info("   Focus: High-opportunity components (T: 52%, IVT: 33%)")
    logger.info("   Goal: Provide safety guardrails without costly clinical mistakes")
    
    # Load configuration
    with open('data/labels.json', 'r') as f:
        labels_config = json.load(f)
    
    # Initialize safety guardrails system
    safety_system = SurgicalSafetyGuardrails(labels_config)
    
    # Train IRL with safety-focused negatives
    from training.irl_direct_trainer import train_direct_irl
    
    # Modify the negative generation to use safety-focused approach
    def safety_focused_negative_generation(expert_actions, current_phase):
        safety_negatives = safety_system.generate_targeted_safety_negatives(
            expert_actions, current_phase
        )
        
        # Combine all safety negatives
        combined_negatives = torch.zeros_like(expert_actions)
        for neg_type, neg_tensor in safety_negatives.items():
            combined_negatives += neg_tensor
        
        # Ensure binary (in case of overlaps)
        combined_negatives = (combined_negatives > 0.5).float()
        
        return combined_negatives
    
    # Train IRL with safety guardrails
    irl_results = train_direct_irl(
        config=config,
        train_data=train_data,
        test_data=test_data,
        logger=logger,
        il_model=il_model,
        custom_negative_generator=safety_focused_negative_generation
    )
    
    # Evaluate safety guardrails effectiveness
    from datasets.autoregressive_dataset import create_autoregressive_dataloaders
    
    train_loader, test_loaders = create_autoregressive_dataloaders(
        config=config['data'],
        train_data=None,  # Use for evaluation only
        test_data=test_data,
        batch_size=32,
        num_workers=4
    )
    
    safety_results = safety_system.evaluate_safety_guardrails(
        irl_results['irl_trainer'], test_loaders, logger
    )
    
    logger.info("🏆 SAFETY GUARDRAILS RESULTS:")
    logger.info(f"   Target Component Safety: Focus on anatomical precision")
    logger.info(f"   Combination Safety: Focus on contextual appropriateness")
    logger.info(f"   Clinical Value: Learn safety without patient risk")
    
    return {
        'status': 'success',
        'approach': 'safety_guardrails_irl',
        'target_components': ['anatomical_targets', 'surgical_combinations'],
        'safety_results': safety_results,
        'irl_system': irl_results['irl_trainer'],
        'clinical_value': 'learn_surgery_without_costly_mistakes',
        'high_opportunity_focus': {
            'target_component': '52% baseline → safety improvement',
            'combination_component': '33% baseline → contextual improvement'
        }
    }

if __name__ == "__main__":
    print("🛡️ SURGICAL SAFETY GUARDRAILS IRL")
    print("=" * 50)
    print("✅ Targets high-opportunity components (T: 52%, IVT: 33%)")
    print("✅ Provides safety guardrails without clinical mistakes")
    print("✅ Focuses on anatomical and combination safety")
    print("✅ Addresses real surgical training constraints")
    print("✅ Perfect narrative for medical AI/MICCAI")
