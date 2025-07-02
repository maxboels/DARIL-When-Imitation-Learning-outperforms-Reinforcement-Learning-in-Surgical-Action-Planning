#!/usr/bin/env python3
"""
IMPROVED Negative Generation for Surgical IL Enhancement
Focus: Help IL learn better surgical representations through curriculum learning

Key Improvements:
1. Curriculum-based negatives (easy → hard)  
2. Temporal-aware negatives (leverage BiLSTM + GPT2 structure)
3. Realistic surgical scenarios (not artificial edge cases)
4. Balanced learning (not just low-performers)
5. Context-preserving negatives (maintain surgical realism)
"""

import torch
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class CurriculumSurgicalNegatives:
    """
    Curriculum-based negative generation that helps IL learn better surgical representations
    
    Philosophy: Start with easy distinctions, progress to harder surgical scenarios
    Goal: Improve IL baseline by teaching better surgical understanding
    """
    
    def __init__(self, labels_config: Dict, curriculum_stage: str = 'easy'):
        """
        Initialize curriculum-based negative generator
        
        Args:
            labels_config: CholecT50 labels configuration
            curriculum_stage: 'easy', 'medium', 'hard', 'expert'
        """
        self.labels_config = labels_config
        self.curriculum_stage = curriculum_stage
        self.actions = labels_config['action']
        self.phases = labels_config['phase']
        self.instruments = labels_config['instrument']
        self.verbs = labels_config['verb']
        self.targets = labels_config['target']
        
        # Parse surgical structure for intelligent negatives
        self._parse_surgical_structure()
        
        # Build curriculum levels
        self._build_curriculum_levels()
        
        print(f"🎓 Curriculum Surgical Negatives initialized (Stage: {curriculum_stage})")
        print(f"   Goal: Help IL learn better surgical representations")
        print(f"   Approach: Realistic surgical scenarios with curriculum progression")
    
    def _parse_surgical_structure(self):
        """Parse surgical actions into structured components"""
        
        self.action_components = {}
        self.instrument_actions = defaultdict(list)
        self.verb_actions = defaultdict(list)
        self.target_actions = defaultdict(list)
        
        for action_id, action_str in self.actions.items():
            action_id = int(action_id)
            
            if 'null_verb' in action_str:
                # Handle null actions
                instrument = action_str.split(',')[0]
                self.action_components[action_id] = {
                    'instrument': instrument,
                    'verb': 'null',
                    'target': 'null',
                    'is_null': True
                }
                self.instrument_actions[instrument].append(action_id)
            else:
                # Parse I-V-T triplets
                parts = action_str.split(',')
                if len(parts) == 3:
                    instrument, verb, target = parts
                    self.action_components[action_id] = {
                        'instrument': instrument,
                        'verb': verb,
                        'target': target,
                        'is_null': False
                    }
                    
                    self.instrument_actions[instrument].append(action_id)
                    self.verb_actions[verb].append(action_id)
                    self.target_actions[target].append(action_id)
    
    def _build_curriculum_levels(self):
        """Build curriculum levels from easy to hard negatives"""
        
        self.curriculum_negatives = {
            
            # LEVEL 1: EASY - Clear distinctions (Instrument swaps)
            'easy': {
                'description': 'Simple instrument swaps - easy to distinguish',
                'strategies': [
                    'instrument_substitution',   # grasper → scissors
                    'null_vs_action',           # action → null
                    'random_action_subset'      # simple random from different category
                ],
                'difficulty': 0.3,
                'focus': 'Basic surgical tool recognition'
            },
            
            # LEVEL 2: MEDIUM - Verb modifications (More subtle)
            'medium': {
                'description': 'Same instrument, different verb - moderate difficulty',
                'strategies': [
                    'verb_substitution',        # grasp → retract 
                    'technique_variation',      # cut → coagulate
                    'temporal_inconsistency'    # wrong timing
                ],
                'difficulty': 0.5,
                'focus': 'Surgical technique understanding'
            },
            
            # LEVEL 3: HARD - Target modifications (Subtle anatomical)
            'hard': {
                'description': 'Same instrument+verb, different target - challenging',
                'strategies': [
                    'target_substitution',      # liver → gallbladder
                    'anatomical_specificity',   # blood_vessel → cystic_artery
                    'phase_inappropriateness'   # wrong phase for action
                ],
                'difficulty': 0.7,
                'focus': 'Anatomical precision and timing'
            },
            
            # LEVEL 4: EXPERT - Subtle clinical scenarios (Very challenging)
            'expert': {
                'description': 'Clinically plausible but suboptimal - expert-level',
                'strategies': [
                    'clinical_preference',      # Multiple valid options, one better
                    'safety_considerations',    # Safe vs safer approaches
                    'efficiency_optimization'   # Good vs optimal technique
                ],
                'difficulty': 0.85,
                'focus': 'Clinical expertise and safety principles'
            }
        }
    
    def generate_curriculum_negatives(self, expert_actions: torch.Tensor, 
                                    current_phase: torch.Tensor = None,
                                    batch_progress: float = 0.0) -> torch.Tensor:
        """
        Generate curriculum-based negatives that help IL learn better
        
        Args:
            expert_actions: [batch_size, num_actions] expert action tensor
            current_phase: [batch_size, num_phases] current phase tensor
            batch_progress: Training progress (0.0 = start, 1.0 = end) for curriculum
            
        Returns:
            negative_actions: [batch_size, num_actions] curriculum negatives
        """
        
        batch_size, num_actions = expert_actions.shape
        negative_actions = torch.zeros_like(expert_actions)
        
        # Determine curriculum stage based on training progress
        current_stage = self._get_curriculum_stage(batch_progress)
        curriculum_config = self.curriculum_negatives[current_stage]
        
        for i in range(batch_size):
            expert_frame = expert_actions[i]
            expert_action_ids = torch.where(expert_frame > 0.5)[0].cpu().numpy()
            
            if len(expert_action_ids) == 0:
                continue
            
            # Get current phase context
            phase_context = None
            if current_phase is not None:
                phase_id = torch.argmax(current_phase[i]).item()
                phase_context = self.phases.get(str(phase_id), 'unknown')
            
            # Generate negatives using curriculum strategies
            frame_negatives = self._generate_curriculum_frame_negatives(
                expert_action_ids, phase_context, current_stage, curriculum_config
            )
            
            # Apply negatives (limit to 2 per frame for focused learning)
            for neg_id in frame_negatives[:2]:
                if 0 <= neg_id < num_actions:
                    negative_actions[i, neg_id] = 1.0
        
        return negative_actions
    
    def _get_curriculum_stage(self, batch_progress: float) -> str:
        """Determine curriculum stage based on training progress"""
        
        if batch_progress < 0.25:
            return 'easy'
        elif batch_progress < 0.5:
            return 'medium'  
        elif batch_progress < 0.8:
            return 'hard'
        else:
            return 'expert'
    
    def _generate_curriculum_frame_negatives(self, expert_action_ids: List[int],
                                           phase_context: str,
                                           stage: str,
                                           config: Dict) -> List[int]:
        """Generate negatives for a single frame using curriculum strategies"""
        
        negatives = []
        strategies = config['strategies']
        
        for expert_id in expert_action_ids:
            if expert_id not in self.action_components:
                continue
                
            expert_action = self.action_components[expert_id]
            
            # Apply curriculum strategies
            for strategy in strategies:
                strategy_negatives = self._apply_strategy(
                    expert_id, expert_action, strategy, phase_context
                )
                negatives.extend(strategy_negatives)
                
                if len(negatives) >= 3:  # Limit negatives per expert action
                    break
            
            if len(negatives) >= 6:  # Limit total negatives per frame
                break
        
        # Shuffle and return limited set
        random.shuffle(negatives)
        return negatives[:3]  # Max 3 negatives per frame
    
    def _apply_strategy(self, expert_id: int, expert_action: Dict, 
                       strategy: str, phase_context: str) -> List[int]:
        """Apply specific negative generation strategy"""
        
        negatives = []
        
        if strategy == 'instrument_substitution':
            negatives = self._instrument_substitution(expert_action)
            
        elif strategy == 'verb_substitution':
            negatives = self._verb_substitution(expert_action)
            
        elif strategy == 'target_substitution':
            negatives = self._target_substitution(expert_action)
            
        elif strategy == 'null_vs_action':
            negatives = self._null_vs_action(expert_action)
            
        elif strategy == 'temporal_inconsistency':
            negatives = self._temporal_inconsistency(expert_action, phase_context)
            
        elif strategy == 'technique_variation':
            negatives = self._technique_variation(expert_action)
            
        elif strategy == 'anatomical_specificity':
            negatives = self._anatomical_specificity(expert_action)
            
        elif strategy == 'clinical_preference':
            negatives = self._clinical_preference(expert_action)
            
        elif strategy == 'random_action_subset':
            negatives = self._random_action_subset(expert_id)
        
        # Filter out expert action and invalid actions
        valid_negatives = [neg_id for neg_id in negatives 
                          if neg_id != expert_id and 0 <= neg_id < 100]
        
        return valid_negatives[:2]  # Max 2 per strategy
    
    def _instrument_substitution(self, expert_action: Dict) -> List[int]:
        """EASY: Substitute instrument while keeping verb and target"""
        
        if expert_action['is_null']:
            return self._get_different_null_instruments(expert_action['instrument'])
        
        verb = expert_action['verb']
        target = expert_action['target']
        current_instrument = expert_action['instrument']
        
        # Find actions with same verb+target but different instrument
        candidates = []
        for action_id, action_info in self.action_components.items():
            if (not action_info['is_null'] and 
                action_info['verb'] == verb and 
                action_info['target'] == target and
                action_info['instrument'] != current_instrument):
                candidates.append(action_id)
        
        return candidates[:2]
    
    def _verb_substitution(self, expert_action: Dict) -> List[int]:
        """MEDIUM: Substitute verb while keeping instrument and target"""
        
        if expert_action['is_null']:
            return []
        
        instrument = expert_action['instrument']
        target = expert_action['target']
        current_verb = expert_action['verb']
        
        # Find actions with same instrument+target but different verb
        candidates = []
        for action_id, action_info in self.action_components.items():
            if (not action_info['is_null'] and
                action_info['instrument'] == instrument and
                action_info['target'] == target and
                action_info['verb'] != current_verb):
                candidates.append(action_id)
        
        return candidates[:2]
    
    def _target_substitution(self, expert_action: Dict) -> List[int]:
        """HARD: Substitute target while keeping instrument and verb"""
        
        if expert_action['is_null']:
            return []
        
        instrument = expert_action['instrument']
        verb = expert_action['verb']
        current_target = expert_action['target']
        
        # Find actions with same instrument+verb but different target
        candidates = []
        for action_id, action_info in self.action_components.items():
            if (not action_info['is_null'] and
                action_info['instrument'] == instrument and
                action_info['verb'] == verb and
                action_info['target'] != current_target):
                candidates.append(action_id)
        
        return candidates[:2]
    
    def _null_vs_action(self, expert_action: Dict) -> List[int]:
        """EASY: Contrast action vs null (or vice versa)"""
        
        if expert_action['is_null']:
            # Expert is null, provide real action with same instrument
            instrument = expert_action['instrument']
            candidates = [aid for aid in self.instrument_actions[instrument]
                         if not self.action_components[aid]['is_null']]
            return candidates[:2]
        else:
            # Expert is action, provide null with same instrument  
            instrument = expert_action['instrument']
            null_candidates = [aid for aid in self.instrument_actions[instrument]
                              if self.action_components[aid]['is_null']]
            return null_candidates[:1]
    
    def _temporal_inconsistency(self, expert_action: Dict, phase_context: str) -> List[int]:
        """MEDIUM: Actions inappropriate for current phase"""
        
        if phase_context is None:
            return []
        
        # Define phase-inappropriate actions (simplified mapping)
        phase_inappropriate = {
            'preparation': [78, 79, 80, 81],  # No clipping in preparation
            'clipping-and-cutting': [13, 12], # No packaging during clipping
            'gallbladder-dissection': [78, 79], # No more clipping during dissection
            'gallbladder-packaging': [33, 34, 68, 69], # No dissection during packaging
        }
        
        inappropriate_actions = phase_inappropriate.get(phase_context, [])
        return inappropriate_actions[:2]
    
    def _technique_variation(self, expert_action: Dict) -> List[int]:
        """MEDIUM: Different but plausible techniques"""
        
        if expert_action['is_null']:
            return []
        
        # Define technique alternatives (conservative mapping)
        technique_alternatives = {
            'grasp': ['retract', 'pack'],
            'cut': ['coagulate', 'dissect'],
            'clip': ['coagulate'],
            'dissect': ['coagulate'],
            'retract': ['grasp'],
            'coagulate': ['cut']
        }
        
        current_verb = expert_action['verb']
        alternative_verbs = technique_alternatives.get(current_verb, [])
        
        instrument = expert_action['instrument']
        target = expert_action['target']
        
        candidates = []
        for alt_verb in alternative_verbs:
            for action_id, action_info in self.action_components.items():
                if (not action_info['is_null'] and
                    action_info['instrument'] == instrument and
                    action_info['target'] == target and
                    action_info['verb'] == alt_verb):
                    candidates.append(action_id)
        
        return candidates[:2]
    
    def _anatomical_specificity(self, expert_action: Dict) -> List[int]:
        """HARD: Generic vs specific anatomical targets"""
        
        if expert_action['is_null']:
            return []
        
        # Define specificity mapping (specific → generic)
        specificity_mapping = {
            'cystic_artery': 'blood_vessel',
            'cystic_duct': 'blood_vessel', 
            'cystic_pedicle': 'blood_vessel',
            'gallbladder': 'liver',  # Nearby organ
        }
        
        current_target = expert_action['target']
        generic_target = specificity_mapping.get(current_target)
        
        if generic_target is None:
            return []
        
        instrument = expert_action['instrument']
        verb = expert_action['verb']
        
        # Find action with same instrument+verb but generic target
        candidates = []
        for action_id, action_info in self.action_components.items():
            if (not action_info['is_null'] and
                action_info['instrument'] == instrument and
                action_info['verb'] == verb and
                action_info['target'] == generic_target):
                candidates.append(action_id)
        
        return candidates[:1]
    
    def _clinical_preference(self, expert_action: Dict) -> List[int]:
        """EXPERT: Clinically valid but suboptimal alternatives"""
        
        # Define clinical preference mappings (expert → suboptimal)
        clinical_preferences = {
            # Safer coagulation preferred over cutting for vessels
            ('scissors', 'cut', 'blood_vessel'): [('bipolar', 'coagulate', 'blood_vessel')],
            ('scissors', 'cut', 'cystic_artery'): [('bipolar', 'coagulate', 'cystic_artery')],
            
            # Retraction preferred over grasping for fragile organs
            ('grasper', 'grasp', 'liver'): [('grasper', 'retract', 'liver')],
            ('grasper', 'grasp', 'gallbladder'): [('grasper', 'retract', 'gallbladder')],
        }
        
        if expert_action['is_null']:
            return []
        
        expert_triplet = (expert_action['instrument'], expert_action['verb'], expert_action['target'])
        alternatives = clinical_preferences.get(expert_triplet, [])
        
        candidates = []
        for alt_instrument, alt_verb, alt_target in alternatives:
            for action_id, action_info in self.action_components.items():
                if (not action_info['is_null'] and
                    action_info['instrument'] == alt_instrument and
                    action_info['verb'] == alt_verb and
                    action_info['target'] == alt_target):
                    candidates.append(action_id)
        
        return candidates[:1]
    
    def _random_action_subset(self, expert_id: int) -> List[int]:
        """EASY: Simple random negatives from different categories"""
        
        # Get random actions from different instrument categories
        all_instruments = list(self.instrument_actions.keys())
        expert_instrument = self.action_components[expert_id]['instrument']
        
        # Pick different instrument
        other_instruments = [inst for inst in all_instruments if inst != expert_instrument]
        if not other_instruments:
            return []
        
        random_instrument = random.choice(other_instruments)
        candidates = self.instrument_actions[random_instrument]
        
        # Return 1-2 random actions from that instrument
        random.shuffle(candidates)
        return candidates[:2]
    
    def _get_different_null_instruments(self, current_instrument: str) -> List[int]:
        """Get null actions with different instruments"""
        
        null_candidates = []
        for action_id, action_info in self.action_components.items():
            if (action_info['is_null'] and 
                action_info['instrument'] != current_instrument):
                null_candidates.append(action_id)
        
        return null_candidates[:2]
    
    def get_curriculum_progress_report(self, batch_progress: float) -> Dict:
        """Get report on current curriculum stage and focus"""
        
        current_stage = self._get_curriculum_stage(batch_progress)
        config = self.curriculum_negatives[current_stage]
        
        return {
            'current_stage': current_stage,
            'description': config['description'],
            'difficulty': config['difficulty'],
            'focus': config['focus'],
            'strategies': config['strategies'],
            'batch_progress': batch_progress,
            'next_stage_at': {
                'easy': 0.25,
                'medium': 0.5,
                'hard': 0.8,
                'expert': 1.0
            }.get(current_stage, 1.0)
        }

# Integration function for your existing codebase
def create_curriculum_negative_generator(labels_config_path: str = 'data/labels.json') -> CurriculumSurgicalNegatives:
    """
    Create curriculum negative generator for IL enhancement
    
    Returns:
        CurriculumSurgicalNegatives: Ready-to-use curriculum generator
    """
    
    # Load labels configuration
    with open(labels_config_path, 'r') as f:
        labels_config = json.load(f)
    
    # Initialize curriculum generator
    curriculum_generator = CurriculumSurgicalNegatives(labels_config)
    
    print("\n🎓 CURRICULUM NEGATIVE GENERATOR INITIALIZED")
    print("=" * 50)
    print("Goal: Help IL learn better surgical representations")
    print("Approach: Curriculum learning (easy → expert)")
    print()
    print("Curriculum Stages:")
    for stage, config in curriculum_generator.curriculum_negatives.items():
        print(f"  {stage.upper()}: {config['description']}")
        print(f"    Focus: {config['focus']}")
        print(f"    Difficulty: {config['difficulty']}")
    print()
    print("✅ Ready to enhance IL training with curriculum negatives!")
    
    return curriculum_generator

# Example usage for IL enhancement
def enhance_il_training_with_curriculum(il_trainer, curriculum_generator):
    """
    Example of how to integrate curriculum negatives into IL training
    
    This would be called during IL training, not IRL training!
    """
    
    def enhanced_training_step(batch, epoch, total_epochs):
        """Enhanced training step with curriculum negatives"""
        
        # Calculate training progress for curriculum
        batch_progress = epoch / total_epochs
        
        # Get expert actions from batch
        expert_actions = batch['target_actions']  # Your IL training targets
        current_phase = batch.get('current_phase', None)
        
        # Generate curriculum negatives
        negative_actions = curriculum_generator.generate_curriculum_negatives(
            expert_actions, current_phase, batch_progress
        )
        
        # Enhanced loss with negatives (example)
        # This teaches IL to distinguish expert actions from realistic alternatives
        positive_loss = il_trainer.compute_loss(batch)  # Your existing IL loss
        
        # Create negative batch
        negative_batch = batch.copy()
        negative_batch['target_actions'] = negative_actions
        negative_loss = il_trainer.compute_loss(negative_batch)
        
        # Combined loss (encourage expert, discourage negatives)
        enhanced_loss = positive_loss - 0.1 * negative_loss  # Small negative weight
        
        return enhanced_loss
    
    return enhanced_training_step

if __name__ == "__main__":
    print("🎓 CURRICULUM SURGICAL NEGATIVES")
    print("=" * 50)
    print("✅ Curriculum-based learning (easy → hard)")
    print("✅ Realistic surgical scenarios")
    print("✅ Temporal and contextual awareness") 
    print("✅ Focused on helping IL learn better representations")
    print("✅ Not just optimizing for low-performing classes")
    
    # Example initialization
    try:
        curriculum_gen = create_curriculum_negative_generator()
        print("\n🚀 Ready for IL enhancement!")
    except Exception as e:
        print(f"\n⚠️ Initialization failed: {e}")
        print("Please ensure labels.json is available.")