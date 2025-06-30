#!/usr/bin/env python3
"""
Advanced Negative Example Generator for CholecT50 Surgical Videos
Generates realistic but suboptimal negatives using domain knowledge
"""

import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Set
import json
from collections import defaultdict
from tqdm import tqdm

class CholecT50NegativeGenerator:
    """
    Domain-aware negative example generator for surgical action recognition
    Uses the actual CholecT50 structure: 6 instruments × 9 verbs × 14 targets
    """
    
    def __init__(self, labels_config: Dict):
        """Initialize with CholecT50 label structure"""
        self.phases = labels_config['phase']
        self.actions = labels_config['action'] 
        self.instruments = labels_config['instrument']
        self.verbs = labels_config['verb']
        self.targets = labels_config['target']
        
        # Build structured mappings
        self._build_action_structure()
        self._build_phase_constraints()
        self._build_anatomical_relationships()
        self._build_instrument_capabilities()
        
    def _build_action_structure(self):
        """Parse action triplets into structured format"""
        self.action_triplets = {}
        self.triplet_to_id = {}
        
        for action_id, action_str in self.actions.items():
            action_id = int(action_id)
            if 'null_verb' in action_str:
                # Handle null actions (instrument only)
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
                    self.triplet_to_id[(instrument, verb, target)] = action_id
    
    def _build_phase_constraints(self):
        """Define which actions are appropriate for each phase"""
        self.phase_actions = {
            'preparation': {
                'primary_instruments': ['grasper'],
                'primary_verbs': ['grasp', 'retract'],
                'primary_targets': ['peritoneum', 'omentum']
            },
            'carlot-triangle-dissection': {
                'primary_instruments': ['grasper', 'bipolar', 'hook'],
                'primary_verbs': ['dissect', 'grasp', 'retract', 'coagulate'],
                'primary_targets': ['cystic_plate', 'cystic_artery', 'cystic_duct', 'cystic_pedicle']
            },
            'clipping-and-cutting': {
                'primary_instruments': ['clipper', 'scissors', 'grasper'],
                'primary_verbs': ['clip', 'cut', 'grasp', 'retract'],
                'primary_targets': ['cystic_artery', 'cystic_duct', 'cystic_pedicle', 'blood_vessel']
            },
            'gallbladder-dissection': {
                'primary_instruments': ['grasper', 'hook', 'bipolar'],
                'primary_verbs': ['dissect', 'grasp', 'retract', 'coagulate'],
                'primary_targets': ['gallbladder', 'liver', 'cystic_plate']
            },
            'gallbladder-packaging': {
                'primary_instruments': ['grasper'],
                'primary_verbs': ['pack', 'grasp'],
                'primary_targets': ['gallbladder', 'specimen_bag']
            },
            'cleaning-and-coagulation': {
                'primary_instruments': ['bipolar', 'irrigator', 'hook'],
                'primary_verbs': ['coagulate', 'irrigate', 'aspirate'],
                'primary_targets': ['liver', 'blood_vessel', 'abdominal_wall_cavity', 'fluid']
            },
            'gallbladder-extraction': {
                'primary_instruments': ['grasper'],
                'primary_verbs': ['retract', 'grasp'],
                'primary_targets': ['gallbladder', 'specimen_bag']
            }
        }
    
    def _build_anatomical_relationships(self):
        """Define anatomical proximity and confusion potential"""
        self.anatomical_groups = {
            'cystic_structures': ['cystic_artery', 'cystic_duct', 'cystic_plate', 'cystic_pedicle'],
            'vessels': ['cystic_artery', 'blood_vessel'],
            'soft_tissue': ['gallbladder', 'liver', 'omentum', 'peritoneum'],
            'spaces': ['abdominal_wall_cavity'],
            'other': ['gut', 'specimen_bag', 'fluid', 'adhesion']
        }
        
        # Define dangerous/safe target confusions
        self.dangerous_confusions = {
            'cystic_artery': ['blood_vessel'],  # Could clip wrong vessel
            'cystic_duct': ['blood_vessel'],     # Could clip wrong structure
            'gallbladder': ['liver'],           # Could damage liver
        }
        
        self.safe_confusions = {
            'omentum': ['peritoneum'],          # Similar tissues
            'cystic_plate': ['cystic_pedicle'], # Adjacent structures
        }
    
    def _build_instrument_capabilities(self):
        """Define what each instrument can/cannot do"""
        self.instrument_verbs = {
            'grasper': ['grasp', 'retract', 'dissect', 'pack'],
            'bipolar': ['coagulate', 'dissect', 'grasp', 'retract'],
            'hook': ['coagulate', 'dissect', 'retract', 'cut'],
            'scissors': ['cut', 'dissect', 'coagulate'],
            'clipper': ['clip'],
            'irrigator': ['irrigate', 'aspirate', 'dissect', 'retract']
        }
        
        # Impossible combinations (major mistakes)
        self.impossible_combinations = [
            ('grasper', 'clip'),      # Graspers can't clip
            ('grasper', 'irrigate'),  # Graspers can't irrigate
            ('clipper', 'irrigate'),  # Clippers can't irrigate
            ('clipper', 'dissect'),   # Clippers don't dissect
            ('irrigator', 'clip'),    # Irrigators can't clip
            ('scissors', 'irrigate'), # Scissors can't irrigate
        ]
    
    def generate_realistic_negatives(self, expert_actions: torch.Tensor, 
                                   current_phase: str = None) -> torch.Tensor:
        """
        Generate sophisticated negative examples using surgical domain knowledge
        
        Args:
            expert_actions: [batch_size, 100] binary tensor of expert actions
            current_phase:  
            
        Returns:
            negative_actions: [batch_size, 100] binary tensor of negative examples
        """
        batch_size = expert_actions.shape[0]
        device = expert_actions.device
        
        negative_actions = torch.zeros_like(expert_actions)
        
        for i in tqdm(range(batch_size), desc="Generating Negatives"):
            expert_frame = expert_actions[i]
            expert_action_ids = torch.where(expert_frame > 0.5)[0].cpu().numpy()

            current_phase_str = self._parse_phase_for_frame(current_phase, i)
            
            frame_negatives = []
            
            # Strategy distribution (balanced difficulty)
            strategies = [
                (0.25, self._temporal_negatives),      # 25% wrong phase
                (0.20, self._instrument_confusion),    # 20% wrong instrument  
                (0.20, self._target_confusion),        # 20% wrong target
                (0.15, self._impossible_actions),      # 15% impossible combinations
                (0.10, self._sparsity_negatives),      # 10% wrong number of actions
                (0.10, self._subtle_timing_errors)     # 10% subtle errors
            ]
            
            for prob, strategy_fn in strategies:
                if random.random() < prob:
                    negatives = strategy_fn(expert_action_ids, 
                                            current_phase_str)
                    frame_negatives.extend(negatives[:2])  # Max 2 per strategy
            
            # Ensure 1-3 negatives per frame (matching expert distribution)
            if len(frame_negatives) == 0:
                frame_negatives = self._fallback_negatives(expert_action_ids)
            frame_negatives = frame_negatives[:3]  # Cap at 3
            
            # Set negative actions
            for neg_id in frame_negatives:
                if 0 <= neg_id < 100:
                    negative_actions[i, neg_id] = 1.0
                    
        return negative_actions.to(device)
    
    def _temporal_negatives(self, expert_action_ids: List[int], 
                          current_phase_str: str) -> List[int]:
        """Actions from wrong surgical phases - medium difficulty"""
        negatives = []
        
        if not current_phase_str or current_phase_str not in self.phase_actions:
            return []
            
        wrong_phases = [p for p in self.phase_actions.keys() if p != current_phase_str]
        
        for action_id in expert_action_ids:
            if action_id not in self.action_triplets:
                continue
                
            action = self.action_triplets[action_id]
            instrument = action['instrument']
            verb = action['verb']
            target = action['target']
            
            # Find actions that would be wrong for current phase
            for wrong_phase in wrong_phases[:2]:  # Check 2 other phases
                wrong_phase_info = self.phase_actions[wrong_phase]
                
                # Action makes sense in wrong phase but not current phase
                if (instrument in wrong_phase_info['primary_instruments'] and
                    verb in wrong_phase_info['primary_verbs'] and
                    target in wrong_phase_info['primary_targets']):
                    
                    # But doesn't fit current phase well
                    current_phase_info = self.phase_actions[current_phase_str]
                    if (instrument not in current_phase_info['primary_instruments'] or
                        verb not in current_phase_info['primary_verbs'] or
                        target not in current_phase_info['primary_targets']):
                        
                        negatives.append(action_id)
                        break
        
        return negatives
    
    def _instrument_confusion(self, expert_action_ids: List[int], 
                            current_phase_str: str) -> List[int]:
        """Wrong instruments for right verb+target - easy/medium difficulty"""
        negatives = []
        
        for action_id in expert_action_ids:
            if action_id not in self.action_triplets:
                continue
                
            action = self.action_triplets[action_id]
            correct_instrument = action['instrument']
            verb = action['verb']
            target = action['target']
            
            if verb == 'null':
                continue
            
            # Find wrong but plausible instruments
            for wrong_instrument in self.instruments.values():
                if wrong_instrument == correct_instrument:
                    continue
                    
                # Check if combination is plausible but suboptimal
                if verb in self.instrument_verbs.get(wrong_instrument, []):
                    # Construct negative triplet
                    if (wrong_instrument, verb, target) in self.triplet_to_id:
                        neg_id = self.triplet_to_id[(wrong_instrument, verb, target)]
                        negatives.append(neg_id)
                        break
                        
        return negatives
    
    def _target_confusion(self, expert_action_ids: List[int], 
                         current_phase_str: str) -> List[int]:
        """Wrong targets for right instrument+verb - medium/hard difficulty"""
        negatives = []
        
        for action_id in expert_action_ids:
            if action_id not in self.action_triplets:
                continue
                
            action = self.action_triplets[action_id]
            instrument = action['instrument']
            verb = action['verb']
            correct_target = action['target']
            
            if verb == 'null':
                continue
            
            # Priority: Dangerous confusions (harder negatives)
            if correct_target in self.dangerous_confusions:
                for dangerous_target in self.dangerous_confusions[correct_target]:
                    if (instrument, verb, dangerous_target) in self.triplet_to_id:
                        neg_id = self.triplet_to_id[(instrument, verb, dangerous_target)]
                        negatives.append(neg_id)
                        break
            
            # Fallback: Safe confusions (easier negatives)
            if not negatives and correct_target in self.safe_confusions:
                for safe_target in self.safe_confusions[correct_target]:
                    if (instrument, verb, safe_target) in self.triplet_to_id:
                        neg_id = self.triplet_to_id[(instrument, verb, safe_target)]
                        negatives.append(neg_id)
                        break
                        
        return negatives

    def _parse_phase_for_frame(self, phase_data, frame_idx):
        phase_vector = phase_data[frame_idx]  # [7] one-hot vector
        phase_id = torch.argmax(phase_vector).item()  # Get phase ID
        phase_name = self.phases.get(str(phase_id))   # Convert to string
        return phase_name
    
    def _impossible_actions(self, expert_action_ids: List[int], 
                           current_phase_str: str) -> List[int]:
        """Impossible instrument-verb combinations - easy difficulty"""
        negatives = []
        
        for action_id in expert_action_ids:
            if action_id not in self.action_triplets:
                continue
                
            action = self.action_triplets[action_id]
            correct_instrument = action['instrument']
            verb = action['verb']
            target = action['target']
            
            if verb == 'null':
                continue
            
            # Find impossible instrument for this verb
            for wrong_instrument in self.instruments.values():
                if (wrong_instrument, verb) in self.impossible_combinations:
                    # Create impossible action
                    if (wrong_instrument, verb, target) in self.triplet_to_id:
                        neg_id = self.triplet_to_id[(wrong_instrument, verb, target)]
                        negatives.append(neg_id)
                        break
                        
        return negatives
    
    def _sparsity_negatives(self, expert_action_ids: List[int], 
                           current_phase_str: str) -> List[int]:
        """Wrong number of simultaneous actions - medium difficulty"""
        negatives = []
        
        # If expert has multiple actions, create single action negative
        if len(expert_action_ids) > 1:
            # Pick one random expert action (under-annotation)
            negatives.append(random.choice(expert_action_ids))
        
        # If expert has few actions, add extra plausible action (over-annotation)
        elif len(expert_action_ids) <= 1:
            if current_phase_str and current_phase_str in self.phase_actions:
                phase_info = self.phase_actions[current_phase_str]
                
                # Add random plausible action for this phase
                for _ in range(random.randint(1, 2)):
                    instrument = random.choice(phase_info['primary_instruments'])
                    verb = random.choice(phase_info['primary_verbs'])
                    target = random.choice(phase_info['primary_targets'])
                    
                    if (instrument, verb, target) in self.triplet_to_id:
                        neg_id = self.triplet_to_id[(instrument, verb, target)]
                        if neg_id not in expert_action_ids:
                            negatives.append(neg_id)
                            break
                            
        return negatives
    
    def _subtle_timing_errors(self, expert_action_ids: List[int], 
                             current_phase_str: str) -> List[int]:
        """Subtly wrong timing (actions slightly early/late) - hard difficulty"""
        negatives = []
        
        phase_sequence = [
            'preparation', 'carlot-triangle-dissection', 'clipping-and-cutting',
            'gallbladder-dissection', 'gallbladder-packaging', 
            'cleaning-and-coagulation', 'gallbladder-extraction'
        ]
        
        if not current_phase_str or current_phase_str not in phase_sequence:
            return []
            
        current_idx = phase_sequence.index(current_phase_str)
        
        # Actions that belong to adjacent phases (subtle timing errors)
        adjacent_phases = []
        if current_idx > 0:
            adjacent_phases.append(phase_sequence[current_idx - 1])  # Previous phase
        if current_idx < len(phase_sequence) - 1:
            adjacent_phases.append(phase_sequence[current_idx + 1])  # Next phase
        
        for adj_phase in adjacent_phases:
            adj_phase_info = self.phase_actions[adj_phase]
            
            # Find actions that would fit adjacent phase but not current
            for instrument in adj_phase_info['primary_instruments'][:2]:
                for verb in adj_phase_info['primary_verbs'][:2]:
                    for target in adj_phase_info['primary_targets'][:2]:
                        if (instrument, verb, target) in self.triplet_to_id:
                            neg_id = self.triplet_to_id[(instrument, verb, target)]
                            if neg_id not in expert_action_ids:
                                negatives.append(neg_id)
                                if len(negatives) >= 2:
                                    return negatives
                                    
        return negatives
    
    def _fallback_negatives(self, expert_action_ids: List[int]) -> List[int]:
        """Simple fallback when other strategies fail"""
        # Random actions excluding expert actions
        all_actions = list(range(100))
        available_actions = [a for a in all_actions if a not in expert_action_ids]
        
        if available_actions:
            return random.sample(available_actions, 
                               min(2, len(available_actions)))
        return []

# Example usage and integration with existing code
def replace_negative_generation_in_irl_trainer():
    """
    Code to replace the existing _generate_realistic_negatives method
    in your DirectIRLTrainer class
    """
    
    # Load your labels configuration
    with open('labels.json', 'r') as f:
        labels_config = json.load(f)
    
    # Initialize the sophisticated negative generator
    negative_generator = CholecT50NegativeGenerator(labels_config)
    
    def enhanced_generate_realistic_negatives(self, expert_actions: torch.Tensor, 
                                            phases: torch.Tensor = None) -> torch.Tensor:
        """Enhanced negative generation for DirectIRLTrainer"""
        
        batch_size = expert_actions.shape[0]
        negatives = []
        
        for i in range(batch_size):
            # Get current phase if available
            current_phase_str = None
            if phases is not None and len(phases) > i:
                phase_id = torch.argmax(phases[i]).item()
                current_phase_str = list(labels_config['phase'].values())[phase_id]
            
            # Generate sophisticated negatives
            frame_expert = expert_actions[i:i+1]
            frame_negatives = negative_generator.generate_realistic_negatives(
                frame_expert, current_phase_str
            )
            
            negatives.append(frame_negatives.squeeze(0))
        
        return torch.stack(negatives).to(expert_actions.device)
    
    return enhanced_generate_realistic_negatives

# Example analysis of negative difficulty
def analyze_negative_difficulty():
    """Analyze the difficulty distribution of generated negatives"""
    
    # Load configuration
    with open('labels.json', 'r') as f:
        labels_config = json.load(f)
    
    generator = CholecT50NegativeGenerator(labels_config)
    
    # Example expert action: "clipper,clip,cystic_artery" during clipping phase
    expert_actions = torch.zeros(1, 100)
    expert_actions[0, 78] = 1.0  # clipper,clip,cystic_artery
    
    print("🔍 Negative Difficulty Analysis")
    print("=" * 50)
    
    # Generate negatives for different phases
    for phase in ['preparation', 'clipping-and-cutting', 'gallbladder-dissection']:
        print(f"\n📍 Phase: {phase}")
        negatives = generator.generate_realistic_negatives(expert_actions, phase)
        
        negative_ids = torch.where(negatives[0] > 0.5)[0].cpu().numpy()
        print(f"Generated {len(negative_ids)} negatives:")
        
        for neg_id in negative_ids:
            if neg_id in generator.action_triplets:
                action = generator.action_triplets[neg_id]
                print(f"  • {action['instrument']},{action['verb']},{action['target']}")
            else:
                print(f"  • Action ID {neg_id}")

if __name__ == "__main__":
    print("🏥 CholecT50 Surgical Negative Generator")
    print("=" * 50)
    print("✅ Domain-aware negative sampling")
    print("✅ Balanced difficulty progression") 
    print("✅ Respects surgical workflow")
    print("✅ Prevents training on trivial negatives")
    
    # Run analysis
    analyze_negative_difficulty()