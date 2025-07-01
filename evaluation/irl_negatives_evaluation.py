#!/usr/bin/env python3
"""
Concrete Implementation Using Your Actual CholecT50 Action IDs
Ready-to-use evaluation code that works with your labels.json
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import json

class CholecT50PracticalEvaluator:
    """
    Practical evaluator using your actual CholecT50 action IDs
    Tests contextual appropriateness within the fixed 100-action vocabulary
    """
    
    def __init__(self):
        # Your actual labels from labels.json
        self.actions = {
            "0": "grasper,dissect,cystic_plate", "1": "grasper,dissect,gallbladder", "2": "grasper,dissect,omentum", 
            "3": "grasper,grasp,cystic_artery", "4": "grasper,grasp,cystic_duct", "5": "grasper,grasp,cystic_pedicle", 
            "6": "grasper,grasp,cystic_plate", "7": "grasper,grasp,gallbladder", "8": "grasper,grasp,gut", "9": "grasper,grasp,liver", 
            "10": "grasper,grasp,omentum", "11": "grasper,grasp,peritoneum", "12": "grasper,grasp,specimen_bag", "13": "grasper,pack,gallbladder", 
            "14": "grasper,retract,cystic_duct", "15": "grasper,retract,cystic_pedicle", "16": "grasper,retract,cystic_plate", 
            "17": "grasper,retract,gallbladder", "18": "grasper,retract,gut", "19": "grasper,retract,liver", "20": "grasper,retract,omentum", "21": "grasper,retract,peritoneum", 
            "22": "bipolar,coagulate,abdominal_wall_cavity", "23": "bipolar,coagulate,blood_vessel", "24": "bipolar,coagulate,cystic_artery", 
            "25": "bipolar,coagulate,cystic_duct", "26": "bipolar,coagulate,cystic_pedicle", "27": "bipolar,coagulate,cystic_plate", "28": "bipolar,coagulate,gallbladder", 
            "29": "bipolar,coagulate,liver", "30": "bipolar,coagulate,omentum", "31": "bipolar,coagulate,peritoneum", "32": "bipolar,dissect,adhesion", 
            "33": "bipolar,dissect,cystic_artery", "34": "bipolar,dissect,cystic_duct", "35": "bipolar,dissect,cystic_plate", 
            "36": "bipolar,dissect,gallbladder", "37": "bipolar,dissect,omentum", "38": "bipolar,grasp,cystic_plate", "39": "bipolar,grasp,liver", 
            "40": "bipolar,grasp,specimen_bag", "41": "bipolar,retract,cystic_duct", "42": "bipolar,retract,cystic_pedicle", "43": "bipolar,retract,gallbladder", 
            "44": "bipolar,retract,liver", "45": "bipolar,retract,omentum", "46": "hook,coagulate,blood_vessel", "47": "hook,coagulate,cystic_artery", 
            "48": "hook,coagulate,cystic_duct", "49": "hook,coagulate,cystic_pedicle", "50": "hook,coagulate,cystic_plate", "51": "hook,coagulate,gallbladder", 
            "52": "hook,coagulate,liver", "53": "hook,coagulate,omentum", "54": "hook,cut,blood_vessel", "55": "hook,cut,peritoneum", "56": "hook,dissect,blood_vessel", 
            "57": "hook,dissect,cystic_artery", "58": "hook,dissect,cystic_duct", "59": "hook,dissect,cystic_plate", "60": "hook,dissect,gallbladder", "61": "hook,dissect,omentum", 
            "62": "hook,dissect,peritoneum", "63": "hook,retract,gallbladder", "64": "hook,retract,liver", "65": "scissors,coagulate,omentum", "66": "scissors,cut,adhesion", 
            "67": "scissors,cut,blood_vessel", "68": "scissors,cut,cystic_artery", "69": "scissors,cut,cystic_duct", "70": "scissors,cut,cystic_plate", "71": "scissors,cut,liver", 
            "72": "scissors,cut,omentum", "73": "scissors,cut,peritoneum", "74": "scissors,dissect,cystic_plate", "75": "scissors,dissect,gallbladder", "76": "scissors,dissect,omentum", 
            "77": "clipper,clip,blood_vessel", "78": "clipper,clip,cystic_artery", "79": "clipper,clip,cystic_duct", "80": "clipper,clip,cystic_pedicle", "81": "clipper,clip,cystic_plate", 
            "82": "irrigator,aspirate,fluid", "83": "irrigator,dissect,cystic_duct", "84": "irrigator,dissect,cystic_pedicle", "85": "irrigator,dissect,cystic_plate", 
            "86": "irrigator,dissect,gallbladder", "87": "irrigator,dissect,omentum", "88": "irrigator,irrigate,abdominal_wall_cavity", "89": "irrigator,irrigate,cystic_pedicle", 
            "90": "irrigator,irrigate,liver", "91": "irrigator,retract,gallbladder", "92": "irrigator,retract,liver", "93": "irrigator,retract,omentum", 
            "94": "grasper,null_verb,null_target", "95": "bipolar,null_verb,null_target", "96": "hook,null_verb,null_target", 
            "97": "scissors,null_verb,null_target", "98": "clipper,null_verb,null_target", "99": "irrigator,null_verb,null_target"
        }
        
        self.phases = {
            "0": "preparation", "1": "carlot-triangle-dissection", "2": "clipping-and-cutting", 
            "3": "gallbladder-dissection", "4": "gallbladder-packaging", 
            "5": "cleaning-and-coagulation", "6": "gallbladder-extraction"
        }
        
        # Build contextual negative mappings using actual action IDs
        self._build_negative_mappings()
    
    def _build_negative_mappings(self):
        """Build mappings for contextual negatives using actual action IDs"""
        
        # Phase-inappropriate action mappings
        self.phase_negatives = {
            # In preparation phase (0), clipping actions are inappropriate (too early)
            'preparation_inappropriate': [78, 79, 80, 81, 77],  # All clipping actions
            
            # In clipping phase (2), packaging actions are inappropriate (too early)
            'clipping_inappropriate': [13, 12, 40],  # Packing/grasping specimen bag
            
            # In dissection phase (3), clipping actions are inappropriate (too late)
            'dissection_inappropriate': [78, 79, 80, 81, 77],  # All clipping actions
            
            # In packaging phase (4), dissection/clipping actions are inappropriate
            'packaging_inappropriate': [0, 1, 2, 32, 33, 34, 35, 36, 37, 78, 79, 80, 81],
            
            # In extraction phase (6), most surgical actions are inappropriate
            'extraction_inappropriate': [78, 79, 80, 81, 24, 25, 26, 27, 28, 29]
        }
        
        # Anatomically dangerous alternatives
        self.anatomical_negatives = {
            # When expert clips cystic artery (78), generic blood vessel (77) is less precise
            78: [77, 23],  # cystic_artery -> blood_vessel, bipolar coagulate blood_vessel
            
            # When expert grasps gallbladder (7), grasping liver (9) is dangerous
            7: [9, 19],    # gallbladder -> liver (dangerous)
            
            # When expert clips cystic duct (79), generic blood vessel is wrong
            79: [77, 67],  # cystic_duct -> blood_vessel, scissors cut blood_vessel
            
            # When expert grasps cystic artery (3), liver is wrong anatomy
            3: [9, 39],    # cystic_artery -> liver
        }
        
        # Technique alternatives (same target, different technique)
        self.technique_negatives = {
            # Clipping artery (78) vs cutting artery (68) - cutting arteries is more dangerous
            78: [68, 67],  # clip cystic_artery -> cut cystic_artery, cut blood_vessel
            
            # Clipping duct (79) vs cutting duct (69)
            79: [69],      # clip cystic_duct -> cut cystic_duct
            
            # Grasping vs retracting (different handling techniques)
            7: [17, 43],   # grasp gallbladder -> retract gallbladder
            3: [24],       # grasp cystic_artery -> coagulate cystic_artery
        }
        
        # Null actions (doing nothing when action is needed)
        self.null_actions = [94, 95, 96, 97, 98, 99]
    
    def generate_test_cases(self, expert_action_id: int, phase_id: int) -> Dict[str, List[int]]:
        """
        Generate contextual negatives for a specific expert action and phase
        
        Args:
            expert_action_id: Action ID (0-99) from your vocabulary
            phase_id: Phase ID (0-6)
            
        Returns:
            Dictionary of negative types with action ID lists
        """
        
        negatives = {
            'phase_inappropriate': [],
            'anatomically_dangerous': [],
            'technique_suboptimal': [],
            'null_actions': []
        }
        
        # 1. Phase-inappropriate actions
        phase_name = self.phases[str(phase_id)]
        if phase_name == 'preparation' and expert_action_id not in self.phase_negatives['preparation_inappropriate']:
            negatives['phase_inappropriate'].extend(self.phase_negatives['preparation_inappropriate'][:3])
        elif phase_name == 'clipping-and-cutting' and expert_action_id not in self.phase_negatives['clipping_inappropriate']:
            negatives['phase_inappropriate'].extend(self.phase_negatives['clipping_inappropriate'])
        elif phase_name == 'gallbladder-dissection' and expert_action_id not in self.phase_negatives['dissection_inappropriate']:
            negatives['phase_inappropriate'].extend(self.phase_negatives['dissection_inappropriate'][:2])
        elif phase_name == 'gallbladder-packaging' and expert_action_id not in self.phase_negatives['packaging_inappropriate']:
            negatives['phase_inappropriate'].extend(self.phase_negatives['packaging_inappropriate'][:3])
        
        # 2. Anatomically dangerous alternatives
        if expert_action_id in self.anatomical_negatives:
            negatives['anatomically_dangerous'].extend(self.anatomical_negatives[expert_action_id])
        
        # 3. Technique alternatives
        if expert_action_id in self.technique_negatives:
            negatives['technique_suboptimal'].extend(self.technique_negatives[expert_action_id])
        
        # 4. Null actions (if expert action is not null)
        if expert_action_id < 94:  # Expert action is not null
            negatives['null_actions'].extend(self.null_actions[:2])
        
        return negatives
    
    def evaluate_contextual_understanding(self, irl_trainer, test_loaders: Dict, 
                                        max_tests_per_video: int = 50) -> Dict[str, any]:
        """
        Evaluate IRL's contextual understanding using your actual data
        
        Args:
            irl_trainer: Your trained IRL trainer
            test_loaders: Your test DataLoaders
            max_tests_per_video: Maximum tests per video for efficiency
            
        Returns:
            Evaluation results
        """
        
        print("🔬 Evaluating Contextual Understanding with Actual CholecT50 Actions")
        
        results = {
            'phase_appropriateness': [],
            'anatomical_safety': [],
            'technique_preference': [],
            'action_vs_inaction': []
        }
        
        total_tests = 0
        
        for video_id, test_loader in test_loaders.items():
            video_tests = 0
            
            for batch in test_loader:
                if video_tests >= max_tests_per_video:
                    break
                    
                states = batch['current_state'].to(irl_trainer.device)
                expert_actions = batch['target_next_action'].to(irl_trainer.device)
                phases = batch['current_phase'].to(irl_trainer.device)
                
                for i in range(min(10, states.shape[0])):
                    if video_tests >= max_tests_per_video:
                        break
                        
                    state = states[i:i+1]
                    expert_action = expert_actions[i:i+1]
                    phase = phases[i:i+1]
                    
                    # Get expert action ID and phase ID
                    expert_action_id = torch.argmax(expert_action).item()
                    phase_id = torch.argmax(phase).item()
                    
                    # Skip null actions as experts (not meaningful to test)
                    if expert_action_id >= 94:
                        continue
                    
                    # Generate contextual negatives
                    negatives = self.generate_test_cases(expert_action_id, phase_id)
                    
                    # Get expert reward
                    expert_reward = irl_trainer.irl_system.compute_reward(
                        state.squeeze(), expert_action.squeeze(), phase.squeeze()
                    ).item()
                    
                    # Test each type of negative
                    for neg_type, neg_action_ids in negatives.items():
                        for neg_action_id in neg_action_ids[:2]:  # Test max 2 per type
                            # Create negative action vector
                            neg_action_vector = torch.zeros(100, device=irl_trainer.device)
                            neg_action_vector[neg_action_id] = 1.0
                            
                            # Get negative reward
                            neg_reward = irl_trainer.irl_system.compute_reward(
                                state.squeeze(), neg_action_vector, phase.squeeze()
                            ).item()
                            
                            # Test if expert is preferred
                            expert_preferred = expert_reward > neg_reward
                            
                            # Store result by category
                            if neg_type == 'phase_inappropriate':
                                results['phase_appropriateness'].append(expert_preferred)
                            elif neg_type == 'anatomically_dangerous':
                                results['anatomical_safety'].append(expert_preferred)
                            elif neg_type == 'technique_suboptimal':
                                results['technique_preference'].append(expert_preferred)
                            elif neg_type == 'null_actions':
                                results['action_vs_inaction'].append(expert_preferred)
                            
                            total_tests += 1
                            video_tests += 1
        
        # Calculate scores
        evaluation_scores = {}
        for category, preferences in results.items():
            if preferences:
                score = np.mean(preferences)
                evaluation_scores[category] = {
                    'score': score,
                    'num_tests': len(preferences),
                    'percentage': f"{score:.1%}",
                    'interpretation': self._interpret_score(score)
                }
        
        overall_score = np.mean([scores['score'] for scores in evaluation_scores.values()])
        
        # Print results
        print("\n🎯 CONTEXTUAL UNDERSTANDING RESULTS:")
        print(f"   Total Tests Conducted: {total_tests}")
        for category, scores in evaluation_scores.items():
            print(f"   {category.replace('_', ' ').title()}: {scores['percentage']} ({scores['num_tests']} tests) - {scores['interpretation']}")
        
        print(f"\n🏆 Overall Contextual Score: {overall_score:.1%}")
        print(f"   Assessment: {self._interpret_score(overall_score)}")
        
        return {
            'detailed_scores': evaluation_scores,
            'overall_contextual_score': overall_score,
            'total_tests': total_tests,
            'demonstrates_contextual_understanding': overall_score > 0.6,
            'paper_claims': self._generate_paper_claims(evaluation_scores, overall_score)
        }
    
    def _interpret_score(self, score: float) -> str:
        """Interpret evaluation score"""
        if score > 0.8:
            return "Excellent"
        elif score > 0.7:
            return "Good"
        elif score > 0.6:
            return "Acceptable"
        else:
            return "Poor"
    
    def _generate_paper_claims(self, scores: Dict, overall_score: float) -> List[str]:
        """Generate paper-ready claims based on results"""
        
        claims = []
        
        if overall_score > 0.7:
            claims.append(f"Model demonstrates contextual surgical understanding with {overall_score:.1%} overall accuracy")
        
        for category, score_info in scores.items():
            if score_info['score'] > 0.75:
                claims.append(f"Shows {category.replace('_', ' ')} with {score_info['percentage']} accuracy")
        
        if scores.get('anatomical_safety', {}).get('score', 0) > 0.7:
            claims.append("Demonstrates anatomical safety awareness in surgical decision-making")
        
        if scores.get('phase_appropriateness', {}).get('score', 0) > 0.75:
            claims.append("Exhibits surgical workflow intelligence and timing understanding")
        
        return claims
    
    def demonstrate_with_examples(self):
        """Demonstrate the evaluation with concrete examples"""
        
        print("📋 DEMONSTRATION: Contextual Negatives with Actual Action IDs")
        print("=" * 60)
        
        test_cases = [
            {
                'name': 'Clipping Cystic Artery in Clipping Phase',
                'expert_action_id': 78,  # "clipper,clip,cystic_artery"
                'phase_id': 2,           # "clipping-and-cutting"
            },
            {
                'name': 'Grasping Gallbladder in Dissection',
                'expert_action_id': 7,   # "grasper,grasp,gallbladder"
                'phase_id': 3,           # "gallbladder-dissection"
            },
            {
                'name': 'Packing Gallbladder in Packaging Phase',
                'expert_action_id': 13,  # "grasper,pack,gallbladder"
                'phase_id': 4,           # "gallbladder-packaging"
            }
        ]
        
        for case in test_cases:
            print(f"\n🔍 {case['name']}")
            print(f"   Expert Action: {self.actions[str(case['expert_action_id'])]}")
            print(f"   Phase: {self.phases[str(case['phase_id'])]}")
            
            negatives = self.generate_test_cases(case['expert_action_id'], case['phase_id'])
            
            for neg_type, neg_ids in negatives.items():
                if neg_ids:
                    print(f"   \n   🚨 {neg_type.replace('_', ' ').title()}:")
                    for neg_id in neg_ids[:2]:  # Show first 2
                        print(f"      • Action {neg_id}: {self.actions[str(neg_id)]}")


# INTEGRATION WITH YOUR EXPERIMENT RUNNER

def run_practical_contextual_evaluation(irl_trainer, test_loaders, logger):
    """
    Drop-in replacement for your evaluation that works with fixed vocabulary
    
    Args:
        irl_trainer: Your trained IRL trainer
        test_loaders: Your test DataLoaders  
        logger: Your logger
        
    Returns:
        Evaluation results compatible with your framework
    """
    
    logger.info("🔬 Running Practical Contextual Evaluation")
    
    evaluator = CholecT50PracticalEvaluator()
    results = evaluator.evaluate_contextual_understanding(irl_trainer, test_loaders)
    
    # Format for integration with your existing framework
    formatted_results = {
        'contextual_evaluation': results,
        'overall_contextual_score': results['overall_contextual_score'],
        'demonstrates_sophistication': results['demonstrates_contextual_understanding'],
        'evaluation_type': 'contextual_appropriateness_within_fixed_vocabulary',
        'paper_ready_claims': results['paper_claims'],
        'clinical_relevance': {
            'phase_awareness': results['detailed_scores'].get('phase_appropriateness', {}).get('score', 0) > 0.7,
            'anatomical_safety': results['detailed_scores'].get('anatomical_safety', {}).get('score', 0) > 0.7,
            'technique_preference': results['detailed_scores'].get('technique_preference', {}).get('score', 0) > 0.7
        }
    }
    
    return formatted_results


if __name__ == "__main__":
    print("🎯 PRACTICAL CONTEXTUAL EVALUATION")
    print("   Using actual CholecT50 action IDs")
    print("=" * 50)
    
    evaluator = CholecT50PracticalEvaluator()
    evaluator.demonstrate_with_examples()
    
    print(f"\n✅ Ready for integration with your IRL evaluation pipeline!")
    print(f"✅ Tests contextual appropriateness within fixed 100-action vocabulary")
    print(f"✅ Generates paper-ready claims about surgical understanding")
