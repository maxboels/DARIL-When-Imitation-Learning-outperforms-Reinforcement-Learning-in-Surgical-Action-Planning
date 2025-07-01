# 🔬 Surgical Prior Extraction for Safety Guardrails IRL

Extract training data patterns to inform targeted negative generation for safety guardrails IRL training.

## 🎯 Purpose

This toolkit extracts prior knowledge from your CholecT50 training data to optimize negative generation for:

- **Target Component (T: 52% baseline)** → Anatomical safety guardrails
- **Combination Component (IVT: 33% baseline)** → Contextual appropriateness guardrails

## 🚀 Quick Start

### 1. Extract Priors (Run Once Before Training)

```bash
# Full extraction
python run_prior_extraction.py --config config_dgx_all_v8.yaml --output surgical_priors

# Quick test (2 videos only)
python run_prior_extraction.py --quick_test
```

### 2. Integration with IRL Training

```python
from prior_integration_example import PriorInformedNegativeGenerator

# Load extracted priors
generator = PriorInformedNegativeGenerator('surgical_priors')

# Generate targeted negatives
negatives = generator.generate_targeted_negatives(
    expert_actions, current_phase, validation_threshold=0.05
)

# Get targeting strategy
strategy = generator.get_component_targeting_strategy()
```

## 📊 What Gets Extracted

### 1. Action-Phase Co-occurrence Patterns
- **Purpose**: Validate negatives against training frequency
- **File**: `surgical_priors.json`
- **Use**: Filter negatives that appear >5% in training

### 2. Component Difficulty Analysis  
- **Purpose**: Identify high-opportunity improvement areas
- **File**: `component_difficulty_analysis.json`
- **Focus**: Rare targets and combinations (T: 52%, IVT: 33%)

### 3. Anatomical Safety Patterns
- **Purpose**: Generate anatomical safety negatives
- **File**: `safety_patterns.json`
- **Target**: Target component improvement (T: 52% → 65%+)

### 4. Combination Safety Patterns
- **Purpose**: Generate combination appropriateness negatives  
- **File**: `safety_patterns.json`
- **Target**: IVT component improvement (33% → 40%+)

### 5. Workflow Constraints
- **Purpose**: Generate phase-inappropriate negatives
- **File**: `workflow_constraints.json`
- **Use**: Phase timing safety guardrails

## 📁 Output Structure

```
surgical_priors/
├── surgical_priors.json                    # Main priors file
├── component_difficulty_analysis.json      # T & IVT component focus
├── safety_patterns.json                    # Anatomical & combination safety
├── workflow_constraints.json               # Phase timing constraints
├── component_difficulty.png                # Rare targets visualization
├── phase_action_frequency.png              # Phase-action patterns
└── safety_violations.png                   # Detected safety issues
```

## 🎯 Safety Guardrails Focus

### High-Opportunity Components

1. **Target Component (T: 52% baseline)**
   - **Challenge**: Anatomical precision
   - **Guardrails**: Specific vs generic anatomy
   - **Example**: Prefer "cystic_artery" over "blood_vessel"

2. **Combination Component (IVT: 33% baseline)**
   - **Challenge**: Contextual appropriateness  
   - **Guardrails**: Safe instrument-verb-target combinations
   - **Example**: Avoid "grasper,grasp,liver" (fragile organ)

### Negative Generation Strategy

- **Anatomical Safety**: Target dangerous anatomy alternatives
- **Combination Safety**: Target inappropriate IVT combinations  
- **Workflow Safety**: Target phase-inappropriate timing
- **Data Validation**: Filter negatives appearing >5% in training

## 🔧 Integration Example

```python
# In your IRL training script
def enhanced_irl_training(train_data, test_data):
    
    # 1. Extract priors (run once)
    extract_surgical_priors(config, output_dir='surgical_priors')
    
    # 2. Load priors for training
    generator = PriorInformedNegativeGenerator('surgical_priors')
    
    # 3. Replace negative generation in IRL trainer
    def targeted_negative_generation(expert_actions, current_phase):
        return generator.generate_targeted_negatives(
            expert_actions, current_phase, validation_threshold=0.05
        )
    
    irl_trainer._generate_realistic_negatives = targeted_negative_generation
    
    # 4. Train with safety guardrails
    irl_trainer.train(train_data, test_data)
```

## 📈 Expected Impact

- **Target Component**: 52% → 60%+ (anatomical safety guardrails)
- **Combination Component**: 33% → 40%+ (contextual appropriateness) 
- **Safety**: 80%+ preference for expert over dangerous alternatives
- **Clinical Value**: Learn surgery without costly mistakes

## 🛠️ Customization

### Adjust Validation Thresholds

```python
# Strict validation (2% threshold)
negatives = generator.generate_targeted_negatives(
    expert_actions, current_phase, validation_threshold=0.02
)

# Relaxed validation (10% threshold)  
negatives = generator.generate_targeted_negatives(
    expert_actions, current_phase, validation_threshold=0.10
)
```

### Focus on Specific Components

```python
strategy = generator.get_component_targeting_strategy()

# Target rare targets for T component improvement
rare_targets = strategy['target_component_focus']['rare_targets']

# Target rare combinations for IVT component improvement  
rare_combinations = strategy['combination_component_focus']['rare_combinations']
```

## 🎯 MICCAI Paper Integration

This approach provides the perfect narrative for your safety guardrails paper:

**Problem**: Surgical AI cannot learn through trial-and-error due to patient safety constraints

**Solution**: Extract prior knowledge from training data to provide safety guardrails through intelligent negative generation

**Technical Contribution**: Domain-aware negative generation framework targeting high-opportunity components

**Clinical Impact**: Learn surgical safety principles without experiencing costly clinical mistakes

## 🚨 Important Notes

1. **Run extraction BEFORE training** - priors inform the entire negative generation strategy
2. **Focus on high-opportunity components** - T (52%) and IVT (33%) have maximum improvement potential  
3. **Validate against training data** - prevents contradicting expert demonstrations
4. **Target clinical safety** - anatomical precision and combination appropriateness
5. **Review visualizations** - understand your data patterns before training

## 📞 Usage Questions

1. **When to re-extract?** When training data changes significantly
2. **How many videos needed?** Minimum 5-10 for meaningful patterns, more is better
3. **Validation threshold?** Start with 5%, adjust based on results
4. **Quick test first?** Yes, run `--quick_test` to verify everything works

---

🎯 **Ready to proceed with safety guardrails IRL training targeting high-opportunity components!**
