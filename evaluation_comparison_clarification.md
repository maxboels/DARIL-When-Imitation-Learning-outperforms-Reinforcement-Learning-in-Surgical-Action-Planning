# What We're Actually Comparing

## 🎯 **The Three Approaches for Action Prediction**

### Method 1: Imitation Learning
```
Input: Surgical state sequence
↓
AutoregressiveIL Model
↓
Output: Action probabilities
```
**What it does**: Directly learns to mimic expert action sequences

### Method 2: RL with World Model
```
Input: Surgical state
↓
RL Policy (trained in world model simulation)
↓
Output: Action probabilities
```
**What it does**: Learns optimal actions through RL training in simulated environment

### Method 3: RL with Direct Video
```
Input: Surgical state  
↓
RL Policy (trained on video episodes)
↓
Output: Action probabilities
```
**What it does**: Learns optimal actions through RL training on real video episodes

## 🔍 **Key Insight: All Methods End Up Predicting Actions**

For fair comparison, we evaluate all three on the **same task**: Given a surgical state, predict the next action(s).

- **IL**: Trained to directly predict actions from demonstrations
- **RL+WorldModel**: Trained to select actions that maximize reward in simulation  
- **RL+DirectVideo**: Trained to select actions that maximize reward on video episodes

## 🎯 **Why This Comparison Makes Sense**

1. **Same Input**: All methods take surgical state embeddings
2. **Same Output**: All methods output action predictions  
3. **Same Evaluation**: All methods evaluated on mAP for action prediction
4. **Different Learning**: They learn through different paradigms

## 🧠 **The Research Question**

> "For surgical action prediction, is it better to:
> - Directly learn from expert demonstrations (IL)?
> - Learn optimal policies in simulation (RL+WorldModel)?  
> - Learn optimal policies on real episodes (RL+DirectVideo)?"

## 📊 **What the World Model Actually Does**

**During Training**:
```
RL Agent: "What if I take action A in state S?"
World Model: "You'll get state S' and reward R"
RL Agent: "Let me try many actions and learn the best policy"
```

**During Evaluation**:
```
Evaluation: "Given state S, what action should we take?"
Trained RL Policy: "Based on my training, action A is best"
```

The world model is **never directly used for action prediction** - it's only used to train the RL policy.

## ✅ **Correct Evaluation Setup**

```python
# Method 1: IL action prediction
action_probs = il_model.predict_next_action(state)

# Method 2: RL policy (trained with world model) action prediction  
action_probs = world_model_trained_policy.predict(state)

# Method 3: RL policy (trained on videos) action prediction
action_probs = video_trained_policy.predict(state)
```

All three are evaluated on the same action prediction task, but they learned through different methods.
