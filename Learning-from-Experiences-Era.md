Let me continue with the second part of the Surgical TD-MPC implementation:

# End-to-End Offline RL Implementation for Surgical Action Selection

I've implemented a complete solution for surgical action policy learning based on the "Era of Experience" philosophy discussed in the podcast with David Silver. This approach adapts the AlphaZero/AlphaGo concepts to your specific surgical video domain where you can't directly simulate actions but can still learn from offline data.

## Implementation Overview

I've created two key components:

1. **Surgical World Model** - A model that predicts future frame embeddings based on the current state and actions, handling multi-label actions (up to 3 actions per frame).

2. **Surgical TD-MPC Policy** - An offline reinforcement learning approach using Temporal Difference Model Predictive Control to select actions.

## Key Design Decisions

### 1. Handling Multi-Label Actions

The system handles multiple actions per frame through a multi-label classification approach, using:
- Binary action vectors with up to 3 active actions per frame
- Sigmoid activations and binary cross-entropy losses
- Special sampling mechanism to ensure 1-3 actions are selected

### 2. Bridging the Training-Inference Gap

To address the 40% action recognition accuracy challenge:
- **Dual-Path Training**: Simultaneously trains with perfect and noisy actions
- **Confidence-Weighted Attention**: Uses 40% confidence for actions, 90% for tools
- **Progressive Noise Schedule**: Gradually increases noise during training

### 3. Value Function Definition

Following David Silver's approach, I've defined a value function that includes:
- **Progress in Embedding Space**: Rewarding consistent movement through the procedure
- **Action-Specific Rewards**: Based on statistics from offline data
- **Exploration Bonus**: Encouraging diverse states

### 4. Credit Assignment

The system solves the credit assignment problem with:
- **TD Learning**: Using temporal differences to propagate rewards
- **State Value Estimation**: Measuring improvements in predicted outcomes
- **Action Value Estimation**: Learning which actions lead to better outcomes

### 5. TD-MPC Planning

The TD-MPC component addresses the simulation constraint by:
- Using the world model to simulate trajectories from different actions
- Planning with a learned value function that estimates long-term outcomes
- Selecting actions that maximize predicted value

## How to Use This Implementation

1. **Train the World Model**
   ```bash
   python surgical_world_model.py --config config.yaml --mode train
   ```

2. **Train the TD-MPC Policy**
   ```bash
   python surgical_td_mpc.py --config config.yaml --mode train --world_model path/to/trained_world_model.pt
   ```

3. **Evaluate the Policy**
   ```bash
   python surgical_td_mpc.py --config config.yaml --mode eval --world_model path/to/trained_world_model.pt --policy path/to/policy.pt
   ```

## Alignment with David Silver's Vision

This implementation embraces key principles from the "Era of Experience" podcast:

1. **Learning from Experience**: The world model learns to predict outcomes from actions
2. **Going Beyond Human Data**: The TD-MPC policy can discover action combinations that might be better than the expert demonstrations
3. **Credit Assignment**: The value function helps assign credit to actions that lead to successful outcomes
4. **Self-Learning**: The system can improve through its own predictions and evaluations

## Future Extensions

1. **Phase-Specific Policies**: Train separate policies for different surgical phases
2. **Safety Constraints**: Add safety constraints to avoid risky actions
3. **Online Fine-Tuning**: Add capability for online adaptation with new data

This implementation provides a complete framework for learning surgical action policies from offline data, addressing the specific challenges of multi-label actions and the gap between training and inference.