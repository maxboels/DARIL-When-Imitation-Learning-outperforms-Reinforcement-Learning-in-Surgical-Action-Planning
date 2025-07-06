

# Abstract:
Surgical action triplet prediction has primarily focused on
recognition tasks for activity analysis. However, real-time surgical assistance requires next action prediction for planning and control applications. This work presents the first comprehensive comparison of Imitation Learning (IL) versus Reinforcement Learning (RL) approaches for surgical next action prediction, evaluating both recognition accuracy and planning capability. We propose an autoregressive IL baseline achieving 0.979 mAP on CholecT50, and systematically compare it against multiple RL variants including world model-based RL, direct video RL, and inverse RL enhancement. Our analysis reveals that ..."

- contribution saying that our baseline is comparable to sota performance on the dataset.

## inlcude the Task formulation:
3.1 Problem Formulation
We formulate surgical action triplet prediction as a sequential decision making
problem. Given a sequence of surgical video frames {f1, f2, ..., ft}, the task is
to predict future action triplets {at+1, at+2, ..., at+H } where H represents the
prediction horizon.
Each action triplet ai = (Ii, Vi, Ti) consists of an instrument Ii ∈ I, verb
Vi ∈ V, and target Ti ∈ T from predefined vocabularies. We evaluate both single-
step prediction (H = 1) for recognition comparison and multi-step prediction
(H > 1) for planning assessment.