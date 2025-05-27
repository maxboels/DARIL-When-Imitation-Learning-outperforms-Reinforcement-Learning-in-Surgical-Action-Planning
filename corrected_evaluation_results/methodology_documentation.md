
# Corrected Evaluation Methodology

## Inference Strategy: Single-Step Prediction
- At each timestep t, predict action for timestep t+1
- Input: Frame embedding at timestep t
- Output: Action prediction for timestep t+1
- Prediction matrix shape: (video_length-1, num_classes)

## Evaluation Strategy: Cumulative mAP
- At timestep t, compute mAP using predictions from start to timestep t
- Shows how prediction quality changes as trajectory progresses
- mAP trajectory shape: (video_length-1,)

## Matrix Shapes:
- Ground Truth: (video_length-1, num_classes)
- Predictions: (video_length-1, num_classes)
- mAP Trajectory: (video_length-1,)

## Methods Compared:
- Imitation Learning
- Ppo
- Sac

## Key Results:
- Imitation Learning: 0.979 mAP
- Sac: 0.331 mAP
- Ppo: 0.000 mAP
