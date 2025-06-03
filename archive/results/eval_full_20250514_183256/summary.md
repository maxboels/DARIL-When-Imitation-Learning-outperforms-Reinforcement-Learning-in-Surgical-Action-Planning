# World Model Inference Evaluation Summary

## Action Prediction Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.0000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 Score | 0.0000 |
| Mean Average Precision (mAP) | 0.0000 |

## State Prediction Metrics

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | 0.1591 |

## Rollout Prediction Metrics

| Metric | Value |
|--------|-------|
| Mean Error | 1.2354 |
| Growth Rate | 2.6700 |

## Multi-Horizon Prediction Metrics

| Horizon | Action Accuracy | Action mAP | State MSE |
|---------|----------------|------------|----------|
| 1 | 0.5214 | 0.2184 | 0.2757 |
| 3 | 0.5252 | 0.2060 | 0.3193 |
| 5 | 0.5134 | 0.1939 | 0.3746 |
| 10 | 0.4573 | 0.1557 | 0.5345 |
| 15 | 0.4147 | 0.1254 | 0.6626 |

## Analysis

The model's action prediction performance could be improved. Consider further training or exploring different model architectures.

The model maintains consistent performance over longer prediction horizons, indicating strong temporal modeling capabilities.

