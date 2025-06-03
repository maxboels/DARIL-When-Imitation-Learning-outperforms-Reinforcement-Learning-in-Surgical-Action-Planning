# Paper Content

## Results Section

### Overall Performance
The best performing method was **Random** with an average mAP of 0.028 across all horizons.

### Performance Across Horizons
We evaluated all methods across multiple prediction horizons (1, 3, 5, 10, 15, 20 steps). Key findings include:
- Random: -4.6% performance degradation from horizon 1 to 20

### Statistical Significance
No statistically significant differences were found between methods.

## Discussion Section

### Key Findings
Our comprehensive evaluation reveals several key insights:
1. **Planning vs. Recognition Trade-off**: Methods showed different strengths for immediate action recognition versus long-term planning capabilities.
2. **Horizon-Dependent Performance**: All methods exhibited performance degradation with increasing prediction horizons, but at different rates.
3. **Clinical Relevance**: Performance varied significantly across different types of surgical actions, with critical actions showing different patterns.

### Method-Specific Insights
**Imitation Learning**: Showed strong performance for immediate action prediction but limited planning capabilities beyond short horizons.
**Reinforcement Learning**: Demonstrated better long-term planning consistency but required more training data and computational resources.

### Limitations and Future Work
1. **Dataset Limitations**: Evaluation was limited to CholecT50 dataset. Future work should validate on additional surgical datasets.
2. **Real-time Performance**: Clinical deployment requires real-time inference capabilities that warrant further optimization.
3. **Surgical Workflow Integration**: Integration with existing surgical workflow systems presents additional challenges.