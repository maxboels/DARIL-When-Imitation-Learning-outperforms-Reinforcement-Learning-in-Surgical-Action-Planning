surgical_action_prediction/
├── README.md
├── requirements.txt
├── config.yaml                    # Single clean config
├── main.py                       # Main experiment runner
│
├── datasets/
│   ├── __init__.py
│   ├── cholect50.py              # Keep this - it's good
│   └── preprocessing/            # Rename from current preprocessing files
│       ├── __init__.py
│       ├── phase_completion.py
│       ├── progression.py
│       └── risk_scores.py
│
├── models/
│   ├── __init__.py
│   └── dual_world_model.py       # Keep - this is correct
│
├── environments/
│   ├── __init__.py
│   ├── world_model_env.py        # From rl/environment.py
│   └── direct_env.py             # New - for non-world-model RL
│
├── training/
│   ├── __init__.py
│   ├── il_trainer.py             # Clean IL trainer
│   ├── rl_trainer.py             # Clean RL trainer
│   └── world_model_trainer.py    # For world model pre-training
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluator.py              # Single clean evaluator
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Keep
│   └── visualization.py
│
├── experiments/
│   ├── __init__.py
│   ├── run_il_baseline.py
│   ├── run_rl_with_world_model.py
│   ├── run_rl_without_world_model.py
│   └── run_comparison.py
│
├── configs/
│   ├── il_config.yaml
│   ├── rl_world_model_config.yaml
│   └── rl_direct_config.yaml
│
└── results/                      # Clean results directory
    ├── il_baseline/
    ├── rl_world_model/
    ├── rl_direct/
    └── comparison/