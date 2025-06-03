

graph TD
    subgraph "Input Processing"
        Video[Video Input] --> Frames[Frame Extraction]
        Frames --> Embeddings[Frame Embeddings z_seq]
    end

    subgraph "Recognition Module"
        Embeddings --> RecHead[Recognition Head]
        RecHead --> CurrentActions[Current Action & Instrument Recognition]
    end

    subgraph "Future Prediction Module"
        Embeddings --> WorldModel[Causal GPT2 World Model]
        WorldModel --> FutureEmbed[Future Frame Embeddings]
        WorldModel --> FutureActions[Future Action Predictions f_a_seq_hat]
    end

    subgraph "Reward Learning"
        CurrentActions --> RewardModel[Reward Predictor]
        FutureActions --> RewardModel
        RewardModel --> RewardSignal[Action-Based Reward Signal]
    end

    subgraph "Training Methods"
        IL[Imitation Learning] --> |Phase 1| WorldModel
        IL --> |Phase 1| BC[Behavior Cloning]
        BC --> RecordedActions[Recorded Actions from Videos]
        RewardSignal --> |Phase 2| OfflineRL[Offline RL]
        OfflineRL --> |Conservative Q-Learning| PolicyModel[Policy Model]
        OfflineRL --> |TDMPC2| WorldModel
    end

    CurrentActions --> Eval[Performance Evaluation]
    FutureActions --> Eval
    PolicyModel --> Eval