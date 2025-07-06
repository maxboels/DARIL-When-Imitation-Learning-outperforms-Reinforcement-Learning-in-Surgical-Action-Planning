

# Abstract:
- First sentence doesn't fit the second sentence.
- replace current action map with action triplet recognition.
- same for next action mAP, replace with for the next frame.

# 1. Introduction
- try to avoid using em dash in the intro.
- I love the tought provoking introduction.

# Methods
## 2.1 Problem Formulation or 2.2 Dataset
- It might be good to add that a frame can have multiple actions at the same time (since there are multiple robotic arms with different instruments). It ranges from 0-3 actions per frame and 100 actions making the problem very sparse (this could be mentioned in the next subsection about the dataset).

## 2.3 DARIL
- The GPT-2 model only takes frame embeddings with 1024-dims, so no action labels is given as input.


# 4. Discussion
## 4.1 Implications for Surgical AI
- I am not a fan of the ressource allocation argument. Could you instead say something about bootstapping an RL model with imitation learning basics skills and practice and then use a good physics simulator for exploring new techniques safely instead of on the patients? Exploration and mistakes are crucial for improving techniques etc but they need to be tested in simulation or with surgical world models.

# 4.2. Limitations and Future Work
- one big limitation in our work is the overfitting problem on a few videos and likely doesnt generalise on test videos, so more data is needed for simulator wheter its a wold model or a physics engine.
- lack of rewards feedback from possible future states and lack of outcome data.


# Conclusion
- all good nothing to comment on.



