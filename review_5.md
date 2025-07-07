
## Introduction

- may be worth adding another sentence here just to give more contedt on why the future prediction is important and the motivation
- can omit if don’t have space

[9]
Not sure on refernce style but out of order here? Ignore if alphabetical 

I also think another sentence could be added here along the lines of: ‘RL world models in particular have shown that single configurations with no hyperparameter tuning can outperform specialized methods across diverse benchmark tasks, complete farsighted tasks such as collecting diamonds in Minecraft without human data or curricula, and capture expectations of future events during autonomous driving ’
cite: 
Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T.: Mastering diverse domains through
world models (1 2023), http://arxiv.org/abs/2301.04104
Hansen, N., Su, H., Wang, X.: Td-mpc2: Scalable, robust world models for continu-
ous control. In: The Twelfth International Conference on Learning Representations
(2024), http://arxiv.org/abs/2310.16828
Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J.,
Corrado, G.: Gaia-1: A generative world model for autonomous driving (9 2023),
http://arxiv.org/abs/2309.17080


- DARIL: think this is first use of abbreviation so need to give meaning

-"Title Suppressed due to excessive length"

- 2.4 Reinforcement Learning Approaches
probably shodl mention the reward function used here

I think that if you have poor results for RL you may get asked questions about your problem formulation so results can be repeated. It may therefore be worth just adding a brief description of how you set the RL up for this context

- Dreamer [12]
I would also mention that dreamer is the current state of the art for image-based RL tasks

- A2C:
reference for A2C + could say why this and PPO were chosen

- Table 2:
formatting of table means too long. Could put ‘next’ and ‘frame’ on seperate lines?


Fig. 1:
not sure if caption/figure if correct here as don’t see any reference to RL in the figure?

## 3.5 Analysis: Why RL Underperformed

Should probably be in the discussion rather than the results?

State-Action Representation Challenges:
as mentioned earlier, its worth explaining your state/reward desgin to make this part valid


4. Discussion
- could add something to say that RL may be advantageous when working across multiple datasets due to its ability to generalize more, especially on non-expert trajectories (add to future work?)