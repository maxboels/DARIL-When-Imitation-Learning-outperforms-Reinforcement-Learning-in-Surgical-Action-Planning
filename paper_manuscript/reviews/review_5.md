
## Introduction

In the first sentence:
- may be worth adding another sentence here just to give more contedt on why the future prediction is important and the motivation
- can omit if don’t have space

Could we make sure the citations are shwoing in the right numbering order? Not sure on refernce style but out of order here? Ignore if alphabetical
- [9] 
- I also think another sentence could be added here along the lines of: ‘RL world models in particular have shown that single configurations with no hyperparameter tuning can outperform specialized methods across diverse benchmark tasks, complete farsighted tasks such as collecting diamonds in Minecraft without human data or curricula, and capture expectations of future events during autonomous driving ’
cite: 
Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T.: Mastering diverse domains through
world models (1 2023), http://arxiv.org/abs/2301.04104
Hansen, N., Su, H., Wang, X.: Td-mpc2: Scalable, robust world models for continu-
ous control. In: The Twelfth International Conference on Learning Representations
(2024), http://arxiv.org/abs/2310.16828
Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J.,
Corrado, G.: Gaia-1: A generative world model for autonomous driving (9 2023),
http://arxiv.org/abs/2309.17080


- DARIL: think this is first use of abbreviation so need to give meaning. DARIL's acronyme is first explained in the abstract, do we need to do it again in the main manuscript before using the acronyme?

# 2. Methods

## 2.2. Dataset

- We need to mention which train and test splits we are using from the CholecT50 dataset i.e. we are using videos [2,6,14,23,25,50,51,66,79,111] for testing and the rest for training as suggested in the evaluation paper cite here[].
@article{nwoye2022data,
  title={Data splits and metrics for method benchmarking on surgical action triplet datasets},
  author={Nwoye, Chinedu Innocent and Padoy, Nicolas},
  journal={arXiv preprint arXiv:2204.05235},
  year={2022}
}

-"Title Suppressed due to excessive length" seems wrong on the top right corner of the paper next to the page number.

- 2.4 Reinforcement Learning Approaches
probably shold mention the reward function used here.

I think that if you have poor results for RL you may get asked questions about your problem formulation so results can be repeated. It may therefore be worth just adding a brief description of how you set the RL up for this context

- Dreamer [12]
I would also mention that dreamer is the current state of the art for image-based RL tasks

- A2C:
reference for A2C + could say why this and PPO were chosen

- Table 2:
formatting of table means too long. Could put ‘next’ and ‘frame’ on seperate lines or use underscores for tighter column names?


Fig. 1:
- not sure if caption/figure if correct here as don’t see any reference to RL in the figure?
- We don't have any RL results in this figure so we need to just comment on the DARIL performance over different IVT mAP score scores and components.

## 3.5 Analysis: Why RL Underperformed

- Should probably be in the discussion rather than the results?
- State-Action Representation Challenges:
as mentioned earlier, its worth explaining your state/reward desgin to make this part valid


4. Discussion
- could add something to say that RL may be advantageous when working across multiple datasets due to its ability to generalize more, especially on non-expert trajectories (add to future work?)