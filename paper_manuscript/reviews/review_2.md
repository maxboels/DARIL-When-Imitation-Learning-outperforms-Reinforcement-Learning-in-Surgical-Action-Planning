# Abstract

I don't understand your point here ", while reinforce-
ment learning could in principle discover new strategies and achieve beyond expert-level performance" after this fist point: "Teleoperated robotic surgery provides a natural interface for acquiring expert demonstrations for imitation learning,". Please remove the last part or improve your argument.

The 45.6% is wrong as it should start from the next action Map score (33.6%) which is at t+1s.

Keywords:
- replace Medical AI with something more appropriate.


# Introduction

Same issue with this sentence, which I find unclear in its argument trying to be made: "eleoperated robotic surgery provides a natural interface for acquiring expert demonstrations for imitation learning, while reinforcement learning could in principle discover new strategies and achieve beyond expert-level performance."

- replace years with decades.
- contributions 4 is unclear.

## 2. Methods

Could you cite the following paper for the Distil-Swin feature extraction?
@inproceedings{yamlahi2023self,
title={Self-distillation for surgical action recognition},
author={Yamlahi, Amine and Tran, Thuy Nuong and Godau, Patrick and Schellenberg, Melanie and Michael, Dominik and Smidt, Finn-Henri and N{"o}lke, Jan-Hinrich and Adler, Tim J and Tizabi, Minu Dietlinde and Nwoye, Chinedu Innocent and others},
booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
pages={637--646},
year={2023},
organization={Springer}
}

It might also be worth citing the original paper on the Swin architecture:
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={10012--10022},
  year={2021}
}

