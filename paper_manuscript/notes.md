# Abstract:
## Motivations:
That's not exactly true: "Surgical action planning requires learning from expert demon-
strations while ensuring safe and effective decision-making".
I would say something like: "Imitation learning on expert-level demonstrations, like toeleoperation from real or training can help learn action policies in robotic surgery." AND "Teleoperated robotic surgery has a natural interface for aqcioring expert demonstrations for imitation learning." Using RL could in principle discover new strategies and go beyond expert level performance. This is what we searched in this paper.
## Methods:
Imitation learning through direct supervised learning on expert demonstrations videos as our baseline.
We tested serveral techniques to improve the performance of the IL baseline.
## Results:
We found that the IL baseline is better than the new policies.
## Conclusion:
Distribution matching problem on the evaluation tests set favours the IL baseline over the potential valid or overn better new policies but different from the expert demonstrations (same as the one used for training).


Vocabulary to be used in the abstract, introduction, and conclusion:
- IL: Imitation Learning
- Supervised learning: A type of machine learning where the model learns from labeled data, typically expert demonstrations in this context.
- Expert demonstrations: High-quality examples of task execution by skilled practitioners, used for training models.
- RL: Reinforcement Learning
- Emulate: To imitate or mimic the behavior of an expert.
- Mimicking: The process of replicating the actions or strategies of an expert in a specific task.
- Surgical action planning: The process of determining the sequence of actions to perform in a surgical procedure
- Teleoperation: Remote operation of a robotic system, often used in surgical contexts.
- Simulation (Physics Engines), world model (Neural Engine or simulator), and real environment: Different contexts in which the models are trained and evaluated.


# 1. Introduction:
I like this but we might need to either define or cite surgical action planning beforehand making this statement: "Surgical action planning represents one of the most challenging applications of artificial intelligence in healthcare, requiring models to learn from expert demonstrations while ensuring safe and effective decision-making."

I love this part: "Expert surgical demonstrations represent years of refined technique and training, potentially making them near-optimal for many evaluation criteria. This raises a fundamental question: under what conditions does RL improve upon well-optimized IL in expert domains?" It's a thought-provoking question that sets the stage for the paper's exploration of RL versus IL in surgical action planning while acknowledging the value of the clinical importance and expert surgeons knowledge.

We either need a specific '2. Related Work' section or we can merge it with the introduction. I like the idea of having a '2. Related Work' section, but we need to make sure it is not too long and that it is focused on the most relevant works in the field.

Personally, I don't read the related work section and prefer setting the stage of related work in the introduction. It might even be more true for miccai workshop papers.

We need to be cautious about this kind of statement in our conclusion or intro since we worked with a limited dataset of offline recorded videos and constraining evaluation metrics and lack of reward and outcome data which is stanadard for classic RL approaches: "Our key finding challenges conventional wisdom: sophisticated RL approaches fail to improve upon a well-optimized IL baseline, achieving comparable or worse performance across multiple evaluation metrics."

# 2. Methods:
I like how you describe the methods, let's keep it like this for now. We can always refine it later.

# 3. Results:
Missing elements:
- Reviwers will ask for a figure that compares the diffent methods which I don't have atm and don't think I have the time to run. Those experiments were kind of failing in performance so I decided to not go until the end but I have some results for a table.

We might therefore just present the comparison table with final metrics and also a figure with the baseline qualitative results.

I really like your analysis in the "3.4 Why RL Underperformed" subsection. You can keep it unchanged.

We could add that one limitation of imitation learning on expert demonstrations from surgeries with good outcome and non-complicated procedures might overlook the trail-and-error learning from RL which permits to recover from mistakes and unexplored events. There is not exploration during learning which limits safety capabilities.

Could we also think of what's missing in terms of states rewards and expected final outcome for RL approaches?


# Final Comments:
Finally, could you think of a better title? Please make 5 suggestions.




