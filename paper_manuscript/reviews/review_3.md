
# Title
- not a big fan of "A Comprehensive Analysis" in the title, it sounds a bit generic. Maybe we can come up with something more specific that reflects the content of the paper.


# Abstract
- Could we avoid using 2x"for" in a single sentence: "Teleoperated robotic surgery provides a natural interface for acquiring expert demonstrations for imitation learning."?
- We need to set up the scene a bit better before explaning what we did e.g. "We conducted..."
- Using RL and IL in the abstract but didnt define those acronyms yet.
- I wouldnt use "mimiching" and "emulate" in the same sentence, they are synonyms.
- Replace "graceful" with "smooth" or "stable".

## 2. Methods
- We need to add more information regarding the dataset, how many videos, how many actions, how many expert demonstrations, etc.

- We should change the order of the RL methods as follows:
    Latent World Model + RL
    Direct RL on videos
    Inversed RL with Learned Rewards
- I think we should explain how the negative rewards are used during training in the IRL method section.
- For the World Model, it would be good to mention that the goal is to have a simulator and that the latent world model is our simulator used for policy learning.

- In the "2.1 Evaluation Framework" section, could you clarify the "statistical validation" part?

## 3. Results
- Not sure how to define our methods names. I think it is usually better to have a short name for each method, while trying to propose our own winning method with a specific name if people decide to use it in the future. We should define CAR as Causal or Masked Autoregressive IL (MAIL) or something like that.
- We need to define Next in the Table 1's caption.
- We also have enough space to add more columns in the table with more planning horizons mAP scores.
- Once I have the results for the other methods, I will add them to the subsections' discussion.
















<!-- ############### WHAT I LIKE IN THE OTHER PAPER ################## -->