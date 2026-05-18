# Reinforcement Learning from Human Feedback for Alignment

Reinforcement learning from human feedback aligns large language models with human
preferences by training a reward model on pairwise comparisons of model outputs.
Human labelers rank candidate responses, and the reward model learns to score
outputs consistent with these preferences. The base language model is then fine
tuned via reinforcement learning to maximize the learned reward, with KL
divergence regularization preventing excessive drift from the initial policy.

This approach addresses the fundamental challenge of training language models on
objectives that resist explicit specification. Human preferences encode complex
judgments about helpfulness, honesty, and harmlessness that are difficult to
capture in handwritten rules. RLHF enables systems to learn from demonstration
alongside the noisier signal of comparative judgment. Constitutional AI extends
this paradigm by using model generated critiques to scale alignment without
extensive human labeling. Direct preference optimization recasts the problem as
supervised learning on preference data, simplifying training while preserving
alignment quality. RLHF and its descendants underlie the helpful instruction
following behavior of modern assistant language models.
