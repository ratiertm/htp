# Dreamer and Latent Imagination for Reinforcement Learning

Dreamer is a model based reinforcement learning agent that learns environment
dynamics in a compact latent representation space and uses learned imagination
rollouts to optimize policy parameters. A recurrent state space model encodes
observations into stochastic latent states and predicts future latent transitions
conditioned on actions. The policy and value functions are trained entirely on
imagined trajectories sampled from this learned world model.

Latent imagination enables sample efficient learning by leveraging gradients
through the differentiable world model. The agent collects modest amounts of real
environment interaction, updates the world model on this data, then performs
extensive policy improvement in the imagined latent space. Recent extensions
including Dreamer V2 and V3 demonstrate strong performance across Atari, robot
manipulation, and Minecraft domains using a single hyperparameter configuration.
This approach represents a maturation of latent space world models for practical
reinforcement learning applications, combining the sample efficiency of model based
methods with the generality of deep neural function approximation.
