# World Models for Reinforcement Learning Agents

Ha and Schmidhuber introduced world models as a framework for learning compact
representations of environment dynamics from agent experience. The architecture
separates perception, memory, and decision making into distinct neural network
components. A variational autoencoder learns a compressed latent representation of
visual observations. A recurrent network with mixture density outputs predicts
future latent states conditioned on actions, forming an internal simulator of
environment dynamics.

Trained on rollouts collected by random policy, the world model captures
environment regularities sufficient for policy learning to occur entirely in
imagination. A small controller network selects actions based on the latent state
representation and predicted dynamics, enabling sample efficient learning through
mental simulation rather than direct environment interaction. This approach
demonstrates that abstract latent space models can support effective decision making
when grounded in self supervised representation learning from agent experience.
World models provide a foundation for combining model based planning with deep
neural network function approximation in reinforcement learning agents.
