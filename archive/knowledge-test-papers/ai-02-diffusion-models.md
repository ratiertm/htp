# Diffusion Models for Generative Modeling

Diffusion models generate data by learning to reverse a gradual noising process
that converts samples into pure Gaussian noise. The forward process incrementally
adds noise to training data through a Markov chain. The reverse process trains a
neural network to denoise progressively, learning the score function of the data
distribution at each noise level. Sampling proceeds by starting from random noise
and iteratively applying the denoising network.

Diffusion models achieve state of the art image generation quality, surpassing
generative adversarial networks on diverse benchmarks. Text to image systems such
as Stable Diffusion and DALL-E condition the denoising process on text embeddings
to generate images from natural language descriptions. Latent diffusion operates in
compressed feature space rather than pixel space, dramatically reducing computation
while preserving fidelity. Recent advances extend diffusion to video generation,
3D shape synthesis, and protein structure prediction. The framework provides a
principled probabilistic foundation for high quality generative modeling across
modalities, with theoretical connections to score matching and stochastic
differential equations.
