# Joint Embedding Predictive Architectures for Video Understanding

The Joint Embedding Predictive Architecture (JEPA) learns by making predictions in
a learned representation space rather than reconstructing raw input pixels. By
abstracting input through an encoder before prediction, JEPA filters noise and
irrelevances to focus on the essence of the predictive task. This contrasts with
generative architectures that must model all pixel level details, including
unpredictable variation.

V-JEPA extends this approach to video, learning self supervised video representations
by predicting masked spatiotemporal regions in latent space. V-JEPA 2 was pretrained
on over one million hours of internet video and achieves strong performance on
motion understanding tasks and action anticipation benchmarks. The architecture
demonstrates that abstract latent space prediction supports robust video
understanding without pixel level reconstruction loss. Combining internet scale
video data with small amounts of robot interaction data enables world models capable
of physical understanding, future prediction, and planning. JEPA based world models
represent a paradigm shift toward representation learning that prioritizes semantic
structure over reconstruction fidelity, supporting downstream tasks in robotics and
embodied agent control.
