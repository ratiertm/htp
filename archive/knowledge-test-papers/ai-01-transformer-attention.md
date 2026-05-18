# Transformer Architecture and Self Attention

The transformer architecture revolutionized natural language processing by replacing
recurrent connections with self attention as the primary mechanism for sequence
modeling. Self attention computes weighted sums of value vectors where the weights
arise from dot products of query and key projections of input tokens. This
mechanism captures long range dependencies in a single layer regardless of
sequence position distance.

Multi head attention runs several parallel attention operations in different
representation subspaces, enabling the model to attend to multiple relationships
simultaneously. Positional encodings provide the model with sequence order
information since attention itself is permutation equivariant. The transformer
stacks multiple attention layers with feedforward networks, layer normalization,
and residual connections to form deep networks. Scaling transformer models to
billions of parameters trained on web scale text data produces the foundation
models underlying modern language understanding systems. The architecture has been
extended to vision, audio, and multimodal domains, becoming the dominant paradigm
for sequence modeling across machine learning.
