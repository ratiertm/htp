# Mixture of Experts and Sparse Neural Networks

Mixture of experts architectures scale neural networks by routing inputs to
specialized sub networks rather than activating all parameters for every input. A
learned gating network selects a sparse subset of expert sub networks to process
each input token, with only the selected experts contributing to the forward pass.
This sparse activation pattern enables dramatic parameter scaling while maintaining
computational efficiency per inference.

Modern mixture of experts language models contain hundreds of billions or trillions
of parameters yet activate only a fraction during any single forward pass. Top
two routing selects the two highest scoring experts per token, balancing capacity
and load distribution across the expert pool. Auxiliary load balancing losses
prevent the gating network from collapsing to a small expert subset. Recent
implementations including Switch Transformer and Mixtral demonstrate that sparse
mixture of experts achieves strong performance with significantly lower
computational cost than dense models of equivalent total parameter count. This
architecture represents a key technique for continued scaling of foundation models
as dense scaling encounters compute and memory limits.
