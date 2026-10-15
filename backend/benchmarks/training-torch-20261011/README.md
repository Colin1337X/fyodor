# Twenty-update independent PyTorch parity — October 11, 2026

The CPU and CUDA reference runs start from native step-zero checkpoints and
execute twenty AdamW updates with identical byte-token data, sequence windows,
clipping semantics and optimizer settings. Short final windows and repeated
passes through the corpus are included. The architecture is a random dense
LLaMA-style decoder: vocabulary 259, width 32, FF width 64, one layer, two query
heads, one KV head, context eight. PyTorch is 2.11.0+cu128 in the separate D:
reference environment. TF32 is disabled; six CPU workers are selected.

Both twenty-step loss trajectories and all final parameter, raw-gradient,
first-moment and second-moment arrays pass the unchanged 1e-5 contract. Maximum
loss absolute error is 9.54e-7 on both devices. Maximum scaled state error is
7.45e-7 on CPU and 1.19e-6 on CUDA. No tolerance was loosened.

The reference uses eager PyTorch operators and its AdamW optimizer. Double
RMSNorm/RoPE intermediates and a double gradient norm match the native numerical
contract. Individual operations can still round differently. The checkpoint
reader and decoder composition are original test tooling, based on the Fyodor
interfaces; PyTorch remains an external oracle and is not linked into Fyodor.

These are correctness checks, not timed competitive benchmarks. They cover a
tiny full-parameter decoder, not real-model LoRA, DPO, mixed precision or compiled
PyTorch throughput. Existing one-step and real-model native evidence is retained
in the preceding milestone. Next comparisons must measure representative larger
shapes, repeated process orders and memory, with correctness rechecked.
