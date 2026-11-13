# Gemma packed Q4 gate/up prototype — 2026-11-13

Original Fyodor CUDA source under `optcpp/matvec/gemma_gate.cu`, adapted from the
existing aligned-byte gate/up kernel. It pairs 3840-column Q4 projections with
Gemma's existing tanh GELU expression. The ABI requires 2400-byte packed rows,
256 threads and eight rows per block; partial last blocks are supported.
No runtime dispatch or build option selects this prototype yet.

The saved Gemma short-decode profile attributes 74.77 of 182.12 ms of matmul
time to the two 15360-by-3840 feed-forward projections. This prototype targets
that measured cost by reusing the input load and combining the activation.
Performance is unmeasured; register pressure may offset those savings.

`check_kernel.py` compares the fused output bit-for-bit against the production
packed Q4 dot kernel called twice followed by native vector GELU. All eight
row counts (1, 7, 8, 9, 63, 64, 65, 15360) pass with guard canaries and finite
outputs, using deterministic synthetic packed weights. It does not establish
real-model parity, independent double-precision accuracy, or end-to-end speed.
CUDA memcheck/racecheck/synccheck/initcheck logs are retained individually.
Runtime integration and matched model timing are the next acceptance steps.
