# Gemma packed Q4 gate/up prototype — 2026-11-13



Original Fyodor CUDA source under `optcpp/matvec/gemma_gate.cu`, adapted from the

existing aligned-byte gate/up kernel. It pairs 3840-column Q4 projections with

Gemma's existing tanh GELU expression. The ABI requires 2400-byte packed rows,

256 threads and eight rows per block; partial last blocks are supported.

Opt-in integration requires `NYA_ENABLE_GEMMA_GATE=ON` at configure time
and `NYA_CUDA_GEMMA_GATE=1` at session creation. Both default off. Only
single-token Gemma layers with 3840 input columns and both eligible packed Q4
matrices use it. Memory-budget checks and native fallback remain; physically
absent optcpp sources are skipped at configure time.



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

The real-model oracle in `model_oracle.c` compares all vocabulary logits with
fusion disabled/enabled at 3, 4, 257 and 1026 total tokens. All are bit-exact in
`model-oracle.log`; the environment is in `model-oracle-exit.json`. Kernel counts
drop by 96 per single-token decode (48 layers, two launches saved per layer),
confirming dispatch. Both modes use packed byte-dot weights. Matched model timing
remains pending; accepted llama.cpp scores are unchanged. Gate-only packing
without byte-dot and long-context allocation fallback are not yet model-validated.



Integration validation: GCC C17 CUDA build and model oracle pass. GCC C23 CUDA
benchmark/TUI targets compile; the three selected TUI tests pass in
`c23-tui-complete.log`. Earlier build/test logs retain a mistaken target name
and an unbuilt parser executable, resolved by building the correct targets.
`removal-complete.log` verifies a fresh source copy physically lacking optcpp,
CUTLASS and CuTe: all 78 registered tests pass (four expected absent-hardware
ROCm/MLX skips), followed by CPU training/export model inference smoke runs.
Earlier removal logs retain missing SQLite/theme test-input failures; the copier
now preserves those native-C dependency and theme inputs. Synthetic smoke timing
is not a performance acceptance result. The TUI long-command scrolling regression
is covered in both C23 and the final C17 removal test run.
