# Gemma packed Q4 gate/up prototype — 2026-11-13

Original Fyodor CUDA source under `optcpp/matvec/gemma_gate.cu`, adapted from the
existing aligned-byte gate/up kernel. It pairs 3840-column Q4 projections with
Gemma's existing tanh GELU expression. The ABI requires 2400-byte packed rows,
256 threads and eight rows per block; partial last blocks are supported.

Opt-in integration requires `NYA_ENABLE_OPTCPP=ON` and
`NYA_ENABLE_GEMMA_GATE=ON` at configure time, then `NYA_CUDA_GEMMA_GATE=1`
at session creation. The gate option defaults off. Only single-token Gemma
layers with 3840 input columns and both eligible packed Q4 matrices use it.
Memory-budget checks and native fallback remain; physically absent optcpp
sources are skipped at configure time. No end-to-end speed benefit is established.

The saved short-decode profile attributes 74.77 of 182.12 ms of matmul time to
the two 15360-by-3840 feed-forward projections. The prototype targets that cost
by reusing the input load and combining activation. Register pressure may offset
those savings.

`check_kernel.py` compares against two production packed Q4 dot calls followed
by native vector GELU. All eight row counts (1, 7, 8, 9, 63, 64, 65, 15360)
pass bit-for-bit with guard canaries and finite synthetic outputs. CUDA memcheck,
racecheck, synccheck and initcheck logs are retained. This establishes agreement
with the existing arithmetic, not an independent double-precision accuracy bound.

`model_oracle.c` compares full vocabulary logits with fusion disabled/enabled at
3, 4, 257 and 1026 total tokens. All are bit-exact in `model-oracle.log`;
`model-oracle-exit.json` records the environment. Both modes use byte-dot packed
weights. Launch counts drop by 96 per single-token decode (48 layers).

`path_oracle.c` strengthens this by asserting the expected launch reduction and
requiring every nonsuppressed logit to be finite. Only model-declared suppressed
tokens may be negative infinity, and their masks must match. The original scaled
error bound remains 0.001; measured errors below are all exactly zero.
`run_paths.py`, `path-oracle-results.json` and `path-oracle-manifest.json` retain
commands, environments, source/binary/model hashes and exit codes.

| Case | Total tokens | Extra packed weights | Launch reduction | Result |
| --- | ---: | ---: | ---: | --- |
| Gate-only, byte-dot disabled | 3 | 3,538,944,000 bytes | 288 (3 decode calls) | bit-exact |
| Gate-only, byte-dot disabled | 1026 | 3,538,944,000 bytes | 192 (2 decode calls) | bit-exact |
| 9,000 MiB budget, gate-only | 3 | 0 | 0 (fallback) | bit-exact |
| Gate-only, 8K context | 8194 | 0 | 0 (fallback) | bit-exact |

The first path-oracle attempt mistakenly rejected Gemma's intentionally suppressed
logits; `oracle-mask-rejected.*` preserves that harness failure. The corrected
oracle checks those tokens explicitly against the tokenizer's suppression list.
It does not accept arbitrary matching infinities or NaNs.

`timing-gate/environment.json` records rejection before any timed run because
competing desktop GPU identities were present. Correctness runs are not timing
claims. Accepted llama.cpp scores remain unchanged; clean matched timing is pending.

GCC C17 CUDA build and model checks pass. GCC C23 CUDA benchmark/TUI targets
compile; three selected TUI tests pass in `c23-tui-complete.log`. Earlier logs
retain a mistaken target name and an unbuilt test executable, both resolved.
`removal-complete.log` verifies a fresh source copy lacking optcpp, CUTLASS and
CuTe: all 78 registered tests pass (four expected absent-hardware ROCm/MLX skips),
followed by CPU inference on a training/export fixture. Earlier removal failures
exposed omitted SQLite/theme test inputs; the copier now preserves those native
C dependency and theme files. Synthetic smoke timing is not performance acceptance.
The TUI command-scrolling regression passes in C23 and the final C17 removal run.

See `../gemma4-partial-gate-20261113/README.md` for the separate opt-in partial
packing experiment. The default fallback policy documented above is unchanged.
