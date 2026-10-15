# Resident decoder graph integration — 2026-10-07

Validation and packaging completed on 2026-10-08.

This milestone connects the private resident session to the existing native C
training tensor API. Dense LLaMA and supported imported Gemma decoders can now
construct complete forward/backward CUDA graphs, accumulate gradients, apply
AdamW, checkpoint, resume, evaluate and export. The public trainer and desktop
training controls still execute on CPU. This is correctness and lifecycle
integration, not an accepted performance improvement or completed GPU product.

Base revision: `bd4c3c6d945a967a9d1072eab7377380f878cda0`. All code remains behind
Fyodor C interfaces; no foreign runtime objects or training framework are linked.
[Raw evidence and reproduction scripts](training-graph-20261007/) preserve the
failed experiments as well as the successful checks.

## Execution and ownership

`training_graph_device.h` supplies the private constructor and explicit forward
entry point. Existing tensor operations record device nodes; CPU graphs retain
their eager behavior. Construction reserves bounded intermediates and uploads
inputs, masks, token maps and RoPE frequencies. Frozen model tensors register
once per session, preserving packed matrix storage. Repeated model preparation
reuses these resident copies. Trainable parameters bind the session's persistent
values and gradient buffers. Keep the decoder and source mapping alive until
its graphs and session are released; frozen tensor descriptors are borrowed keys.

Forward and backward enqueue nodes on one ordered stream, without allocation,
host transfers or host synchronization inside the execution region. Tests check
these counters before and after execution. Explicit data access synchronizes the
numerical status and downloads a cached host snapshot. Calling forward again
without adding nodes launches no work. Adding nodes after observation is allowed;
new metadata uploads are explicit construction costs and may synchronize.

Each graph owns one scratch scope and retains its session. A caller may release
its session/parameter references before graph cleanup. Parameters remain valid
until that cleanup. The session rejects overlapping graphs and optimizer/reset/
checkpoint/detach while a graph owns its scope. A failed or abandoned training
graph requires explicit gradient reset before another update. Numerical status
checks guard all node values and gradients; an invalid status prevents AdamW
publication. Fatal device errors keep the earlier session recovery contract.

The graph budget charges host graph metadata, device intermediates and explicit
host snapshots. Persistent parameter arrays, frozen model bindings, arena
alignment and driver overhead are separate. The caller also bounds total device
arena bytes. The real model probe reserves 3 GiB, with approximately 144 MB of
charged CE graph allocations and 288 MB for paired DPO, excluding persistent
state and alignment. These are allocation counts, not a peak-VRAM measurement.

A new column-slice kernel covers Gemma per-layer embeddings and accumulates
backward contributions in place. Invalid extents, overflow and source/destination
aliasing are rejected. General linear input/weight gradient aliases and shared
query/key/value gradient buffers remain unsupported by the resident graph and
fail explicitly; supported decoder graphs do not require them. Mixed precision,
activation checkpointing, persistent graph reuse and public GPU selection remain
future work.

## Rejected numerical and capacity experiments

The initial default CUDA compilation passed small LLaMA training but diverged
on Gemma PLE full-weight training. At parameter 63, CPU value 0.951686382 and
GPU value 0.951636374 exceeded the unchanged 1e-5 scaled state bound
(error 2.56231213e-5). Reference compilation passed. Disabling multiply/add
contraction for the training module corrected the normal compilation trajectory;
inference compilation options remain unchanged. A cancellation regression now
requires an exactly zero accumulated gradient where fused arithmetic would
leave a small residual that AdamW could amplify. This is a precision correction;
its throughput cost has not been isolated or accepted as a speed improvement.

The former 4,096 buffer-descriptor limit admitted real-model CE but rejected the
second branch of DPO. The bounded descriptor table is now 16,384 entries; its
exhaustion test uses the same constant and enough arena space to reach the
actual descriptor boundary. The byte arena remains caller-bounded. The real
probe uses 3,530 live buffers for CE and 5,617 for paired DPO.

The first real-model export check compared quantized-base/separate-LoRA training
against merged F32 inference at a newly introduced 1e-5 scaled bound. It failed:
maximum scaled difference 4.4451796e-5, maximum absolute 6.36875629e-5. At the
worst token/vocabulary entry, training produced 0.432733178 and merged inference
0.432796866. These are different projection representations and reduction paths.
The diagnostic remains in every real-model run and its failed logs are retained.

The accepted GPU integration check compares independently CPU-trained and
GPU-trained models after the same F32 merge/export boundary, at the same 1e-5
bound. Its maximum scaled logit error is 5.00843671e-6 over all 8 x 32,000 logits.
Every parameter, gradient and Adam moment is separately compared after each
update at 1e-5; maximum observed scaled error is 4.63128629e-10. This does not
claim that unmerged quantized training agrees with merged inference at 1e-5.
Existing tiny export tests retain their 3e-5 absolute bound; existing real
inference backend parity retains its 0.001 bound. No existing tolerance changed.

## End-to-end cases

The random LLaMA fixture runs 100 full GPU updates on the deterministic byte
sequence, comparing all four persistent state arrays with CPU after each update.
Loss falls from 5.5934391 to 0.000674245763, with zero observed CPU state
difference in the default lifecycle test. A checkpoint after update 50 is restored into a new decoder and new GPU session;
its final checkpoint is byte-identical to uninterrupted training. Forward-only
evaluation leaves the complete checkpoint unchanged. The exported model reloads
through native inference and generates exactly `bcabcabc`.

Gemma fixtures cover ordinary attention, PLE/shared-KV and tied output weights,
each with full-weight and rank-2 LoRA training. Each trajectory runs two masked
cross-entropy updates and one paired DPO update with fixed reference logprobs.
Native inference of the exported fixtures passes the strict comparison.

TinyLlama 1.1B Chat Q4_K_M (SHA-256
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`)
uses rank-2 LoRA, eight actual tokens, two masked CE updates and one paired DPO
update. Losses are 8.51056767, 3.89509153 and 0.000178601491 respectively; CE and
DPO are different objectives and these three numbers are not one convergence
curve. Both exported models are 4,400,929,248-byte F32 GGUFs. The desktop GPU was
shared during these correctness probes; timings are not performance evidence.

## Validation

The 17 focused checks passed, including fixture setup, default/reference GPU
execution, three Gemma variants, ten injected launch failures, exact cancellation,
slice boundaries/accumulation, memory-budget rejection and retained ownership.
The 15-stage build/removal matrix passes with the following results (the four
skips are only unavailable ROCm/MLX hardware checks):

| Configuration | Passed | Skipped |
|---|---:|---:|
| C17 CPU | 57 | 4 |
| C17 CUDA/Vulkan | 154 | 4 |
| C23 CUDA/Vulkan | 154 | 4 |
| C23 ONNX | 58 | 4 |
| Clang ASan/UBSan CPU | 57 | 4 |
| optcpp physically absent | 57 | 4 |
| CUDA physically absent | 57 | 4 |
| CUTLASS/CuTe physically absent | 149 | 4 |

All 11 frontend tests and the production frontend build also pass. The first
C23 run encountered `Memory allocation failure` in the existing cuBLAS legacy
layout test. It is preserved as `matrix/build-cuda-c23-tests.log`. After stopping
Wallpaper Engine and NVIDIA overlays, the unchanged full suite passed in
`matrix/build-cuda-c23-tests-1.log`. This is consistent with resource pressure,
not proof of a kernel defect or a controlled performance result. The stopped
process identities are recorded; OS, development and data-processing processes
were preserved.

The NSIS installer was built and extracted to an isolated directory on D:.
All bundled backend files matched their source bytes; the desktop executable
matched after the expected Tauri bundle marker change. Its native trainer passed
the CLI lifecycle suite, and eight authenticated/unauthenticated HTTP checks
covered native CUDA load, selection, repeated 725-token-prompt generation,
runtime reporting and unload. No CPU replay was reported. The artifact was not
installed and interactive UI behavior was not inspected.

All eight GPU sanitizer runs passed: memcheck, initcheck, synccheck and
racecheck each exercised the full random-model lifecycle and Gemma PLE
full-weight/LoRA trajectories. Each reported zero errors; racecheck reported
zero hazards and zero warnings. Results are audited against source and binary
hashes before this milestone is committed. Race checking runs with a
four-launch synchronization limit and two analysis workers, with no kernel
filters or skipped graph operations. Package correctness verification overlapped
part of race checking; neither run is used for performance claims.

## Provenance

Before graph integration, MIT-licensed llama.cpp/ggml revision
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e` (b10809) was studied at
`ggml/src/ggml-backend.cpp`: split planning (1370–1490), user input copies and
allocation dependencies (1654–1710), asynchronous split execution and explicit
observations (1785–1847), and allocation/compute separation (1996–2055).
The adopted techniques are preparation before execution, explicit observation
boundaries and ordered allocation reuse. The implementation is original Fyodor
C/CUDA; no upstream source or runtime object was copied into this graph layer.
Earlier kernel provenance remains in the individual primitive reports.

## Reproduction

Use the configured GCC C17/C23 Release builds with -O3 and strict warnings.
The evidence snapshot records exact configurations, source hashes and hardware.

```powershell
cmake -S backend -B build-cuda
cmake --build build-cuda --parallel 6
ctest --test-dir build-cuda -R training-graph --output-on-failure
build-cuda/nya-test-training-graph.exe --model .tools/tinyllama-q4_k_m.gguf D:/fyodor-validation/new-run/tinyllama
python backend/benchmarks/training-20260923/validate.py --output backend/benchmarks/training-graph-20261007/new-matrix
python backend/benchmarks/training-graph-20261007/sanitize.py --output backend/benchmarks/training-graph-20261007/new-sanitizers
```

Create the output parent before the real-model command; allow at least 9 GB for
both exports. Keep independent raw-log directories for reruns. The package
scripts build an NSIS artifact and verify extracted binaries, trainer lifecycle
and repeated authenticated CUDA HTTP generation; they do not install the app or
claim interactive UI validation.
