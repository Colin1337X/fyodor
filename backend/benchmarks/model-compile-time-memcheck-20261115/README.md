# Compiler-instrumented model memcheck diagnostics

Separate optimized strict C17/C23 diagnostic builds now load a CUDA 13.1 PTXAS module built with `-sanitize=memcheck`. The whole inference module is instrumented; the runtime memcheck command has no kernel filter, and no binary-patching-disable override is inherited. The existing lifecycle numerical bound, reset/rewind checks, phase logs, token rates and heartbeats are retained. Production sources and production artifacts are unchanged.

The diagnostic loader compares the runtime NVRTC PTX byte-for-byte to the offline PTX before loading the instrumented module. It rejects training, reference-math mode, and devices other than SM120 for this local experiment. These are diagnostic constraints, not new production backend restrictions. Existing allocations and model arithmetic are unchanged. The artifact manifest hashes both PTX and CUBIN, source and host binaries; each run rechecks its build manifest before and after execution.

## Completed checks

- Strict C17/C23 builds pass with the existing optimized command lines and warnings-as-errors.
- Prefix-5 real Gemma lifecycle: C23 34.6430 seconds, C17 27.4001 seconds, both original numerical/reset/rewind checks pass and unique zero-error memcheck summaries. Exact-PTX module markers are present. These are one run each, not comparable production throughput scores.
- Previous runtime-patched filtered C23 diagnostic took 168.4620 seconds. Compiler instrumentation substantially reduces this validation cost, while the new command is unfiltered. No speedup is attributed to the inference engine itself.
- Real Q4 microkernels under unfiltered compiler memcheck: approximately 11.11 / 1.58 / 2.41 ms for scalar / warp / paired kernels, versus approximately 113.16 / 20.15 / 15.40 ms with runtime patching and no-match filtering. All outputs pass the existing float64 oracle bounds.
- Two compiler-instrumented intentional global-write faults are detected (exit 99). Static and dynamic shared-write faults are detected under both runtime and compiler instrumentation (four further expected exit-99 cases). These validate those faults, not all possible sanitizer coverage.
- verify.py validates all result hashes and proof dependencies without starting CUDA. Its output explicitly leaves long_candidate_gate_satisfied=false.

Prefix 5 does not execute the long cached-query candidate. Longer prefixes, graph paths and cached-query execution still require validation. The old interrupted 8K result stays rejected. No accepted timing gate is bypassed. Numerical tolerances are unchanged. Short/long accepted decode remains 85.61025294% / 106.8707255% of llama.cpp, target at least 125% for both.

## Tool scope and limitations

NVIDIA added compiler memcheck in 2025.4 and documents matching compiler/sanitizer toolkit versions. This uses PTXAS 13.1.115 with sanitizer distribution 13.1.118 (tool 2025.4.1.0). The installed PTXAS help lists `-sanitize=memcheck`. Compiler instrumentation is for memcheck; racecheck/synccheck remain separate. Newer releases list shared-memory detection changes, and current documentation lists remaining shared-memory/hybrid/HMM limitations. Our deliberate fault checks establish detection only for the tested cases. All functions in the inference module are compiled together with instrumentation; no function-level hybrid scope is assumed.

Sources: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#compile-time-patching and https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html .

## Reproduction

Use build_v2.py in a fresh build directory, then `python -B backend/benchmarks/model-compile-time-memcheck-20261115/run.py c23 5 NewLabel --memcheck`. Labels must be unique. The runner refuses non-memcheck execution and clears the patching-disable override in the child only. Log redirection preserves heartbeat lines; child stdout/stderr are retained separately. All paths in this diagnostic build are local to the engineering checkout.

The initial build.py failed because NVRTC could not locate its builtins DLL; its logs and incomplete directory are retained. build_v2.py explicitly loads the matching builtins DLL and uses a fresh directory. The shared-fault PTX initially placed an extern shared declaration inside a function; PTXAS rejected it before execution. The rejected source is retained alongside the corrected module-scope declaration. Kernel controls and fault evidence live in ../sanitizer-kernel-overhead-20261115.
