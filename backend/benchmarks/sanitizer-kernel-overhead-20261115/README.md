# Sanitizer binary-patching overhead isolation

The dominant slowdown was isolated to Compute Sanitizer binary patching. This is a diagnostic finding, not an inference optimization or a claim that the underlying NVIDIA implementation defect has been identified. All runs are terminal; no 8K validation was restarted.

## Observations

Original Fyodor Q4 kernels use columns=3840, rows=15360, deterministic synthetic Q4_K weights, and a native C float64 oracle with its existing bounds. The scalar and warp kernels produce one vector; the paired kernel produces four. Each condition has one warmup and five individually event-timed launches per kernel. Whole-device synchronization completes uploads before timing. Every output is checked. An unused allocation is varied through 0, 8 GiB, 0 to separate allocation footprint from kernel execution. No CUDA graphs or model lifecycle are involved.

Representative first-block means from the offline-CUBIN control (milliseconds per kernel):

| Kernel | Normal | Memcheck no-match | Patching disabled | Patched repeat | Unpatched repeat |
|---|---:|---:|---:|---:|---:|
| nya_matvec | 0.4728 | 113.1564 | 0.4746 | 113.1690 | 0.4744 |
| nya_dot_12 | 0.1168 | 20.1478 | 0.1200 | 20.1832 | 0.1204 |
| nya_q4_pair_rows | 0.1175 | 15.4019 | 0.1215 | 15.4031 | 0.1246 |

The PTX-loaded run also slows by roughly 131-239x. Offline compilation of exactly the same PTX does not remove it. Adding an unused 8 GiB allocation does not materially change these kernel times. A nonmatching substring filter and an explicit exclude filter retain the slowdown. Switching binary patching off and back on reproduces the large difference; retained standard deviations and all repetitions are in verification.json and the underlying run JSON files. These repeated microkernel measurements do not have interval GPU-activity coverage and are NOT accepted decode scores.

A separate real-model prefix-5 control, using the same verbose C23 executable and numerical bound as the prior comparison, took 13.1761 seconds with patching disabled, versus 168.4620 seconds with no-match filtering and patching enabled, and 8.1177 seconds without sanitizer. Both reset/rewind comparisons and progress inventories pass. This connects the kernel finding to the model diagnostic, while residual startup/tool overhead remains. The result is under ../lifecycle-progress-20261115/OverheadNoPatching5. Do not extrapolate the short whole-run ratio to long decode.

## Scope and safety evidence

`NV_COMPUTE_SANITIZER_BINARY_PATCHING=0` is used ONLY as a diagnostic control, scoped to the child environment. It disables instrumentation and is not a safety-validation workaround. Command manifests explicitly mark safety_evidence=false. The previous filter selectivity fixture establishes that filtering selects errors, but the large cost persists for excluded kernels in these measurements. The precise reason that excluded kernels retain binary-patching overhead has not been identified. Do not claim that an upstream bug or its implementation is proven.

NVIDIA documents the variable as a debugging control over user-kernel instrumentation: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html . Installed tool: compute-sanitizer 2025.4.1.0, CUDA sanitizer distribution 13.1.118, RTX 5060 Ti SM120, driver 617.42. No system-wide environment or driver setting was changed.

No-match, exclude, or unpatched zero-error summaries do not prove candidate safety. None of these runs satisfies the interrupted 8K candidate gate. Numerical tolerances remain unchanged. Accepted short/long decode scores remain 85.61025294% / 106.8707255% of llama.cpp, versus the required 125%+ for both.

## Reproduction and provenance

The harness is original diagnostic code using existing Fyodor kernels and the existing native C oracle. Production sources are unchanged. probe.py imports the existing q4-pair-rows common module; common_cubin.py is a diagnostic copy with only module loading changed. It compiles the original source through NVRTC, requires its PTX to match the exported bytes exactly, and loads offline CUBIN instead. Every run hashes its source, oracle and applicable offline inputs. verify.py validates nine runs, 81 case records, 405 GPU samples, the model control, and all retained hashes without initializing CUDA.

From the engineering checkout, export once with `python -B backend/benchmarks/sanitizer-kernel-overhead-20261115/export_ptx.py`, then compile with:

```
.tools/cuda-sdk/cuda_nvcc-windows-x86_64-13.1.115-archive/bin/ptxas.exe -arch=sm_120 -O3 build-cuda/sanitizer-kernel-overhead-20261115/kernels.ptx -o build-cuda/sanitizer-kernel-overhead-20261115/kernels.cubin
```

Use fresh labels; all harnesses refuse to overwrite evidence. Run probe.py or probe_cubin.py directly for baseline. For filtered runs, prefix the command with compute-sanitizer `--tool memcheck --kernel-name regex=fyodor_deliberately_no_matching_kernel_20261115 --error-exitcode 99`. run_control.py accepts `patched LABEL` or `unpatched LABEL`, runs probe_cubin.py, and saves the exact command and scoped override. run_filter_control.py accepts `substring LABEL` or `exclude LABEL` and explicitly clears the patching override. All runs in this report were sequential, and GPU idle snapshots bracketed the experiments.
