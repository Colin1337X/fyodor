# Native C training executor — September 25, 2026

This update targets the measured CPU training bottleneck: mapped matrix forward
and backward, followed by dense matrix operations. The serial TinyLlama profile
spent roughly 1,065 ms/update in mapped forward and 558 ms in mapped backward
after its first update. See [baseline profiling](training-execution-20260925/baseline-profile.log).
The implementation remains native C17/C23 and does not use PyTorch, OpenMP or an
optional C++ runtime. GPU training remains unfinished.

## Execution and numerical contract

An opaque, reusable executor owns up to 63 sleeping workers; the controlling
thread participates as the remaining worker. Construction either creates the
requested pool or fails and joins every successfully created worker. A mutex,
condition variables and a completion barrier bound the lifetime of borrowed
task arguments. Independent executors can run concurrently. A single executor
requires one controlling thread; concurrent operations on graphs borrowing it
are unsupported. Destroy borrowing graphs before destroying their executor.

The existing public graph constructor stays serial. The new
`nya_train_graph_create_with_executor` borrows an executor without spawning
threads per graph. CLI runs share one executor across microbatches and DPO
reference evaluation. `--threads 0` defaults to automatic selection, capped at
64 (Windows physical cores; POSIX online processors). `1` selects serial
execution; explicit values 1–64 are accepted. The desktop exposes the setting,
and progress/CSV reports configured `cpu_threads`, not measured utilization.

Only matrix work with at least 1,048,576 row × column × token products dispatches
workers. Dense and mapped forward partition output rows; dense backward
partitions input-gradient tokens and weight-gradient rows. Mapped backward
partitions input columns at quantization-block boundaries, using disjoint parts
of the same budgeted decode scratch. Each output retains its original reduction
order. No atomics or unordered floating-point gradient reduction are needed.
Aliased dense gradients and mapped backward without enough scratch retain the
serial fallback. Attention, normalization, activations and AdamW remain serial.

The graph gains one borrowed pointer (8 bytes on this host), without additional
graph allocations. Worker stacks and executor metadata are outside graph
budgets. Changing worker count is intentionally excluded from checkpoint
identity: identical inputs must retain identical parameter trajectories. No
existing numerical tolerance was relaxed.

## Paired measurements

Windows 11 build 26200, Ryzen 5 9600X with six processors exposed, 32 GiB RAM;
GCC 16.2 Release `-O3 -DNDEBUG`, C17 CPU-only trainer. The RTX 5060 Ti is not used
for training. Baseline is the preserved pre-executor trainer, including earlier
kernel, accumulation, safe-stop and Unicode fixes. Measurements were serial,
alternating before/after order between process pairs, without concurrent builds
or tests. Shared-desktop scheduling and thermal conditions are not isolated.
Every sample is retained; no known contaminated sample was removed.

Means ± sample standard deviations below are tokens/sec across processes.
Rates exclude startup, checkpoint and export; raw metadata also records total
process time. Synthetic models start from identical seeded random weights and
use F32 training. All compared losses, checkpoints and GGUFs match exactly.

| Workload | Before | After | Ratio |
|---|---:|---:|---:|
| Small: width 64, FF 176, 2 layers, context 32 | 9,754.93 ± 171.53 | 8,829.36 ± 1,704.84 | 0.91× |
| Medium: width 256, FF 704, 2 layers, context 128 | 1,529.95 ± 2.26 | 2,637.55 ± 15.11 | 1.72× |
| Long: width 128, FF 352, 2 layers, context 512 | 3,390.01 ± 29.48 | 4,593.62 ± 34.36 | 1.36× |
| Tail: width 68, FF 183, 1 layer, context 31 | 14,234.74 ± 164.51 | 14,495.52 ± 54.07 | 1.02× |
| TinyLlama Q4_K_M, rank 2, context 8 | 4.539 ± 0.010 | 11.690 ± 0.031 | 2.58× |

[Synthetic evidence](training-execution-20260925/dense-final/metadata.json):
four processes per version and shape, 14 updates, first two excluded from
timing. [Real-model evidence](training-execution-20260925/real-final/metadata.json):
two processes per version, four updates, first excluded from timing. All use
accumulation 1. The real model retains mapped quantized base weights and F32
adapters. These short runs establish execution equivalence and throughput,
not useful model quality or long-run convergence.

The short small-model batch includes an unexplained 6,280 tokens/sec after
sample. It is preserved in the table, with no unsupported contamination claim.
A predeclared longer window (100 updates, 10 warmup, eight process pairs) gave
9,640.27 ± 138.04 before and 9,583.24 ± 426.00 after, or 0.994×. The tail shape
gave 14,294.10 ± 214.30 and 14,604.23 ± 75.66. Small work stays below the dispatch
threshold; no small-model speedup is claimed. See [all sustained samples](training-execution-20260925/small-sustained/metadata.json).
The earlier [pilot](training-execution-20260925/dense-paired/metadata.json) and
[instrumented after profile](training-execution-20260925/after-profile.log)
are diagnostic records, not substitutes for final paired measurements.

## Exact resume and memory

A real checkpoint produced by the old serial trainer at update 3 resumed with
three workers for update 4. Both checkpoint and GGUF exactly match continuous
four-update runs before and after the executor change:

- Base SHA-256: `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`
- Export SHA-256: `1f9f5d9a4198399f36b4063b7c22a057182ea87c940a697a58869f807fb48005`
- Checkpoint SHA-256: `3a1a21efdade6dc4d4afadd52252df1ac15ca00718bebd4e4e6ca2e3e94423ca`

The [resume/resource driver](training-execution-20260925/real_resume.py) observed
812,249,088 bytes OS peak working set, including mapped/shared pages, and
180,019,200 bytes sampled peak private commit. It queried Windows process
memory every 50 ms across initialization, resume, training and export (413
samples). Private commit is not physical RSS; this is not a paired memory
comparison or VRAM measurement. The component graph budget was 512 MiB and
actual graph use 143,521,216 bytes. The 4.4 GB streamed F32 export was hashed
and removed by its owning driver; checkpoint and [measurement record](training-execution-20260925/real-resume.json)
remain. The component budget is not a process-memory ceiling.

## Correctness coverage

Native tests compare serial with 2, 3 and 6 workers, including empty worker
partitions, odd row/token tails, and F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K mapped
weights. They require bitwise matching forward output, losses, gradients and
parameters across three AdamW updates. Dense gradients, forced no-scratch
fallback, allocation-budget failure/reuse, pool lifecycle and simultaneous
independent executors are covered. Overflow partition checks include SIZE_MAX.
Existing finite-difference and independent gradient oracles remain in place.

CLI coverage resumes a nontrivial random model from one worker to three and
compares it against six-worker continuous training. Invalid counts, accumulation,
all four modes, Unicode paths and graceful stop remain covered. Rust validates
serialization/defaults/ranges and actually launches the native trainer via
`FYODOR_TEST_TRAINER`; [both tests passed](training-execution-20260925/rust-tests.log).

All 15 stages of the [working-tree matrix](training-execution-20260925/validation.json)
passed: C17 CPU 46 passed/4 unavailable-hardware skips; CUDA+Vulkan C17 and C23
71/4 each; ONNX 47/4; ASan+UBSan 46/4; physical optcpp and CUDA removal 46/4
each, and CUTLASS removal 66/4. Frontend tests passed 10/10 and its production
build succeeded. The four skips are actual ROCm/MLX hardware tests; mocked
provider tests do not establish AMD/Apple hardware support. This matrix includes
pre-existing provider work which is kept separate from the training commit.

The exact staged source was also exported independently of those provider edits:
[CPU 26/26](training-execution-20260925/staged-tests.log),
[C23 CUDA+Vulkan 51/51](training-execution-20260925/staged-cuda-tests.log),
[physical optcpp removal 26/26](training-execution-20260925/staged-without-optcpp.log),
and [frontend 9/9](training-execution-20260925/staged-frontend.log) passed. The
tracked trainer binary is built from this isolated source. These tests use
fresh build directories and do not borrow provider objects from the dirty tree.

The Windows NSIS installer was rebuilt from the complete working tree. Package
verification compares every backend resource byte-for-byte and permits only
Tauri's single expected UNK-to-NSS marker change in the desktop executable.
The packaged trainer runs the 54-process stop suite, 24-process Unicode suite
and CLI workflows including cross-worker exact resume. The initial package
CLI driver reached the comparisons but failed because the system CMake lacked
its module directory; [that log is retained](training-execution-20260925/packaged-cli-system-cmake-failed.log).
The verifier uses the same complete bundled CMake as the build matrix. No
installation or interactive UI verification is claimed.

## Reproduction and provenance

Run performance experiments before validation, without simultaneous builds:

```powershell
python backend/benchmarks/train_compare.py --before .tools/fyodor-train-before-executor.exe --after build-cpu/fyodor-train.exe --output new-dense-evidence
python backend/benchmarks/train_compare.py --before .tools/fyodor-train-before-executor.exe --after build-cpu/fyodor-train.exe --output new-real-evidence --base .tools/tinyllama-q4_k_m.gguf --repetitions 2 --steps 4 --warmup 1
python backend/benchmarks/train_compare.py --before .tools/fyodor-train-before-executor.exe --after build-cpu/fyodor-train.exe --output new-small-evidence --workload small tail --repetitions 8 --steps 100 --warmup 10
python backend/benchmarks/training-20260923/validate.py --output new-validation-evidence
```

Metadata stores exact commands, binary/model/corpus hashes, all loss CSVs,
process order and environment. Outputs must use fresh directories.

This is original Fyodor C code using its existing native thread abstraction.
Source study covered PyTorch's [ParallelNative.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/ParallelNative.cpp)
(BSD-style license, main as inspected September 25) and
[ggml CPU scheduling](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cpu/ggml-cpu.c)
(MIT). Relevant ideas were a persistent pool, caller participation, bounded
grain size and completion barriers. No upstream source was copied and no new
third-party runtime dependency was introduced.

The larger project remains unfinished: GPU-resident training, evaluation,
schedulers, periodic atomic checkpoint publication and the documented inference
performance gaps remain separate work. POSIX executor code has not been run on
this Windows host; native interactive desktop validation is also not established
by the command-line and frontend tests.
