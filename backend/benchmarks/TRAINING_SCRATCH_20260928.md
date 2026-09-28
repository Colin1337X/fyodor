# Reusable CUDA training scratch scopes — September 28, 2026

The private resident training device previously retained every buffer until
device destruction. A repeated graph would eventually exhaust its arena or
4,096 descriptors. It now supports one non-nesting scratch scope: allocations
before the scope survive, while ending it invalidates subsequent handles and
rewinds the arena and descriptor count. Buffer and scope tokens share a global,
never-reused identifier namespace, rejecting expired, foreign and wrong-kind
handles even after physical storage is reused.

Ending a scope adds no CUDA operation or synchronization. All queued consumers
and subsequent allocation zeroing run on the device's single serialized stream;
the arena itself stays allocated. Callers must still finish explicitly to detect
asynchronous failures. A poisoned device cannot begin or end a scope. CPU callers
must serialize operations. Persistent weights, gradients and optimizer state
must be allocated before the scratch scope.

This is original Fyodor C implementation. It extends the internal device API,
not the public training graph. The CLI and UI still execute training on the CPU;
resident non-matrix operations, loss, optimizer and graph integration remain
unfinished. No end-to-end GPU training throughput improvement is claimed.

## Initial evidence

The GCC C17 CUDA build and all four training-device tests passed; see
[the test log](training-scratch-20260928/initial-tests.log). New coverage includes
5,000 successive scopes with queued zero, forward and gradient accumulation,
variable allocation lengths, stale handles, invalid scope tokens, descriptor
exhaustion and recovery, active-scope destruction and injected launch failure.
The repeated-work fixture retains three buffers occupying 516 bytes, reaches
1,036 bytes peak arena usage, adds zero transfers or fences inside the loop, and
produces exactly 60,000 accumulated gradient units.

The real TinyLlama Q4_K_M matrix harness covers first-layer query, feed-forward
down and up matrices at 8 and 64 tokens. It checks 64 spread elements per
operation against a double-precision oracle. Existing synthetic tests check all
elements, all seven supported weight formats and CPU autograd agreement.

| Harness mode | Arena capacity | Peak arena bytes | Final live buffers | Scope resets |
| --- | ---: | ---: | ---: | ---: |
| Retain six jobs | 268,435,456 | 265,928,704 | 36 | 0 |
| Reuse, forward order | 67,108,864 | 59,531,264 | 0 | 6 |
| Reuse, reverse order | 67,108,864 | 59,531,264 | 0 | 6 |

All three runs pass and report identical work/transfer counts: 2,826 launches,
18 uploads totaling 42,221,568 bytes, and 1,152 downloads totaling 4,608 bytes.
The comparison retires each complete matrix job, including its weights and
gradient buffers; it is a storage-lifetime demonstration, not a full persistent
model memory estimate. Capacity is the actual reserved device arena; peak usage
is its allocation high-water mark and excludes CUDA context/module overhead.

Raw outputs and counters are preserved in
[the evidence directory](training-scratch-20260928). Timings in these outputs
are not accepted performance measurements: desktop GPU activity was not isolated.
The harness uses two warmups and five timed batches of twenty calls, with
`NYA_CUDA_BLAS=0`, `NYA_CUDA_CUTLASS=0`, and `NYA_COMPUTE=cpu` (CPU model loading;
the training device explicitly requires CUDA). Reproduce with:

```powershell
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf forward
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf reuse-forward
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf reuse-reverse
```

Broader configuration and GPU sanitizer validation are pending at this checkpoint.
