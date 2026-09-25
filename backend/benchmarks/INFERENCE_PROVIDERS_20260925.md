# Optional inference providers — September 25, 2026

This commit records the previously uncommitted inference work separately from
the native training changes. ROCm and MLX are detachable C adapters behind the
existing Fyodor compute interface. Runtime libraries are dynamically loaded;
their absence preserves native CPU execution. They support matrix-vector and
batched matrix multiplication with persistent F32 weight caches. CPU/Vulkan/CUDA
remain independent, and general-purpose C++ is not required.

The host prefill graph now dispatches batched projections by capability rather
than a CPU-only name check. Runtime/API metadata distinguishes compiled support,
successful selection, CPU fallback and accelerator-assisted execution. Explicit
CPU selection releases a failed accelerator context. Benchmarks and the desktop
recognize both providers and report retained memory after cache population.

These providers are **host-controlled matrix accelerators**. Attention, KV,
normalization and inter-projection activations remain on the host. Quantized
weights expand to F32; allocator/workspace overhead may exceed reported retained
bytes. No AMD/Apple throughput, resident graph or GPU training claim is made.
See [ROCm setup and limitations](../../rocm/README.md),
[MLX setup and limitations](../../mlx/README.md) and the
[native C / optional C++ policy](../CPP.MD). ABI provenance and upstream licenses
are included with each adapter and its mock test declarations.

The unchanged production source passed the complete
[15-stage working-tree matrix](training-execution-20260925/validation.json):
C17 CPU, C17/C23 CUDA+Vulkan, ONNX, ASan+UBSan, physical optcpp/CUDA/CUTLASS removal,
frontend tests and production build. The CPU configuration passed 46 tests with
four unavailable ROCm/MLX hardware tests skipped; both CUDA configurations passed
71 with the same four skips. Mock libraries exercise all seven supported weight
formats, batched tails, model logits, API switching, cache budgets, missing
libraries/symbols, cross-thread serialized use and failure cleanup. Mock success
does not establish real vendor hardware compatibility.

The [verified Windows package](training-execution-20260925/desktop-package.json)
contains this inference state and the current trainer. The package is built and
extracted for resource/hash checks and native process tests; it has not been
installed. Native interactive UI testing, Linux/macOS execution and actual
ROCm/MLX GPU validation remain outstanding. The earlier
[matched TinyLlama inference measurements](TRAINING_20260923.md) remain the
performance baseline; committing these adapters introduces no new speed claim.
