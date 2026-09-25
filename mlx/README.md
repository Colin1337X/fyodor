# Optional MLX C adapter

This directory implements Fyodor's own compute interface using the optional
[MLX C library](https://github.com/ml-explore/mlx-c), normally backed by Metal
on Apple silicon. Delete `/mlx` or set `-DNYA_ENABLE_MLX=OFF` to omit it.
CPU, Vulkan, CUDA, ROCm, training and server code remain independent. The
adapter is C17/C23; building Fyodor does not build MLX or require C++.

Install/build a shared MLX C library with GPU support following its upstream
instructions, then provide its absolute path if needed:

```sh
export NYA_MLX_LIBRARY=/absolute/path/libmlxc.dylib
fyodor-bench -m model.gguf -b mlx -p 512 -n 128 -r 5 --warmup 2 --json
```

Default names are `libmlxc.dylib` on macOS, `libmlxc.so` on other Unix systems
and `mlxc.dll` on Windows. Windows only searches beside the executable by
default; transitive runtime dependencies must also be installed correctly.
A compiled loader on Windows is not evidence that an MLX GPU build is present.
There is no automatic download or bundled MLX dependency.

The adapter uses explicit GPU and CPU devices/streams, F32 matrix arrays and
an LRU cache of transposed contiguous weights. GGUF's seven supported storage
types are decoded through Fyodor's C decoder. Matrix results are explicitly
copied to a CPU stream before reading a host pointer. Streams use MLX's
`mlx_stream_new_thread_unsafe` API, allowing serialized calls to migrate between
server workers. Concurrent calls on one Fyodor context are unsupported.

The loaded module is reference-counted across contexts. MLX C's default error
callback terminates the process, so Fyodor replaces it with a nonfatal logging
handler before initialization. This handler is process-global within that MLX
module; embedding applications sharing MLX C must coordinate error-handler
ownership. Errors deactivate the affected context, which falls back to CPU.
Selection requires an available GPU and a successful matrix probe. Missing
required symbols reject initialization instead of calling a partial API.

`NYA_MLX_CACHE_MIB` optionally caps the default budget (MLX limit minus current
active/cache usage, then a 10%/256 MiB safety reserve). The cache accounts for
retained F32 weight arrays. Temporary matrices and allocator/workspace overhead
are not fully represented in Fyodor's retained-memory counters; use MLX/device
telemetry for total memory. Host arrays must stay immutable until destruction.

**Current limitation:** the transformer is host-controlled with a synchronized
matrix call per projection. Norm, RoPE, attention, activations and KV do not
remain MLX-resident. There is no MLX quantized matmul, resident graph, GPU
training or measured Apple GPU performance yet. Expanding GGUF weights to F32
can make memory usage substantially larger than the model file.

The by-value opaque-handle ABI and required functions were checked against
[MLX C 0.6.0 source, commit c74db53](https://github.com/ml-explore/mlx-c/tree/c74db5307cc8ce122f48d97ef951b30578674e7f/mlx/c).
Older builds without required stream/memory functions are unavailable.
`vendor-mlx-*` tests use an explicit host mock shared library to validate ABI,
quantized oracles, Gemma, API switches, cross-thread use, cache and failure
cleanup. `hardware-mlx-quant` uses real MLX and skips if unavailable. Apple
hardware logits and throughput remain to be validated.
