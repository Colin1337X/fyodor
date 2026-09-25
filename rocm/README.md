# Optional ROCm C adapter

This directory contains a dynamically loaded HIP/rocBLAS provider for Fyodor.
Delete `/rocm` or configure `-DNYA_ENABLE_ROCM=OFF` to omit it. CPU, Vulkan,
CUDA, MLX, training and the server remain independent. All project source here
is C; there is no HIP C++ compilation or mandatory ROCm SDK dependency.

Install the official ROCm runtime with support for your AMD GPU and matching
rocBLAS kernel packages. Set absolute library paths when they are not in the
normal loader location:

```powershell
$env:NYA_ROCM_HIP_LIBRARY='C:\ROCm\bin\amdhip64.dll'
$env:NYA_ROCM_BLAS_LIBRARY='C:\ROCm\bin\rocblas.dll'
$env:NYA_ROCM_DEVICE='0'
.\build\fyodor-bench.exe -m model.gguf -b rocm -p 512 -n 128 -r 5 --warmup 2 --json
```

On Linux, defaults are `libamdhip64.so` and `librocblas.so`; the same environment
variables can identify absolute `.so` paths. Windows defaults search beside the
executable using restricted DLL search flags, never the current directory or
ambient PATH. The HIP/rocBLAS runtime and their transitive libraries/data must
be a compatible installation. Fyodor does not bundle or download them.

The adapter owns a HIP stream, a rocBLAS handle, persistent input/output
buffers and a bounded F32 weight cache. Quantized GGUF data is decoded by the
portable C decoder, uploaded once while retained, then passed to SGEMM with
transpose flags matching Fyodor's row-major weights and batch-major vectors.
The selected device is saved/restored per call so serialized server requests
can migrate between threads. Errors include operation and numeric status,
deactivate the context, and leave CPU fallback available. Initialization runs
a matrix probe, so an enumerated GPU with missing BLAS kernels is rejected.

`NYA_ROCM_CACHE_MIB` optionally caps the memory budget. Free VRAM determines
the default, reserving 10% or at least 256 MiB. Host weights must remain
immutable and alive until context destruction. All operations synchronize
before returning; eviction never frees an in-flight cache entry.

**Current limitation:** this is an accelerator-assisted host transformer, not
a resident ROCm graph. KV, attention and activations between projections remain
on the CPU. Weights expand to F32, so larger models may exceed the cache even
when their GGUF file fits VRAM. There are no native HIP quantized kernels,
GPU training, layer offload, or ROCm performance claims yet.

ABI declarations were checked against official sources:

- [HIP](https://github.com/ROCm/HIP/tree/bb52be79fb0ec8664576b409f978c36e9043e70b/include/hip)
- [rocBLAS](https://github.com/ROCm/rocBLAS/tree/defce200a69e5346eeadd7ac1e199238758add61/library/include)

CTest `vendor-rocm-*` uses explicitly named host mock libraries for shape,
quantized oracle, Gemma, API, cache, failure and ownership tests. It is not GPU
validation. `hardware-rocm-quant` uses real runtime discovery and skips with
code 77 if unavailable. Run it plus real-model logit checks on AMD hardware
before treating the provider as production validated.
