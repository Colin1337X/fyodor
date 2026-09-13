# Optional CUTLASS prefill

This directory is the only Fyodor host C++ boundary. Removing `CUDA/cutlass`
before configuring/building produces a CUTLASS-less C/NVRTC CUDA backend.
Removing the whole `CUDA` directory still produces the CPU/independent Vulkan
build. No ordinary build downloads CUTLASS or requires its compiler.

The adapter builds a separate `fyodor-cutlass.dll` / `libfyodor-cutlass.so` with
a versioned C ABI. The portable core remains GCC C17/C23. Model weights, F32 KV,
activation buffers and streams remain owned by Fyodor. CUTLASS receives existing
device addresses and enqueues on the same stream; it performs no allocation,
host transfer or explicit synchronization. The first supported kernel is dense
prefill GEMM using CUTLASS's 3xTF32 operator with F32 inputs/output/accumulation.
It is not the less accurate single-TF32 operator. Decode retains native kernels.

**Optional fallback accelerator.** cuBLAS retains priority when available;
otherwise an available CUTLASS bridge accelerates eligible prefill. Set
`NYA_CUDA_CUTLASS=1` to force it for comparison, or `=0` to disable it. Missing headers,
compiler or library retain the normal CUDA paths. Unsupported architectures
(pre-SM80) and shapes retain the existing matrix implementation. Execution errors
are reported and trigger Fyodor's complete-prefix CPU recovery. `NYA_CUDA_REFERENCE=1`
always bypasses CUTLASS. No accuracy threshold is relaxed for this adapter.

## Source and build

Validated source: NVIDIA CUTLASS **v4.5.1**, commit
`2e602843e75100d0e03934efb386b3e1e35d7907` (BSD-3-Clause, upstream LICENSE retained).
The ignored `vendor` directory contains unmodified upstream dependencies, not
generated Fyodor code. Obtain the pinned source explicitly:

```sh
git clone --depth 1 --branch v4.5.1 --filter=blob:none --sparse https://github.com/NVIDIA/cutlass.git CUDA/cutlass/vendor/cutlass
git -C CUDA/cutlass/vendor/cutlass sparse-checkout set include
git -C CUDA/cutlass/vendor/cutlass rev-parse HEAD
```

With an installed CUDA toolkit and compatible host C++ compiler:

```sh
cmake -S CUDA/cutlass -B build-cutlass -DCMAKE_BUILD_TYPE=Release
cmake --build build-cutlass --config Release
```

On Windows run this in an x64 Visual Studio developer terminal; use `-G "NMake
Makefiles"` or Ninja. This compiler builds only the optional bridge. Fyodor's
GCC/MinGW core build remains separate. CMake's MinGW/MSYS generators do not
support CUDA C++ compilation, so their normal builds only enable the C loader.
Other supported generators can build the bridge as a subdirectory automatically
when headers and compiler are present. `NYA_BUILD_CUTLASS=OFF` disables compiling
the bridge; `NYA_ENABLE_CUTLASS=OFF` disables the C adapter in a full Fyodor build.
An external checkout can be selected with `NYA_CUTLASS_ROOT`.

The default target is SM80 machine code plus forward-compatible compute-80 PTX.
Override `NYA_CUTLASS_ARCHITECTURES` for known hardware, e.g. `120` for the tested
RTX 5060 Ti. Native CUDA/NVRTC still selects its own compatible architecture.
`NYA_CUTLASS_TILE` selects one of the four documented CMake tile shapes; the
default `64x64x32` was the fastest tested CUTLASS candidate on this device.

`cmake --build build-cuda --target verify-without-cutlass` configures, builds and
runs the suite in a fresh source copy where `CUDA/cutlass` is physically absent.
The existing `verify-without-cuda` target independently tests removal of all CUDA.

## Run

PowerShell example (the DLL path may be any explicit absolute path):

```powershell
$env:NYA_CUDA_CUTLASS='1'
$env:NYA_CUDA_CUTLASS_LIBRARY=(Resolve-Path build-cutlass/fyodor-cutlass.dll).Path
$env:NYA_CUDA_BLAS='0' # validate the CUTLASS + native path without cuBLAS
build-cuda/fyodor-bench.exe -m MODEL.gguf -b cuda -p 512 -n 128 -r 9 --warmup 3 --json
```

Without an explicit library path, Windows looks beside the executable; Linux
uses the configured dynamic-loader search path. The loader does not search the
Windows working directory. `NYA_CUDA_CUTLASS=0` disables the optional bridge.
`NYA_CUDA_DEBUG=1` reports selection. JSON reports `cutlass-3xtf32` as the selected
prefill implementation and actual `cutlass_matmul_calls` per workload. Small or
unaligned jobs may still use native/cuBLAS fallback; inspect the call count.

For Tauri packaging, set `NYA_CUDA_CUTLASS_LIBRARY` when running
`npm --prefix frontend run prepare:sidecar`. The script copies the DLL and the
upstream license produced beside it by CMake. Omitting the override removes a
previous generated CUTLASS copy. cuBLAS retains priority in packages that include
both libraries; the native CUDA path remains available without either library.

Compressed F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K weights use Fyodor's existing decoder
and bounded F32 expansion scratch. This first bridge does not fuse quantized
loads into CUTLASS, implement GPU training, or change KV precision. Scratch is
accounted in existing `external_matmul_bytes`/`scratch_bytes` fields. The shared
optional-matrix scratch policy still reserves the existing 4 MiB workspace even
when only CUTLASS is enabled; CUTLASS itself needs no extra workspace here.
