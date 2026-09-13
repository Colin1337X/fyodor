# Matching F32 KV workloads against llama.cpp

`llama_capi_bench.c` is a separate comparison executable linked to upstream
llama.cpp shared libraries. It is never linked into Fyodor. b10809's llama-bench
parser excludes F32 KV, while its C API accepts it; this harness avoids forcing
Fyodor to change cache precision merely to match that CLI.

Validated upstream commit: `5266f24da75dc449bd56cbed7addb9c8e4a6a73e` (b10809).
Use headers and shared libraries from the **same commit/build**: upstream C
structure layouts can change. No headers/libraries are downloaded automatically.

Windows GCC example with the reference paths used in this workspace:

```powershell
$env:PATH=(Resolve-Path .tools/w64devkit/bin).Path+';'+$env:PATH
gcc -std=c17 -O3 -Wall -Wextra -Werror `
  -I.tools/reference-llama-b10809/include `
  -I.tools/reference-llama-b10809/ggml/include `
  backend/benchmarks/llama_capi_bench.c `
  .tools/llama-b10809-cuda/llama.dll `
  .tools/llama-b10809-cuda/ggml.dll `
  .tools/llama-b10809-cuda/ggml-base.dll `
  -o .tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe
```

Keep the executable with upstream's backend DLLs so its normal backend discovery
can find CUDA. Linux uses the equivalent matching include/library directories and
links `-llama -lggml -lggml-base -lm`, with the appropriate shared-library path.
The C source supports CPU and CUDA. Vulkan comparison through this harness has
not been implemented; the existing `compare.py` still drives official llama-bench.

```powershell
python backend/benchmarks/compare_capi.py `
  --fyodor build-cuda/fyodor-bench.exe `
  --llama-capi .tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe `
  --llama-commit 5266f24da75dc449bd56cbed7addb9c8e4a6a73e `
  --model .tools/tinyllama-q4_k_m.gguf `
  --repetitions 9 --warmup 3 --batch 512 --threads 6 `
  --output comparison-short
```

Add `--before OLD_FYODOR_BENCH.exe` for a preserved implementation baseline. Use
`--prompt 0 --context 1024` for long-context decode. Output directories must be
new so prior evidence cannot be overwritten. Each engine is run serially, then
in reversed order. Commands, hashes, stderr and telemetry accompany raw JSON.
The helper rejects a changed Fyodor provider/residency or missing upstream full
GPU offload. The standalone C harness checks every final logit for finiteness
outside its timed region; it does not assert cross-engine numerical identity.

Both engines use deterministic IDs `(i*15485863+17)%vocabulary`, F32 keys/values,
matching requested capacity, batch size and thread count. Model loading,
tokenization, sampling, cache reset, prefix preparation and complete-workload
warmups are excluded. Prefill produces final-position logits; decode produces
CPU-visible logits at every step and synchronizes accordingly. Upstream flash
attention is disabled. JSON records any upstream allocator rounding of context.
Rates are arithmetic means of individual runs, with sample standard deviation.

Internal arithmetic, kernel choices and intermediate formats remain properties
of each engine. Record both engines' build/version and environment. Stop competing
GPU applications before comparison when authorized, but report remaining desktop
activity honestly. Device telemetry includes all processes; it is not per-process
VRAM accounting or proof of headless isolation. Never compare a contaminated run
to an isolated run and attribute the entire difference to code.
