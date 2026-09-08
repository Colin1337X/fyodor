# Fyodor backend / nya 0.3

Fyodor's backend is an independent C model engine with an authenticated HTTP server. The desktop frontend talks directly to this process; Tauri only needs to launch it, read its readiness line, and manage its lifetime. The engine library can also be used without opening a socket.

The default build uses C17, libc, and native operating-system APIs. C23 is selectable. Vulkan is an independent **C** provider. CUDA implementation lives entirely in the repository's **`/CUDA` directory**, with a C host adapter and CUDA device kernels compiled through NVRTC at runtime. CPU/Vulkan builds need no CUDA SDK, `nvcc`, C++ compiler, or CUDA directory. Even `NYA_ENABLE_CUDA=ON` falls back to a CPU/Vulkan build when `/CUDA` or its required headers are absent. No backend code uses Python, Rust, JavaScript, a web framework, or a third-party JSON library.

## What works

| Capability | Default C build | Vulkan build |
|---|---|---|
| GGUF v2/v3 container inspection | Yes | Yes |
| SafeTensors container inspection | Yes | Yes |
| ONNX protobuf envelope inspection | Yes | Yes |
| Dense LLaMA and Gemma 4 GGUF text generation | F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q6_K kernels | Same CPU kernels, optional F32 matrix-vector dispatch to Vulkan |
| Gemma 4 PLE, shared KV, dense/shared + routed MoE blocks | Native C; exact tensor shapes required | Same architecture support |
| Ordinary draft and Gemma 4 MTP speculative decoding | Target verification and rejection sampling | Same; verification currently uses scalar steps |
| Gemma 4 Unified image/audio projection | Native `gemma4uv` / `gemma4ua` GGUF projectors | Optional F32 matrix-vector acceleration |
| Projected multimodal prompt embeddings | C generation API; local image-block attention | Same; HTTP generation integration pending |
| Autograd, LoRA/full-weight SFT/CPT/DPO, random-weight pretraining | C library and `fyodor-train`; dense LLaMA/Gemma 4 training | CPU training; GPU backward kernels pending |
| Generic ONNX graph execution | Optional external adapter, disabled by default | Same independent option |
| Native REST plus OpenAI/Anthropic text compatibility | Model management, generation, tokenization; documented non-streaming subset | Same |

Inspection and execution are separate capabilities. A valid model file can be loaded for inspection even when its architecture or quantization cannot execute. Check `generation_supported`, `draft_supported`, `inference_supported`, `generation_error`, and `execution_error` in model responses. `inference_supported` describes named-tensor execution through a native projector or the optional ONNX adapter. An MTP assistant is draft-only; load its target separately.

The LLaMA path requires the `llama` architecture/tokenizer pair, dense blocks, full even-sized head RoPE dimensions, and unscaled RoPE. The Gemma path recognizes `gemma4` and `gemma4-assistant` with the `gemma4` tokenizer. It implements per-layer head sizes, local/global attention, split-half RoPE and frequency factors, head normalization, GELU, post-branch norms, PLE, shared KV, and routed experts. Gemma tokenization uses explicit pair-ranked BPE, atomic special pieces, newline boundaries, suppressed IDs, and model stop tokens. Native generation accepts caller-formatted prompts. The compatibility adapters use a bounded text template described below; they do not execute arbitrary GGUF Jinja templates.

**Scope still in progress:** original Gemma 4 vision/audio encoder towers, DiffusionGemma's encoder/denoising decoder and scheduler, clustered MTP output acceleration, and MoE/MTP/multimodal training graphs. These are not enabled just because a container carries a familiar architecture label. DiffusionGemma is a separate execution graph, not an alias for autoregressive Gemma. Qwen, LLaMA 3's different tokenizer, and arbitrary GGML quantizations remain outside the validated native graph set.

## Build

Requirements: CMake 3.25+, a C17 compiler, and the platform development headers/libraries. GCC/Clang and MSVC use separate warning flags. C17 is the portable baseline; selecting C23 requires a compiler that supports that mode. A 64-bit build is strongly recommended for model files.

Run from the repository root:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release --parallel
ctest --test-dir build-cpu -C Release --output-on-failure
```

The backend can also be configured directly:

```sh
cmake -S backend -B build-backend -DCMAKE_BUILD_TYPE=Release
cmake --build build-backend --config Release --parallel
ctest --test-dir build-backend -C Release --output-on-failure
```

Single-configuration generators place `fyodor-backend` (or `fyodor-backend.exe`) in the build directory. Visual Studio places it under `Release/`. Both builds produce a reusable `nya-engine` static library and a separate `nya-server` static library. No network download happens in the default build.

| CMake option | Default | Purpose |
|---|---|---|
| `NYA_C_STANDARD` | `17` | Select `17` or `23`; extensions remain disabled |
| `NYA_ENABLE_VULKAN` | `OFF` | Compile the C Vulkan provider |
| `NYA_ENABLE_CUDA` | `OFF` | Compile the optional C driver/NVRTC adapter when `/CUDA` and CUDA/NVRTC headers are available |
| `NYA_ENABLE_ONNXRUNTIME` | `OFF` | Compile the optional ONNX Runtime C API adapter |
| `NYA_FETCH_ONNXRUNTIME` | `OFF` | Explicitly permit the pinned SDK download |
| `NYA_ONNXRUNTIME_ROOT` | empty | Existing extracted SDK or NuGet package |
| `NYA_WARNINGS_AS_ERRORS` | `OFF` | Use `-Werror` or `/WX` |
| `NYA_ENABLE_IPO` | `OFF` | Enable link-time optimization after a toolchain check |
| `NYA_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UBSan; configure probes compiler/runtime support, including LLVM-MinGW on Windows |
| `BUILD_TESTING` | `ON` | Build C regression programs and register CTest tests |

Release uses the compiler's optimization defaults. The project deliberately does not enable fast-math or host-specific instruction sets: finite-value checks and portable binaries matter. IPO is available explicitly after correctness validation.

### Vulkan, independently from CUDA

Install a Vulkan SDK with headers version 1.3 or newer and a Vulkan loader/driver. The compute pipeline itself targets Vulkan 1.0 and needs no optional arithmetic features. CMake uses `Vulkan::Vulkan`; it does not probe or enable the CUDA language.

```sh
cmake -S . -B build-vulkan -DNYA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release --parallel
ctest --test-dir build-vulkan -C Release --output-on-failure
```

If SDK discovery needs help, set `VULKAN_SDK` or provide `Vulkan_INCLUDE_DIR` and `Vulkan_LIBRARY` to CMake. Vulkan executables require the Vulkan loader at process startup. Once started, a missing usable device falls back to CPU.

Vulkan execution is opt-in at runtime. Set the environment variable **before loading models**:

```powershell
$env:NYA_COMPUTE = 'vulkan'
.\build-vulkan\fyodor-backend.exe --port 0
```

```sh
NYA_COMPUTE=vulkan ./build-vulkan/fyodor-backend --port 0
```

An unset variable, `NYA_COMPUTE=cpu`, or an unrecognized value selects CPU. A CPU-only executable also stays on CPU if the variable requests Vulkan. This permits the same launcher configuration across both builds.

The Vulkan provider accelerates row-major F32 matrix-vector products. One workgroup computes each row using 64 lanes and shared-memory reduction. Each immutable weight matrix is uploaded once through a staging buffer into device-local storage; input/output mappings and command resources are reused. A model caches at most 256 matrices and 512 MiB of weight payload. Individual buffer/dispatch limits are checked against the selected device. Oversized operations, other tensor types, unsupported memory layouts, and allocation/driver failures use the existing CPU kernel. Device errors disable further GPU commands for that context.

This is a bounded initial compute provider, not full GPU inference: attention, normalization, activation functions, embeddings, and quantized kernels remain on CPU. Per-operation synchronization and transfers can make small models slower. No universal speedup is claimed. Floating-point reduction order differs from CPU, so sampled output need not be bit-identical across devices.

`vulkan/matvec_spv.h` contains the SPIR-V and the complete shader source in a comment, together with its regeneration command. Ordinary builds compile C and need no shader compiler. GPU API types remain confined to `vulkan/vulkan.c`; the core only sees the opaque handle in `include/compute.h`.

### CUDA in a removable directory

`/CUDA/cuda.c` loads the CUDA driver and NVRTC dynamically through their C APIs. `/CUDA/matvec.cu` contains the GPU kernels. CMake embeds the kernel source; NVRTC compiles PTX for an architecture supported by both the compiler and the selected GPU. No `nvcc` or host C++ compiler is required.

```sh
cmake -S . -B build-cuda -DNYA_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

Headers are found through `CUDA_PATH`/`CUDA_HOME` or explicit `NYA_CUDA_INCLUDE_DIR` (contains `cuda.h`) and `NYA_NVRTC_INCLUDE_DIR` (contains `nvrtc.h`). Runtime selection is `NYA_COMPUTE=cuda`. Set `NYA_CUDA_NVRTC` to an absolute NVRTC library path, or configure `NYA_CUDA_NVRTC_LIBRARY`; Windows also searches the indicated `CUDA_PATH`, while Linux can use its system library loader. Keep the matching NVRTC builtins library beside NVRTC. NVIDIA libraries are external dependencies and are not bundled in source. `NYA_CUDA_DEBUG=1` prints compilation/fallback diagnostics to stderr.

CUDA accelerates F32, F16, BF16, Q4_0, Q8_0, Q4_K and Q6_K matrix-vector products. Quantized matrices stay compressed on device and are decoded inside the kernel. Immutable weight uploads are cached by source address, shape and type. Each context caches at most 2048 matrices and the smaller of 8 GiB or three quarters of free device memory measured at creation. Vector buffers grow as needed. Attention, normalization and training backward currently remain CPU operations. GPU allocation/driver failures return to complete CPU kernels; missing runtime libraries do not prevent process startup.

CUDA and Vulkan can be compiled together and selected independently. CUDA uses its own stream and balances primary-context retain/release and push/pop calls; it does not reset another user's device context. All CUDA headers, dynamic loading, device allocation, source embedding and kernels are confined to `/CUDA`. The backend retains only the optional generic dispatch boundary.

### Prove the CUDA directory is unnecessary

```sh
cmake --build build-cpu --target verify-without-cuda
cmake --build build-vulkan --target verify-without-cuda
```

Each target creates a fresh source copy inside its build directory, **excludes `/CUDA` and the old `backend/cuda` path physically**, configures that copy with **`NYA_ENABLE_CUDA=ON`**, builds it with warnings as errors, and runs its CTest suite. It inherits the parent build's Vulkan selection and C standard, disables the external ONNX adapter, and leaves the original source tree untouched. The reported `without-cuda-*` directory contains the inspectable verification result. This also works when CUDA is already absent from the original checkout.

### Optional existing ONNX adapter

The default backend is self-contained C. The retained ONNX adapter calls a C ABI, but ONNX Runtime is an external implementation that contains C++; enable it only if that dependency is acceptable. It is not required for native GGUF generation, inspection, or Vulkan.

```sh
cmake -S . -B build-ort -DNYA_ENABLE_ONNXRUNTIME=ON -DNYA_ONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build-ort --config Release --parallel
ctest --test-dir build-ort -C Release --output-on-failure
```

Alternatively, explicitly add `-DNYA_FETCH_ONNXRUNTIME=ON`. The existing pin is Microsoft.ML.OnnxRuntime `1.26.0` with SHA-256 `50cc3772668f04b8373ad65a36793f94699bc4e818f6e691fc68f1578c38ce42`. The SDK needs `build/native/include/onnxruntime_c_api.h` or `include/onnxruntime_c_api.h`, plus its matching architecture library under `runtimes/<platform>-<architecture>/native` or `lib`. CMake selects the target architecture rather than searching incompatible binaries in order.

Keep the copied `onnxruntime.dll`, `libonnxruntime.so`, or `libonnxruntime.dylib` beside the executable. The adapter loads it from the executable directory. The current HTTP interface supports one named dense input and one named dense output, positive dimensions, and numeric tensor types; it is not a general multi-input graph API.

## Run and configure

```sh
./build-cpu/fyodor-backend --port 0
```

On Windows use `.\build-cpu\fyodor-backend.exe`, or the `Release` subdirectory for Visual Studio. The process applies compiled defaults, optional `config.yaml` from its working directory, then command-line overrides. `--config PATH` makes that file required. `--help` works even if the configuration is malformed.

```text
fyodor-backend [--config PATH] [--bind IPV4] [--port PORT] [--allow-remote]
```

The strict YAML subset supports one `server:` map, two-space indentation, simple scalar values, comments, and matching outer quotes. It is not a complete YAML implementation: no nesting beyond that map, lists, anchors, escapes, tabs, unknown keys, duplicate keys, NUL bytes, or partial application of a failed file. Files are bounded to 64 KiB and lines to 4 KiB.

```yaml
server:
  bind: 127.0.0.1
  port: 0
  allow_remote: false
  auth_token_env: ""
  request_timeout_ms: 5000
  max_request_bytes: 1048576
  max_output_bytes: 67108864
  max_generation_tokens: 256
  backlog: 64
  worker_threads: 4
  queue_capacity: 128
  allowed_origins: "http://localhost:5173"
```

The final origin above is supplied by the repository's config file; compiled defaults have no additional configured origins. Built-in origins are `tauri://localhost`, `http://tauri.localhost`, `https://tauri.localhost`, and HTTP development origins at `localhost` with an optional numeric port. Non-browser requests without `Origin` still require authentication.

| Setting | Range / meaning |
|---|---|
| `bind` | Four decimal IPv4 octets; no DNS names, shortened forms, or leading-zero octets |
| `port` | `0..65535`; zero selects an available port |
| `allow_remote` | Required for any bind other than `127.0.0.1` |
| `auth_token_env` | Optional environment variable holding exactly 64 hex digits |
| `request_timeout_ms` | `100..300000`; absolute budget for receiving the whole request; each response send buffer also has a deadline |
| `max_request_bytes` | `4096..16777216`; headers plus body |
| `max_output_bytes` | `4096..1073741824`; returned tensor/text serialization bound, not a limit on all model or provider memory |
| `max_generation_tokens` | `1..4096`; maximum new tokens per request |
| `backlog` | `1..512` pending connections in the kernel |
| `worker_threads` | `1..64` fixed request workers |
| `queue_capacity` | Worker count through `4096`; bounded accepted-connection queue |
| `allowed_origins` | At most 16 additional exact origins, each under 256 bytes, in a comma-separated scalar |

Remote serving requires an explicit opt-in, for example `--bind 0.0.0.0 --allow-remote --port 6969`. The protocol is plaintext HTTP; use a trusted network or a TLS proxy for remote traffic. Binding remotely grants authenticated clients access to model paths readable by the backend process. Authentication does not provide filesystem sandboxing.

## Desktop launch contract

Human logs go to stderr. After binding and starting its workers, the process flushes one discovery line to stdout:

```text
FYODOR_READY 127.0.0.1 51837 <64-hex-digit-token>
```

The token is generated from 32 bytes of OS randomness unless supplied by `auth_token_env`. The launcher should spawn the executable directly, read and validate the discovery fields with a startup timeout, keep the token in memory, and give the frontend the connection details. With `0.0.0.0`, the discovery host is `127.0.0.1` for the local launcher; remote clients use the machine's reachable address.

Every API call except CORS preflight requires the local process token. Native/OpenAI calls use `Authorization: Bearer <token>`; Anthropic routes also accept `x-api-key: <token>`. Origin checks are independent of authentication. The frontend can use browser `fetch` directly; no application request needs to pass through Rust. On normal exit, send `POST /shutdown` and wait for the child. A supervisor may enforce a separate process-exit deadline because active model computation is synchronous.

## HTTP API, version 1

Requests use HTTP/1.0 or HTTP/1.1 with a fixed `Content-Length` for bodies. Each connection carries one request and closes after its response. Chunked bodies, WebSockets, SSE, persistent sessions, and token streaming are not implemented. Binary tensor payloads already use raw bytes regardless of size; the conversation's proposed 10 MB transport switch is not a second implemented protocol.

JSON bodies must be complete UTF-8 objects with unique top-level keys. Unicode escapes and surrogate pairs decode correctly; malformed encodings, embedded NUL, trailing data, and duplicate escaped-equivalent keys fail before model state is changed. Unknown values can nest to 32 levels; top-level objects are bounded to 64 fields with decoded names up to 127 bytes. Headers have a separate 32 KiB limit. Duplicate recognized headers and ambiguous framing are rejected.

| Method and path | Body | Result |
|---|---|---|
| `GET /health` | None | Readiness, engine version, uptime, bind, port, worker count, remote flag |
| `GET /models` | None | Registry entries, validated container metadata, execution capabilities and failure reasons |
| `POST /model/load` | JSON `{"path":"/models/model.gguf"}` | Loaded model metadata; loading the identical path again returns its existing entry |
| `POST /model/unload` | JSON `{"model_id":1}` | Removes the entry and frees provider state |
| `POST /generate` | JSON described below | Complete generated text and token statistics |
| `POST /model/run` | Raw tensor bytes plus headers below | Raw output tensor from a native projector or supported ONNX session |
| `POST /shutdown` | Empty body | Acknowledges shutdown and stops accepting work |
| `OPTIONS` | None | Allowed-origin preflight, no bearer token needed |

Model IDs are process-local, monotonically increasing, and limited to 32 simultaneously loaded models. Paths are UTF-8, under 1024 bytes including termination, and relative to the backend working directory unless absolute. Clients must not edit or truncate files while their model mappings are active.

### Text generation

```http
POST /generate HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json
Content-Length: <body-byte-count>

{"model_id":1,"prompt":"Hello","max_tokens":64,"temperature":0.8,"top_p":0.95,"top_k":40,"seed":42}
```

Required fields are `model_id` and `prompt`. Defaults are `max_tokens = min(128, configured limit)`, `temperature = 0.8`, `top_p = 0.95`, and `top_k = 40`. Omitted seeds come from OS randomness and are returned. `temperature = 0` selects greedy sampling; otherwise temperature must be in `(0,5]`. Top-p is in `(0,1]`; top-k is `0..1000000`, with zero disabling the filter. Top-k selection uses a bounded heap, then top-p operates on the retained mass. Reproducibility applies to the same weights, implementation, device, and request.

```json
{"version":1,"ok":true,"model_id":1,"text":"...","prompt_tokens":2,"generated_tokens":64,"seed":42,"stop_reason":"length","draft_tokens":0,"accepted_draft_tokens":0,"target_steps":65}
```

`stop_reason` is `length`, `eos`, or `context`. An EOS token counts as generated even though it adds no text. The prompt must leave space in the model context. Requests own a fresh KV cache; no conversation state is retained between calls. Output bounds reserve worst-case JSON escaping overhead. A generated sequence ending in invalid or incomplete UTF-8 produces an explicit generation error rather than malformed JSON.

To use a draft, load it and add `"draft_model_id":2,"speculative_tokens":4` to the request. The window is 1–32; omitting it selects four. Ordinary drafts require matching tokenizer IDs and normalization. MTP assistants additionally require a target with compatible hidden width and local/global KV layouts. Both mappings remain pinned for the call.

Sampling verifies proposals against the target after applying temperature/top-k/top-p. Rejected suffixes are rolled back, and replacements use the positive residual distribution. EOS is honored only after target acceptance. `draft_tokens` counts proposals, `accepted_draft_tokens` counts accepted proposals, and `target_steps` counts evaluated target positions, including prefill. Current verification evaluates positions serially: **acceptance counters do not imply a throughput improvement**. Independent draft randomness is separate from target randomness; sampled strings can differ from a non-speculative run while preserving the target distribution.

### Binary tensor execution

`POST /model/run` uses `Content-Type: application/octet-stream` and these headers:

```text
X-Fyodor-Model-Id: 1
X-Fyodor-Input-Name: input
X-Fyodor-Input-Type: float32
X-Fyodor-Input-Shape: 1,3
X-Fyodor-Output-Name: output
```

The body is contiguous tensor storage with exactly the byte count implied by type and shape. Shapes have 1–16 positive dimensions. Values use the backend host's native numeric representation; supported/tested desktop hosts are little-endian. Responses contain raw bytes and `X-Fyodor-Output-Type` / `X-Fyodor-Output-Shape` headers. No base64 is used. Size limits apply before execution; ONNX Runtime may allocate internal graph memory before the final output-size check.

Errors normally use `{"version":1,"ok":false,"error":{"code":"...","message":"..."}}`. Relevant status codes include `400` invalid syntax, `401` authentication, `403` origin, `404` model/path, `405` method, `408` request deadline, `409` unsupported execution capability, `413` request size, `415` media type, `422` invalid model or execution failure, `431` headers, and `503` resource/queue saturation. A saturated queue can return an empty best-effort `503`; disconnects need not yield a JSON response.

### Gemma 4 Unified multimodal inputs

Load the `mmproj` GGUF separately. The native projector uses output name `embeddings` and F32 input/output. Returned shape is `[soft_tokens, language_width]`.

| Input name | Shape | Contract |
|---|---|---|
| `image_rgb` | `[height,width,3]` | RGB HWC floats in `[0,1]`; dimensions must be whole effective model patches (48 for the tested 12B projector) |
| `image_patches` | `[patches,2+patch_elements]` | Each row is integer-valued `x,y`, then CHW patch pixels in `[0,1]`; omit padding patches |
| `audio_features` | `[soft_tokens,audio_width]` | Preprocessed, packed audio features; width 640 for the tested 12B projector |

Image decoding/resizing and waveform-to-feature preprocessing are not yet supplied. Coordinates must fit the learned positional table. The vision graph applies three affine LayerNorms, patch projection, both positional axes, RMSNorm, and the language projection. Audio applies RMSNorm and its own projection. The GGUF patch tensor determines effective patch size; teacher-patch metadata alone is insufficient.

The C `nya_generation_request` accepts sorted `nya_generation_soft_tokens` spans that replace positions in the **tokenized** prompt. Each span states its width, element count, and whether it represents one image/frame. Embeddings are already in language space and are not multiplied by `sqrt(D)` again. One placeholder token is required per embedding row. Each image span allows future keys within that image in local layers; global layers remain causal. Audio remains causal. The caller owns prompt formatting and all borrowed span storage. The JSON `/generate` route does not yet accept these spans.

## OpenAI-compatible, Anthropic-compatible and native REST

All three interfaces execute **local loaded models**. They do not forward requests to OpenAI or Anthropic, and require no vendor key. The API key is the per-process token from `FYODOR_READY`; restart generates a new token unless `auth_token_env` supplies a fixed value. Load a model through the native endpoint first, then use its numeric ID as a **string** in compatibility requests (for example `"1"`). Compatibility model lists contain only models with active text generation; the native list also includes inspect-only models and MTP assistants.

| Interface | Base URL | Implemented endpoints |
|---|---|---|
| Native | `http://127.0.0.1:PORT/api/v1` | `GET /health`, `/models`, `/capabilities`; `POST /model/load`, `/model/unload`, `/model/run`, `/generate`, `/tokenize`, `/shutdown` |
| OpenAI | `http://127.0.0.1:PORT/openai/v1` | `GET /models`; `POST /chat/completions`, `/completions` |
| Anthropic | `http://127.0.0.1:PORT/anthropic` | `GET /v1/models`; `POST /v1/messages`, `/v1/messages/count_tokens` |

Existing unprefixed native routes remain available. `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/messages/count_tokens` are compatibility aliases. `/v1/models` uses OpenAI's schema; use `/anthropic/v1/models` for Anthropic's schema. Requests use JSON except native `/model/run`, whose binary tensor contract is described above. The server retains its auth, CORS, request/output limits, execution serialization and model lifetime protection across all interfaces.

### Supported compatibility subset

OpenAI Chat accepts `model`, `messages`, `max_tokens` **or** `max_completion_tokens`, `temperature`, `top_p`, `seed`, `stream:false`, and `n:1`. Completions accepts a single string `prompt` instead of `messages` and uses `max_tokens`. Anthropic Messages accepts `model`, `messages`, required `max_tokens`, optional `system`, `temperature`, `top_p`, `top_k`, and `stream:false`. Send `anthropic-version: 2023-06-01`. Anthropic authentication accepts the local token in either `x-api-key` or `Authorization`; if both are present, both must match.

Messages accept text strings or arrays of `{ "type": "text", "text": "..." }` blocks. Roles are `user` and `assistant`; OpenAI also accepts one leading `system` or `developer` message. Anthropic puts system instructions in the top-level `system` field. At most 256 messages/blocks are accepted, and the last message must be a user turn. Duplicate interpreted fields, non-text blocks and unknown fields are rejected.

**Not implemented:** SSE/token streaming, tool calls, JSON-schema output constraints, user stop sequences, logprobs, multiple choices, image/audio message input, assistant prefills, OpenAI Responses/embeddings/batch APIs, or provider model retrieval/pagination. Requests for these features fail clearly; they are not silently ignored. Model listing returns the complete small local registry in one response. Compatibility errors use the respective provider's error envelope; malformed HTTP framing rejected before routing retains the native transport error format.

The Gemma 4 text template uses `<|turn>ROLE\n...<turn|>\n`, maps assistant to `model`, and starts generation with `<|turn>model\n<|channel>thought\n<channel|>` (the checkpoint's empty-thought prefix). System/developer content uses a system turn. Text whitespace is preserved. The LLaMA fallback uses `role: text\n` and ends in `assistant: `. These adapters do not interpret a checkpoint's arbitrary Jinja template or apply model-specific tool/vision logic. For exact chat formatting or deeper sampling control, construct the prompt yourself and call native `/generate`.

Compatibility sampling shares the native defaults `temperature=0.8`, `top_p=0.95`, and a secure random seed when omitted. Top-k is disabled for compatibility calls unless provided through Anthropic. An omitted OpenAI token limit uses the smaller of 128 and the configured server limit. Usage counts come from the local tokenizer, including rendered prompt markers and BOS/EOS rules. Completion counts include a sampled stop token. EOS maps to OpenAI `stop` / Anthropic `end_turn`; token/context limits map to `length` / `max_tokens`. Context overflow before generation is an error. These local models do not have vendor model names or vendor token counts.

### Native examples and deeper control

Set `BASE` to `http://127.0.0.1:PORT` and `FYODOR_TOKEN` to the readiness token. Examples use shell environment-variable syntax; in PowerShell use `$env:FYODOR_TOKEN` and `curl.exe` or `Invoke-RestMethod`.

```sh
curl "$BASE/api/v1/model/load" -H "Authorization: Bearer $FYODOR_TOKEN" -H "Content-Type: application/json" --data '{"path":"D:/Models/model.gguf"}'
curl "$BASE/api/v1/models" -H "Authorization: Bearer $FYODOR_TOKEN"
curl "$BASE/api/v1/capabilities" -H "Authorization: Bearer $FYODOR_TOKEN"
curl "$BASE/api/v1/tokenize" -H "Authorization: Bearer $FYODOR_TOKEN" -H "Content-Type: application/json" --data '{"model_id":1,"text":"Hello"}'
curl "$BASE/api/v1/generate" -H "Authorization: Bearer $FYODOR_TOKEN" -H "Content-Type: application/json" --data '{"model_id":1,"prompt":"Hello","max_tokens":64,"temperature":0.7,"top_p":0.95,"top_k":40,"seed":42}'
```

`/tokenize` returns `{"version":1,"ok":true,"model_id":1,"count":N,"tokens":[...]}` using that model's native tokenizer. Native `/generate` additionally accepts `draft_model_id` and `speculative_tokens` for ordinary/MTP draft verification and returns proposal/acceptance/target-step counters. Load the draft through `/model/load` first. Native `/model/run` provides named binary tensor execution for supported projectors/ONNX. `/capabilities` describes the available API routes and compatibility limitations. Training remains in the C library/CLI; no HTTP training endpoint is advertised.

### OpenAI client examples

```sh
curl "$BASE/openai/v1/chat/completions" -H "Authorization: Bearer $FYODOR_TOKEN" -H "Content-Type: application/json" --data '{"model":"1","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":64,"stream":false}'
```

```python
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["BASE"] + "/openai/v1",
                api_key=os.environ["FYODOR_TOKEN"])
model_id = client.models.list().data[0].id
reply = client.chat.completions.create(
    model=model_id, messages=[{"role": "user", "content": "Hello"}],
    max_completion_tokens=64, stream=False)
print(reply.choices[0].message.content)
```

### Anthropic client examples

```sh
curl "$BASE/anthropic/v1/messages" -H "x-api-key: $FYODOR_TOKEN" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" --data '{"model":"1","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":false}'
```

```python
import os
from anthropic import Anthropic

client = Anthropic(base_url=os.environ["BASE"] + "/anthropic",
                   api_key=os.environ["FYODOR_TOKEN"])
model_id = client.models.list().data[0].id
messages = [{"role": "user", "content": "Hello"}]
print(client.messages.count_tokens(model=model_id, messages=messages).input_tokens)
reply = client.messages.create(model=model_id, messages=messages, max_tokens=64)
print(reply.content[0].text)
```

The SDKs are optional **client** dependencies. They were exercised against the local server during verification; the backend itself is still C/CUDA. Wire shapes follow the official [OpenAI Chat reference](https://developers.openai.com/api/reference/resources/chat) and [Anthropic Messages/token-counting reference](https://platform.claude.com/docs/en/api/messages/count_tokens); the supported local subset above is authoritative for this implementation.

## Training

`fyodor-train` and `include/pretraining.h` provide runnable dense LLaMA and Gemma 4 training paths. It supports randomly initialized LLaMA decoders, full-weight training, and LoRA over mapped GGUF weights. Gemma training starts from an imported checkpoint; a random Gemma factory is not implemented. Training uses the eager C autograd API in `include/training.h`; it has no Python dependency or PyTorch ABI. MoE/MTP training, multimodal encoder training, GPU backward, mixed precision, distributed training and large-scale streaming loaders remain unfinished.

| Workflow | Starting point | Implemented behavior |
|---|---|---|
| LoRA | Frozen supported GGUF, including quantized tensors | Train `Wx + (alpha/r) B(Ax)` on attention/MLP/output matrices; B starts at zero; export merges into a complete GGUF |
| SFT | Pretrained full weights or adapters | Mean next-token cross-entropy over response labels; prompt-only labels are masked |
| CPT | Pretrained full weights or adapters | Continue next-token prediction on consecutive corpus windows |
| Pretraining | **Random weights** | Seeded Gaussian matrices, unit RMSNorm scales, AdamW updates, checkpoint/resume, complete inference GGUF export |
| DPO | Trainable policy and frozen initial reference | Summed response log-probabilities for chosen/rejected pairs; fixed reference values are cached before updates |

The LLaMA model graph includes token embeddings, RMSNorm, grouped-query causal attention, full-head RoPE, SwiGLU, residuals, and an untied vocabulary head. LLaMA imports require the native dense `llama` graph with an explicit output tensor. Dense Gemma 4 imports additionally support per-layer head widths, local/global attention, head/post-branch normalization, GELU, shared KV, proportional split-half RoPE, PLE, scalar layer scales and final logit softcapping. Full-weight Gemma training preserves tied input/output parameters; LoRA freezes tied embedding/head tensors so merging cannot change only one side of the tie. Suppressed tokens are an inference policy; training losses use the full finite vocabulary logits. A newly initialized model uses 259 tokens: UNK=0, BOS=1, EOS=2, and byte `b` has ID `3+b`. Other vocabulary sizes can train through the C graph but cannot yet export without a custom tokenizer implementation. Imports preserve their validated tokenizer and suppressed IDs.

### Command-line workflow

The normal build produces `fyodor-train` alongside `fyodor-backend`. These examples use paths relative to the repository root; on Windows the executable has an `.exe` suffix. Put UTF-8 training text in `corpus.txt`, then train a small model from scratch:

```sh
build-cpu/fyodor-train --mode pretrain --data corpus.txt --output trained.gguf --checkpoint trained.ckpt --steps 1000 --lr 0.001 --dimension 64 --ff 128 --layers 2 --heads 4 --kv-heads 2 --context 128
```

Resume with the same configuration and data, writing **new output paths**:

```sh
build-cpu/fyodor-train --mode pretrain --data corpus.txt --output resumed.gguf --checkpoint resumed.ckpt --resume trained.ckpt --steps 1000 --dimension 64 --ff 128 --layers 2 --heads 4 --kv-heads 2 --context 128
```

`--steps` counts additional updates. Checkpoints restore AdamW settings, including the learning rate, so a resumed `--lr` does not override the saved setting. Resume chooses the next record/window from the saved update count. Checkpoints contain ordered parameter shapes and state, **not** model configuration, tokenizer, frozen base weights, dataset or objective. Supply the same configuration/base/data/objective; matching shapes alone cannot detect a different base model. For DPO, the original base is reloaded to reconstruct the frozen reference before restoring the trainable policy. Exact resumed-versus-uninterrupted GGUF/checkpoint equality is covered by tests.

For full-weight continued pretraining use rank zero. Positive rank selects LoRA; the CLI defaults to rank 8 and `alpha=2*rank`:

```sh
build-cpu/fyodor-train --mode cpt --base trained.gguf --rank 0 --data corpus.txt --output continued.gguf --steps 100
build-cpu/fyodor-train --mode sft --base trained.gguf --rank 8 --data sft.tsv --output adapted.gguf --checkpoint adapter.ckpt --steps 100
build-cpu/fyodor-train --mode dpo --base trained.gguf --rank 8 --data preferences.tsv --output preferred.gguf --beta 0.1 --steps 100
```

SFT input has one `prompt<TAB>completion` record per line. DPO has `prompt<TAB>chosen<TAB>rejected`. Use literal tab separators; embedded tabs and multiline fields are not supported. Include desired spacing, chat markers and end-of-turn text in the fields. The tokenizer processes each concatenated prompt/response once. The common token prefix with the prompt is masked; a subword crossing that boundary is supervised. No template or EOS marker is invented for a record. Records exceeding context are rejected rather than truncated. Corpora use consecutive next-token windows and include the final short window. The simple loader accepts at most 64 MiB of text and 100,000 structured records; it visits records in file order and does not shuffle or split validation data.

`--memory-mib` defaults to 256 for **each** of persistent parameter state and graph intermediates. Parameter state uses F32 weights, gradients and two Adam moments. AdamW additionally stages three F32 arrays for a transactional update; checkpoint loading stages four. Dataset tokens, metadata and mapped base weights are separate. These are component budgets, not a hard process-memory ceiling. Full-weight import expands quantized values to F32. LoRA keeps base weights mapped and propagates gradients through them without creating base-weight gradients; LLaMA output GGUFs stream F32 weights. Gemma exports convert updated tensors to F32 and preserve frozen tensors in their original representation. They preserve tokenizer/architecture metadata and omit the stale optional `general.file_type` summary; the tensor directory describes actual types. Merged files can be much larger than quantized inputs. A single forward/backward/AdamW LoRA step was exercised on the supplied 12B Q4_K_M Gemma (656 trainable adapter tensors, about 110.4 MiB of graph intermediates). This is a smoke test, not evidence of large-model convergence or practical training throughput.

Output paths must not already exist. The runner creates files exclusively so neither a base model nor an earlier checkpoint can be overwritten. A detected write failure removes its newly created partial file; an interrupted process can leave an incomplete file that must be discarded. A successful checkpoint is retained if a later GGUF export fails. There is no HTTP training endpoint yet.

### C integration

Parameters outlive graphs. Create a decoder with `nya_train_decoder_create`, or import a loaded model with `nya_train_decoder_from_model` (rank zero for full weights). `nya_train_decoder_parameters` returns unique parameter pointers for AdamW and checkpoint calls. An imported model's mapping must remain loaded until its decoder and active graphs are freed.

```c
/* model, tokens and labels are supplied by the application. Labels identify
   the NEXT token for each row; an optional mask selects supervised rows. */
size_t parameter_count;
nya_train_parameter *const *parameters =
    nya_train_decoder_parameters(model, &parameter_count);
for (size_t i = 0; i < parameter_count; ++i)
    nya_train_zero_grad(parameters[i]);
nya_train_graph *graph = nya_train_graph_create(256U * 1024U * 1024U);
nya_train_tensor *logits = nya_train_decoder_forward(model, graph, tokens, count);
nya_train_tensor *loss = nya_train_cross_entropy(logits, labels, mask, count);
int ready = loss != NULL && nya_train_backward(loss) == 0;
if (!ready) {
    /* Report nya_train_error(graph); discard this failed update. */
}
nya_train_graph_free(graph);
/* optimizer was initialized once with nya_train_adamw_defaults. */
if (ready && nya_train_adamw_step(&optimizer, parameters, parameter_count,
                                error, sizeof(error)) != 0) {
    /* Report error; AdamW has left the complete parameter state unchanged. */
}
```

The low-level graph supports embedding lookup, linear layers, broadcasting, GELU/SiLU, RMSNorm, reshape, RoPE, grouped-query attention and scalar losses. Gradients accumulate until explicitly cleared, permitting microbatch accumulation. Scale each microbatch loss appropriately for the desired total-token mean. `nya_train_logprob` returns a sequence **sum**; `nya_train_cross_entropy` averages unmasked labels. DPO takes two policy sequence sums, two frozen-reference sums, and positive beta. The caller can combine operations to build other objectives.

AdamW applies bias correction, decoupled weight decay and optional global gradient clipping. It validates and stages the complete update before changing weights, moments or step. Failed backward may have partially accumulated gradients; clear them before retrying. Checkpoint reading validates shapes, finite values, nonnegative variances, checksum and exact stream end before committing state. The checksum detects corruption, not malicious modification. Do not mutate parameter data while a graph borrows it, and serialize access to shared models/parameters. Training currently uses CPU operations even when CUDA/Vulkan inference is enabled.

## Code map and ownership

| Files | Responsibility |
|---|---|
| `core/main.c`, `include/nya.h` | Process lifecycle, CLI overrides, readiness contract, version |
| `core/config.c` | Bounded transactional configuration parsing and validation |
| `core/server.c`, `core/server_compat.inc` | HTTP framing, JSON, routing, authentication, worker/connection lifecycle |
| `core/thread.c`, `core/runtime.c` | Win32/POSIX threads, monotonic clock, OS randomness |
| `core/model.c` | Registry ownership and capability attachment |
| `core/format*.c` | Container detection, checked 64-bit reader, GGUF/SafeTensors/ONNX inspection |
| `core/file.c` | UTF-8 model paths on Windows/POSIX and opened-descriptor size checks |
| `core/llm_gguf.c` | Read-only file mapping, metadata/tensor binding, tokenizer and detokenizer |
| `core/llm_cpu.c` | CPU kernels, per-request KV/scratch state, transformer and sampling loop |
| `core/multimodal.c` | Native Unified image/audio projector execution |
| `core/training.c`, `include/training.h` | Eager differentiation, losses, trainable parameters, AdamW and checkpoints |
| `core/pretraining.c`, `include/pretraining.h`, `core/train_main.c` | Random/imported decoder training, LoRA merge, GGUF export and CLI datasets |
| `core/generation.c`, `core/execution.c` | Public text/tensor execution boundaries |
| `core/compute.c`, `vulkan/*` | Optional acceleration dispatch and Vulkan implementation |
| `../CUDA/*` | Optional CUDA driver/NVRTC implementation and quantized GPU kernels |
| `include/*` | Public C types and functions |
| `tests/*` | Self-generated model fixtures, numerical and protocol regression programs |

The registry owns models; each model owns its mapped generation context and optional native projector or ONNX session. Request structs borrow their inputs for the synchronous call. Successful generation/execution responses own their allocated buffers until the corresponding `*_response_free` function is called. Free all model state before process-wide provider shutdown. A compute context must be freed before its immutable source weights disappear. Contexts with accelerator state require external serialization.

The server uses separate execution and registry locks. Model loading/unloading and provider operations are serialized so unload cannot invalidate active model pointers. Generation copies the registry record while retaining the execution lock, then releases the registry lock during computation. `/models` can therefore respond during inference; network sends never hold the registry lock. Model load still holds the registry lock during inspection/attachment. Long-running inference remains synchronous and can occupy workers waiting for the execution lock. This is not a continuous-batching scheduler.

Accepted sockets use bounded queues and absolute receive deadlines. Shutdown closes queued sockets, interrupts active socket I/O, wakes workers, joins them, and frees state. Windows waits poll cancellation in short slices because shutting down a socket does not reliably wake every `select` wait immediately. Active CPU, ONNX, or GPU computation must finish before teardown; there is no hard inference cancellation or GPU-hang recovery guarantee.

## Changes and verification

The 0.3 refactor removes duplicate engine compilation from test targets, makes default builds offline, splits engine from transport, and supplies independent Vulkan compilation. Parsers validate complete payload extents, integer overflow, duplicate tensor names, overlaps, Unicode, JSON shape, and file-size consistency. Seeking skips large opaque payloads without reading them into a discard buffer. Tokenizer merges use indexed text spans and a heap instead of rescanning and moving every token pair. Prefill skips vocabulary projection until logits are actually needed; attention heads reuse one score buffer; RoPE frequencies and per-position trigonometry are reused; greedy decoding needs no candidate allocation.

On Windows x64, validation used GCC 16.2 with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`:

- CPU C17: configuration, 46 container fixtures plus API edges, native generation, HTTP/socket behavior, and CPU fallback tests.
- Vulkan C17: the same suite plus real device matrix-vector comparisons, cached matrix reuse, buffer resizing, and native generation with Vulkan requested.
- Standalone backend C23 with ONNX Runtime: identity graph execution and the common regressions.
- Both CPU and Vulkan builds from source copies physically missing repository `/CUDA` and legacy `backend/cuda`.
- Live process checks: readiness, auth, CORS, model lifecycle, malformed JSON/HTTP, binary input over 32 KiB, slow-drip timeout, and shutdown with stalled clients.

Generation fixtures cover F32/F16/Q4_0/Q8_0, malformed tokenizer arrays, invalid tensor bindings, sampling mass, context/output limits, Unicode merge ordering against a separate slow reference, long prompts, nonfinite values, and invalid output UTF-8. Vulkan tests compare against double-precision CPU reference sums within a stated tolerance. `validate-vulkan-compute` is marked skipped when no usable device is available, rather than passing without GPU work. The provider was exercised on actual hardware during this change.

Final September 7–8 checks cover Gemma dense/PLE/MoE/fused-expert graphs, MTP pairing and rollback, independent K-quant packing, speculative sampling distributions, Unified projectors and multimodal prefill against float64 references, and finite-difference training gradients. Dense Gemma full-weight and LoRA tests include shared KV, PLE, tied embeddings, loss reduction, metadata preservation and merged GGUF reload.

| Final configuration | Passed |
|---|---:|
| CPU C17 | 11/11 |
| LLVM-MinGW AddressSanitizer + UBSan | 11/11 |
| CUDA + Vulkan on RTX 5060 Ti | 16/16 |
| Standalone C23 + ONNX Runtime | 12/12 |
| Fresh source without `/CUDA`, CUDA requested, CPU | 11/11 |
| Fresh source without `/CUDA`, CUDA requested, Vulkan | 13/13 |
| Live HTTP/process checks | 36/36 |

OpenAI Python SDK 3.8.0 successfully exercised model listing, Chat Completions and Completions. Anthropic Python SDK 1.4.0 exercised model listing, Messages and token counting, including usage-count agreement. These calls used an actual local server and the trained tiny GGUF fixture. The C API regression also covers provider error envelopes, authentication, unsupported options, Unicode text blocks, prompt markers and tokenization errors.

The final sweep removed a tokenization error path's pointer use after releasing the model-lifetime lock, and avoided narrowing Winsock handles in the API test. Sanitizers reported no findings in the exercised paths. GCC `-fanalyzer` emitted no diagnostics for the training core, decoder factory or training CLI; its server diagnostics concerned POSIX file-descriptor assumptions against Winsock and were inspected against the explicit socket cleanup paths. An earlier, already configured CUDA source-copy build also rebuilt automatically and passed its then-current 10/10 suite after its copied CUDA directory was removed, without clearing the build directory.

The local 12B Q4_K_M target, its MTP assistant, and E4B Q6_K_P model load through the native provider. The 12B target generated eight tokens on CUDA; MTP accepted all six proposals and matched ordinary greedy output in that smoke test. MTP permits a normal/control classification difference for otherwise identical token IDs and spellings because it consumes target tokens directly; byte-token classes, vocabulary text, normalization, merge ranks and stops remain checked. The real 12B Unified projector's 7680 image output values differed from an independent float64 calculation by at most `4.01e-5`; 3840 audio values differed by at most `4.41e-6`. No DiffusionGemma checkpoint was found in the supplied model directory. The random-weight decoder fixture reduces loss from 5.593439 to 0.000670, exports to GGUF, and generates the expected eight-token continuation. Sequence training logits and cached inference logits agree within `3e-5` for LLaMA and `8e-5` for the Gemma fixtures before/after training and after LoRA merge. These tiny numerical fixtures test implementation behavior; they do not establish general language quality.

Tests do not prove absence of all undefined behavior, trained-model accuracy, broad tokenizer compatibility, or a performance figure on arbitrary hardware. MSVC and Linux/macOS execution were not run in this Windows environment. For a sanitizer build, configure with `-DCMAKE_BUILD_TYPE=Debug -DNYA_ENABLE_SANITIZERS=ON` using a compiler that supplies both runtimes (LLVM-MinGW was used here), then run CTest. GCC static analysis has finite path-complexity limits and reports some POSIX-descriptor assumptions against Winsock; it is supplementary to numerical, parser and sanitizer tests.

## Format and GPU references

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [GGML block layouts](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h)
- [SafeTensors format](https://github.com/huggingface/safetensors/blob/main/README.md)
- [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html) and [optional ONNX Runtime C API](https://onnxruntime.ai/docs/get-started/with-c.html)
- [Khronos compute shader guide](https://docs.vulkan.org/guide/latest/compute_shaders.html) and [synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
- [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) and [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), [Gemma 4 implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4), and [Unified implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4_unified)
- [LoRA](https://arxiv.org/abs/2106.09685) and [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
