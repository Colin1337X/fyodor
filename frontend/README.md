# Fyodor desktop

The frontend is pure vanilla HTML, CSS, and JavaScript. Tauri 2 only supplies the cross-platform window, native file picker, installers, and C sidecar supervision. Inference requests travel directly from the webview to the authenticated loopback HTTP API.

## Development

```sh
cmake -S .. -B ../build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build ../build-cpu --config Release --parallel
npm install
npm run desktop:dev
```

`prepare:sidecar` selects `build-cuda`, then `build-vulkan`, then `build-cpu`, and copies the backend and trainer into Tauri resources. Set `FYODOR_BACKEND_BIN` and `FYODOR_TRAIN_BIN` to use explicit executables. CUDA packaging includes the matching NVRTC/builtins DLLs and their license when available; CPU packaging has no CUDA dependency.

Use `npm run dev` for browser-only UI work. Set `VITE_FYODOR_URL` and `VITE_FYODOR_TOKEN` to connect it to an already running backend.

Set `NYA_CUDA_CUTLASS_LIBRARY` to a standalone `fyodor-cutlass.dll` to bundle it
and its adjacent `CUTLASS-LICENSE.txt`. Omitting this setting removes previous
generated CUTLASS copies. cuBLAS retains priority; CUTLASS is used when cuBLAS is
unavailable. `NYA_CUDA_CUTLASS=0` disables it; `=1` forces it for comparison. See
the [removable adapter documentation](../CUDA/cutlass/README.md).

## Installers

`npm run desktop:build` produces formats supported by the current host: NSIS/MSI on Windows, DMG/app on macOS, and AppImage/deb/rpm on Linux. Build on each target platform with its matching C executable.

On Windows, the backend and trainer are supervised child processes with `CREATE_NO_WINDOW`; they do not open separate console windows. Launch the desktop with `--verbose` or `FYODOR_VERBOSE=1` to capture detailed backend diagnostics in the Logs view. The child readiness line and authentication token are consumed internally. Closing the desktop requests a safe training stop, waits for checkpoint/model saving, then shuts down the inference sidecar. A failed training save keeps the window open with the error log.

## Workspace

The workspace shows loaded models, runtime capabilities, and explicit CPU/CUDA/Vulkan controls. Provider selection is validated by the backend. Dense CUDA inference can use resident plans; selecting CUDA alone does not guarantee every architecture is accelerated. The API panel links the native, OpenAI-compatible and Anthropic-compatible routes. See [the backend API documentation](../backend/README.md#http-api-version-1) for request examples and compatibility limits.

Chat preserves conversation history and supports bounded JSON import/export. Playground exposes sampling and speculative-decoding settings. Training runs the native CLI for random-init pretraining, CPT, SFT and DPO, with full weights or LoRA as appropriate; dataset/output paths refer to local files. Training currently executes on CPU. The native trainer accepts Unicode paths for datasets, base models, checkpoints, resume files and output models on Windows, including Korean and emoji filenames.

“Sequences / pairs per update” controls gradient accumulation (1–1024, default
1). Graph memory holds one sequence, or a chosen/rejected DPO pair, while
gradients accumulate before one optimizer update. Unequal sequences are weighted
by supervised token count; DPO averages pairs. Resume requires the same setting.
The loss trace reports the group objective, and live throughput includes all
processed input tokens. **Stop & save** finishes the current complete optimizer
update, writes the optional checkpoint and output GGUF, and exits. The UI shows
SAVING / STOPPING and disables repeat requests while the child remains alive.
Closing the window waits for the same save; a failed save leaves the app open.
Supply a new checkpoint path to preserve optimizer state for exact resume.
Without it, stopping still exports the model but cannot preserve AdamW moments.
Large merged exports may take time; there is no automatic timeout that kills
the trainer. OS force termination and power loss are not graceful stops.

An optional OpenAI-compatible remote endpoint supports agent tool calls for local runtime information. Remote API keys remain in memory unless the user enables remembering the key. Local API compatibility does not yet implement tool calling or streaming.

## Current interface

Navigation groups Overview/Chat/Models under Use, Playground/Training under
Create, Benchmarks/Logs under Run, and API access under Serve. The sidebar
collapses to icons; Ctrl/Cmd+backslash toggles it. Narrow windows use a keyboard
accessible navigation overlay, with Escape to close it. The header preserves
the selected model; the footer reports its actual compute provider and activity.

Settings offers blue-black, blue-white, orange-white, graphite, forest, violet,
and system palettes, plus compact spacing. All screens use semantic CSS tokens.
The small local `public/appearance.js` script applies the palette before first
paint and stores appearance separately from sessions or credentials. No remote
fonts or frontend framework is required.

Chat includes safe fenced-code rendering, copy/code-copy, regeneration,
edit-and-resend, system prompts, and collapsible sampling controls. Long chats
initially render the latest 80 messages; earlier messages remain in history and
can be revealed. A failed request preserves the prompt and shows its error.
Remote OpenAI-compatible endpoints can stream text with Stop; interrupted text
is retained. SSE handles split UTF-8, multiline records and bounded buffers.
Stream updates touch one output node per animation frame and respect manual
scrolling. The **local C API still returns completed responses**; no local Stop
or fake streaming control is shown. Agent tools use their existing complete
response path, not streaming tool-call fragments.

Training's CPU threads setting accepts 0 for automatic selection or 1–64 for
an explicit limit. Large matrix operations share persistent native C workers;
the live thread count reports configured capacity, not measured utilization.
The setting can change when resuming a checkpoint. Training remains CPU-only.
See the [execution and correctness report](../backend/benchmarks/TRAINING_EXECUTION_20260925.md).

Training shows actual trainer log lines and a loss trace from reported
`step=... loss=...` values. Status polling updates telemetry without replacing
the configuration form. Launching training requires the desktop app. Logs have
source filtering, copy/clear and pause/resume automatic scrolling. The bounded
log text updates without replacing the toolbar.

Benchmarks imports real `fyodor-bench` schema-1 JSON and displays throughput,
standard deviation, token latency, prefix length and runtime metadata. Run the
documented CLI command and import its output; this view does not invent data or
claim to launch benchmarks. Imported results stay in the current window.

Optional cuBLAS packaging: set `NYA_CUDA_BLAS_LIBRARY` to an installed
`cublas64_13.dll` or `cublas64_12.dll` before `prepare:sidecar`. Its matching
cuBLASLt DLL is copied too. They are optional; the backend discovers adjacent
DLLs at runtime and retains native CUDA kernels if they are removed. This adds
substantial installer size, so builds without that variable omit them.

Validation: `node --test tests/ui.test.mjs` checks SSE framing, truncation and
size limits, inert model text, benchmark input and existing session bounds.
The September 11 browser pass exercised all eight screens across six palettes,
390px layouts, persistence, native generation/error recovery and remote
streaming/cancellation. [Screenshots and recorded checks](qa/stage3/results.json).
