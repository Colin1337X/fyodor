# Native training stop and recovery — September 24, 2026

This follows the [gradient accumulation work](TRAINING_20260924.md). Desktop
Stop previously called `Child.kill()`, discarding unsaved updates. Stop now sends
a byte through a private stdin pipe. The C runner checks between complete AdamW
updates, saves its optional checkpoint and required GGUF, and exits. Accumulation
groups remain indivisible. An empty pipe never blocks execution; EOF requests
the same stop. Losing the progress reader does not prevent saving in control
mode. Invalid regular-file stdin is rejected before loading the model.

The desktop keeps the child registered while it saves, exposes stopping status,
and disables repeat stop requests. Normal window close prevents immediate
closure and uses the same asynchronous wait. A save failure preserves the error
log and keeps the window open. There is no timeout that kills a slow save. A
completed checkpoint is retained if the subsequent GGUF export fails.

## Semantics and limits

- Opt in with `--control-stdin 1`; send ASCII `S` or close the pipe's write end.
  Ordinary CLI runs do not inspect stdin.
- Stop before the first update saves the initialized or resumed state, including
  a valid step-zero checkpoint. Stop during an update waits for all microbatches,
  clipping, moments and weight updates. Checkpoint identity is unchanged.
- Save uses the existing exclusive output paths. It never overwrites a user's
  model/checkpoint. Include a checkpoint path to recover optimizer moments;
  GGUF alone is insufficient for exact resume.
- Startup and export are not cancellable. Stop latency includes the remainder
  of an update and potentially a large merged-model export.
- This is cooperative stopping, not recovery from Ctrl+C, force-kill, OS shutdown,
  power loss or an accelerator hang. Periodic checkpoints and atomic publication
  remain separate unfinished work. Forced termination during a write can still
  leave a partial new file.

## Verification

The native control-pipe test exercises empty/nonblocking input, an ignored
unknown command, single-command consumption, EOF and regular-file rejection.
It runs in `train-accumulation` with no Python requirement. The optional Python
CTest driver starts the actual native executable; it is not a runtime dependency.

[`stop-workflows.json`](training-recovery-20260924/stop-workflows.json) records
54 real process runs. All four modes (random pretraining, LoRA CPT, SFT and DPO)
were stopped using a command, control EOF, and loss of both control/progress
readers. At each observed update count, losses, checkpoints and GGUFs were
byte-identical to the corresponding uninterrupted run. A fresh process resumed
each checkpoint for three more updates and again matched uninterrupted results.
Additional cases cover step-zero resume and saving to an existing GGUF path:
the original file survives, the checkpoint remains readable, and exit is failure.

The Rust tests execute the native trainer through the production stop/status
helpers, including duplicate requests and both successful and failed saves.
The trainer executable was supplied explicitly with `FYODOR_TEST_TRAINER`;
this was not the test's optional no-executable skip.

### Real-model interruption

[`real-stop.json`](training-recovery-20260924/real-stop.json) records TinyLlama
1.1B Q4_K_M, rank-2 LoRA CPT, context 8, accumulation 3, and a 512 MiB component
budget. Base SHA-256 is
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
The process received a stop after its first progress line, completed update 2,
and saved. Its GGUF and checkpoint exactly match the earlier uninterrupted
two-update accumulation run. A fresh process resumed for one further update;
all three losses and the final checkpoint/GGUF exactly match a fresh continuous
three-update run:

- Final GGUF: `a4fe4178535c24d0aa0355e016310dd641890aa7413d7eebd6558b9e70a10663`
- Final checkpoint: `b3fd1978e35711f7fa2701078ee6e434ad0c0b3c2f8778a747243cd1c34ae5ff`

Observed request-to-process-exit time was 28.43 seconds, including the remaining
update, checkpoint and approximately 4.4 GB merged F32 export. This is one
functional latency observation, not a repeated throughput benchmark or an upper
bound. The three phases ran serially after validation and before packaging.
The driver retained checkpoints and hashes and removed only its own large
exports after hashing. This tiny corpus/context validates real-model execution
and state recovery; it does not establish useful language-model convergence.

All 15 stages of the [full validation driver](training-recovery-20260924/validation.json)
passed. Test totals below exclude the four unavailable ROCm/MLX hardware tests:

| Configuration | Passed | Skipped |
|---|---:|---:|
| GCC C17 CPU Release | 45 | 4 |
| GCC C17 CUDA + Vulkan Release | 70 | 4 |
| GCC C23 CUDA + Vulkan Release | 70 | 4 |
| Optional ONNX adapter | 46 | 4 |
| Clang ASan + UBSan CPU | 45 | 4 |
| Source copy with optcpp physically absent | 45 | 4 |
| Source copy with CUDA physically absent | 45 | 4 |
| CUDA source copy with CUTLASS physically absent | 65 | 4 |
| Frontend and sidecar tests | 10 | 0 |
| Rust desktop release tests | 2 | 0 |

The new process-stop test ran in all eight native configurations. Vite production
build passed. [Source hashes](training-recovery-20260924/source-hashes.json)
identify the tested worktree; the underlying Git HEAD remains
`ec3f6fa0437a3069a8ccd599ffe5cb9ef6497667` with uncommitted work.

The host remains Windows 11, Ryzen 5 9600X, 32 GB RAM, RTX 5060 Ti 16 GB,
driver 610.88, GCC 16.2 Release C17/C23 with `-O3`. These changes do not modify
training kernels or claim a throughput improvement. The optional C++ removal
checks retain the portable C training path.
The CPU trainer imports only `bcrypt.dll`, `KERNEL32.dll` and `msvcrt.dll`, as
recorded in [native imports](training-recovery-20260924/native-imports.txt).

## Desktop package

The updated Tauri application and NSIS installer built successfully. Installer
size is 406,299,573 bytes and SHA-256 is
`bca745dd98af8b547ed41141448a81db8101e46ba4278b4c9ca756ee141e1e5a`.
All extracted sidecars, optional CUDA libraries and notices match the prepared
resources byte-for-byte. The extracted desktop executable matches the release
binary after exactly one expected Tauri `UNK` to `NSS` bundle marker change.
The packaged trainer independently passed the complete 54-process stop suite.
The Rust stop/status tests also passed against the extracted trainer, recorded
in `training-recovery-20260924/rust-packaged-tests.log`.
[Package evidence](training-recovery-20260924/desktop-package.json) records all
hashes and commands. The installer was extracted and tested, not installed.
The preceding accumulation installer is preserved locally as
`.tools/Fyodor-accumulation-before-safe-stop-setup.exe`.

## Coverage boundaries and provenance

Windows pipe execution is tested. The POSIX implementation uses `poll`, `read`
and `SIGPIPE` suppression only in control mode; it has not been executed on this
host. WSL enumeration fails with “The system cannot find the path specified.”
The browser/native UI automation initialization failure recorded in the prior
report persists as a coverage limitation: no interactive window-close or visual
QA is claimed here. Rust process tests, event-handler compilation and production
frontend build are narrower evidence.

The implementation is original C and Rust; no upstream source was copied.
PyTorch's [checkpoint guidance](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
was reviewed for separating a consistent state snapshot from ongoing mutation.
This runner saves synchronously after an optimizer boundary and imports none of
the distributed/asynchronous machinery. Tauri's locally installed 2.11.5
`src/app.rs` was checked for `prevent_close`, `prevent_exit`, and the distinction
between a normal exit and restart. No new runtime dependency was introduced.
