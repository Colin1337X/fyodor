# Training Unicode paths — September 24, 2026

Dates here use UTC; final packaging falls on September 25 in Asia/Seoul.

The recovery audit found that Windows training rejected a valid dataset in a
Korean/emoji-named directory. [The original failure](training-paths-20260924/before.json)
records exit 1 and no output model. The runner used narrow CRT command-line
arguments and narrow `fopen`/`_open`/`remove` calls; the inference file layer
already accepted UTF-8 paths. Fixing only a file-open call would still leave
arguments corrupted before the runner saw them.

The Windows trainer now starts at `wmain`, converts its UTF-16 arguments to UTF-8,
and enters the same private C runner as POSIX. MinGW builds select the wide CRT
startup with `-municode`. Only one platform entry point is compiled into the
trainer. No process-global locale or code-page setting changes. Dataset and
resume reads use the existing UTF-8 file layer. Exclusive binary creation and
failure cleanup now use shared UTF-8 helpers (`_wopen`/`_wremove` on Windows), so
metrics, checkpoints and GGUF output paths have the same interpretation.

Existing destinations remain protected by exclusive creation. Embedded file
bytes are unchanged, including binary newlines and NUL bytes. Invalid UTF-8
Windows paths are rejected before filesystem access. POSIX retains ordinary
byte-string filename behavior. The change does not alter model kernels, loss
weighting, checkpoint identity or optimization semantics.

## Evidence

The [process driver](../tests/train_paths.py) runs the actual trainer. Its
[24-run record](training-paths-20260924/paths.json) covers:

- Random pretraining, LoRA CPT, SFT and DPO, with spaces, Korean, accented Latin
  characters and non-BMP emoji in directory/file names.
- Unicode dataset, base-model, metrics, checkpoint, resume and output paths.
- Exact loss, checkpoint and GGUF equality for split versus uninterrupted runs.
- Moving identical input/base/checkpoint bytes between ASCII and Unicode paths
  without changing the resumed trajectory.
- Step-zero stop/resume, existing output/checkpoint/metrics preservation, and
  corrupt-checkpoint rejection before model creation.

The native C test independently exercises binary create/read/remove, size
queries, existing-file rejection and invalid UTF-8 rejection. It requires no
Python. The optional process driver is registered as `train-unicode-paths` when
Python is available; Python remains a test-only dependency. Both step-zero
process tests now fill the control pipe before launching the child, eliminating
a parent/child scheduling race from that coverage.

The Rust stop/status tests run the real native child with a Korean/emoji-named
temporary directory, including checkpoint saving and an existing-file failure.
`FYODOR_TEST_TRAINER` was supplied explicitly, so the native integration test
executed. [Rust log](training-paths-20260924/rust-tests.log).

### Real-model resume

The [TinyLlama run](training-paths-20260924/real-paths.json) resumed the existing
rank-2 CPT checkpoint using Korean/emoji directory names for the base, corpus,
resume checkpoint, metrics, new checkpoint and output GGUF. Context was 8,
accumulation 3, with a 512 MiB component budget. A new hard link named the
immutable Q4_K_M base without copying or modifying its bytes. One additional
update produced the exact previously recorded ASCII-path two-update results:

- Base SHA-256: `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`
- GGUF SHA-256: `ca9d1933acffdc0bca4bba383e8093480df12d02df005fe36fd01769123be824`
- Checkpoint SHA-256: `ae503658320cedd4ed8e75ba41db6e444df918eaa4f84eb17e13e9fbab2d1127`

The driver retained the checkpoint/metrics and removed only its own large
export after hashing. This is a real-model I/O and resume check, not a repeated
throughput measurement or evidence of useful convergence.

## Validation scope

All 15 stages of the [full validation matrix](training-paths-20260924/validation.json)
passed for the shared file-layer change. During that run the CLI implementation
was separated from its platform entry points to avoid having both `main` and
`wmain` present in a Windows trainer. A subsequent [entry-point validation](training-paths-20260924/entry-validation.json)
rebuilt the final trainer/test and passed all four affected CLI/control/Unicode
tests in every configuration, including the retained physical-removal source
copies. [Current source hashes](training-paths-20260924/source-hashes.json)
identify the final sources. No tolerance was changed.

| Configuration | Passed | Skipped |
|---|---:|---:|
| GCC C17 CPU Release | 46 | 4 |
| GCC C17 CUDA + Vulkan Release | 71 | 4 |
| GCC C23 CUDA + Vulkan Release | 71 | 4 |
| Optional ONNX adapter | 47 | 4 |
| Clang ASan + UBSan CPU | 46 | 4 |
| Source copy with optcpp physically absent | 46 | 4 |
| Source copy with CUDA physically absent | 46 | 4 |
| CUDA source copy with CUTLASS physically absent | 66 | 4 |
| Frontend/sidecar tests | 10 | 0 |
| Rust desktop release tests | 2 | 0 |

The four skips are unavailable ROCm/MLX hardware. Vite production build passed.
The final CPU trainer still imports only `bcrypt.dll`, `KERNEL32.dll` and
`msvcrt.dll`; no C++ runtime or shell library was introduced. This is a path
correctness change, not a throughput optimization or benchmark win.

Host/build context: Windows 11, Ryzen 5 9600X, 32 GB RAM, RTX 5060 Ti 16 GB,
driver 610.88, GCC 16.2 Release `-O3`, Git HEAD
`ec3f6fa0437a3069a8ccd599ffe5cb9ef6497667` plus the recorded uncommitted work.
POSIX execution, MSVC compilation and interactive desktop file-picker/window
coverage were not verified on this host. Prior UI-tool initialization failures
and unavailable WSL remain documented limitations, not passing coverage.

## Desktop package

The final Tauri/NSIS package built successfully: 406,305,472 bytes, SHA-256
`b0e267a336921f7ff3abf662d144f9a4eceade96e2b0333014baedd3b3028213`.
All extracted backend resources match the prepared files. The desktop matches
the release executable after the expected single Tauri `UNK` to `NSS` marker
replacement. The extracted trainer passed both the 24-run Unicode suite and
the 54-run stop/resume suite. [Package record](training-paths-20260924/desktop-package.json).
The installer was extracted and tested, not installed. Its predecessor remains
at `.tools/Fyodor-safe-stop-before-unicode-setup.exe`.
The Rust supervisor tests also passed against the extracted trainer with their
Unicode temporary directory: `training-paths-20260924/rust-packaged-tests.log`.

## Provenance and remaining work

The implementation is original Fyodor C. Microsoft's primary documentation for
[`wmain`](https://learn.microsoft.com/en-us/cpp/c-language/using-wmain?view=msvc-170)
and [`_wopen`](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/open-wopen?view=msvc-170)
was checked for wide arguments and exclusive-creation semantics. No upstream
source was copied. General file helpers remain native C.

This fixes the trainer's boundary, not every command-line executable in the
repository. Periodic/atomic checkpoint publication, evaluation, schedules and
GPU-resident training remain unfinished. The previously implemented cooperative
stop protocol still does not promise recovery from force-kill or power loss.
