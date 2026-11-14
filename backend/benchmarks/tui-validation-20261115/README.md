# Current TUI validation with optional C++ source absent

Fresh Linux source copies include the validation-loss graph and latest-evaluation
readout. Root/backend optcpp and CUDA/cutlass/cute are physically absent before
configuration. NYA_ENABLE_OPTCPP remains ON to exercise its absence fallback;
CUDA, Vulkan, ROCm, MLX and ONNX are disabled for these CPU-only TUI checks.

Ubuntu 26.04 WSL C17 and C23 Release builds both pass all four selected tests:
state/render, snapshot, theme CLI and the real POSIX PTY interaction test.
Strict warnings are enabled for Fyodor. The vendored SQLite compiler warning
remains visible in the logs. Neither build configures a C++ compiler, and ldd
reports no C++ runtime dependency for the TUI. Manifests retain source, executable,
cache and log hashes. Existing removal-build artifacts were not overwritten.

The latest-evaluation regression checks that a later non-evaluation row retains
the earlier value with its step label, graph gaps remain empty, and replacing
the metrics history clears the old readout. The same current state/render suite
also passed Windows GCC C17/C23 with strict warnings against the unchanged
production archive.

Run linux.py 17 or linux.py 23 under WSL from the engineering checkout to create
the corresponding fresh tree. Existing evidence/work directories are refused.
These are TUI-only portability results, not a full-suite rerun, backend validation
or performance evidence. GPU full-model sanitization ran independently throughout.
