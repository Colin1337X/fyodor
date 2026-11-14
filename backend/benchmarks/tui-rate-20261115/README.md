# Learning-rate telemetry and TUI portability

Fresh Ubuntu 26.04 WSL C17/C23 Release builds contain the current learning-rate
CSV/readout and memory telemetry changes. Optional C++ directories are physically
absent while NYA_ENABLE_OPTCPP remains ON. CUDA, Vulkan, ROCm, MLX and ONNX are
disabled. Both trainer and TUI have no C++ runtime dependency; neither configure
uses a C++ compiler. Strict Fyodor warnings are enabled; the existing vendored
SQLite warning remains in the logs.

C17 initially passed all six selected CTests. C23 passed evaluation, learning-rate
resume, state/render, snapshot and theme CLI, but its PTY test failed: the child
exited successfully while the test had not yet read the terminal restoration
sequence. The test stopped polling on child exit even after a successful read,
which could leave more buffered PTY output unread. The updated test drains ready
output before treating an exited child as exhausted.

recheck.py retains both compiled binaries unchanged, reruns the five other tests,
and runs the updated external PTY test twenty times per standard. All checks pass.
The copied source trees retain the original test; verification.json explicitly
identifies the replacement test path and hashes it, along with the original
source trees, binaries, logs and dependency evidence. The initial C23 failure log
is preserved. audit.py independently verifies physical absence and dependencies.

These are focused CPU-only checks, not a full test-suite rerun or performance
evidence. The GPU full-model sanitizer continued independently. Existing earlier
removal-build evidence was not overwritten.
