# Linux physical-removal validation

Ubuntu 26.04 WSL, GCC 15.2.0 and CMake 4.2.3. Both C17 and C23 completed the
existing `verify-without-non-c` target after the strict socket-length portability
fix. The target makes isolated copies; it never removes user checkout files.

Each copy physically lacks root/backend optcpp and CUDA/cutlass/cute. Optional
optimization flags remain enabled so absence exercises graceful configure/build
fallback. CUDA and Vulkan are disabled; the optional C ROCm/MLX loaders remain.
Both copies build with strict warnings enabled, then run 82 registered tests:
78 pass and four unavailable ROCm/MLX hardware tests skip. The suite covers
training/export/reload, inference fixtures, HTTP/API, storage, CLI and POSIX TUI.
This is not evidence of real-model GPU performance or all-backend functionality.

`audit.py c17` and `audit.py c23` independently check physical absence, the C
standard, enabled optional flags, lack of a configured C++ compiler, and `ldd`
output for server, trainer, benchmark and TUI. No C++ runtime dependency is
reported. JSON retains source-file hashes, build/log hashes, test counts and
linkage evidence. The vendored SQLite build emits a GCC string-overread warning;
Fyodor's strict-warning targets pass.

Reproduce by configuring a Linux CPU build with `NYA_C_STANDARD=17` (then 23),
`NYA_WARNINGS_AS_ERRORS=ON`, `NYA_ENABLE_OPTCPP=OFF`, and CUDA/Vulkan/ROCm/MLX
OFF, building it, then running `cmake --build BUILD --target verify-without-non-c`.
The child target restores optional switches to ON while omitting their source.

The target's tiny synthetic before/after benchmark is only a smoke check. It
ran while a separate GPU sanitizer was active, so no performance conclusion is
accepted from those numbers. Accepted Gemma decode remains 85.61025294% short
and 106.8707255% at 8K versus llama.cpp; 125% has not been achieved.
