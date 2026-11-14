# SIMD scratch alignment under Linux ASan

GCC `15.2.0-16ubuntu1` on Ubuntu 26.04/WSL, Ryzen 5 9600X.
The original C17 ASan/UBSan suite completed with 36/58 passes and 22 failures
or dependency skips (`tests.log`). AVX-512 functions faulted at an aligned
`vmovaps` store to a fake-stack address that was only 32-byte aligned.
The standalone original C reproducer also fails; explicitly aligning its
scratch array to 64 bytes makes the same computation return 120.
This is consistent with GCC PR120201 and the upstream alignment fix:
https://github.com/gcc-mirror/gcc/commit/1339a4b6e8b6911003b71430986e0a3be7841e91
No upstream source was copied.

`core/cpu_simd.inc` now explicitly aligns each reduction scratch array to its
vector width (32 bytes for AVX2, 64 for AVX-512), using standard C `_Alignas`.
The arithmetic, ISA dispatch, numerical tolerances and sanitizer settings are
unchanged. This does not require optional C++ components.

Validation after the change:
- C17: 58/58 CTests passed with ASan/UBSan, including use-after-return checking.
- C23: 58/58 passed under the same instrumentation.
- Windows C17: all 8 selected quantization/attention CTests passed.
- Deliberate out-of-bounds access in the aligned standalone reproduction still
  fails with UBSan; ASan-only also reports `stack-buffer-overflow`.

Configure from the backend source directory with:
```
cmake -S backend -B /home/colin/.cache/fyodor-asan-20261114-c17 \
  -DNYA_ENABLE_OPTCPP=OFF -DNYA_ENABLE_CUDA=OFF -DNYA_ENABLE_VULKAN=OFF \
  -DNYA_ENABLE_ROCM=OFF -DNYA_ENABLE_MLX=OFF -DNYA_WARNINGS_AS_ERRORS=ON \
  -DNYA_ENABLE_SANITIZERS=ON -DNYA_C_STANDARD=17
cmake --build /home/colin/.cache/fyodor-asan-20261114-c17 -j 4
ASAN_OPTIONS=detect_stack_use_after_return=1 ctest \
  --test-dir /home/colin/.cache/fyodor-asan-20261114-c17 --output-on-failure -j 2
```
For C23, reconfigure the same directory with `-DNYA_C_STANDARD=23`, rebuild,
and repeat CTest. The directory name retains `c17`; `c23-build.log` and
`c23-tests.log` correspond to the reconfigured C23 build.

Reproducer: compile `repro.c` with the flags in `repro-results.json`;
`-DEXPLICIT_ALIGN` selects the fixed case. Add `-DCHECK_REDZONE` to deliberately
access element 16. For the ASan-only diagnostic use `-fsanitize=address` instead
of `-fsanitize=address,undefined`. All executions explicitly set
`ASAN_OPTIONS=detect_stack_use_after_return=1`.

These are correctness checks while a separate Windows GPU memcheck is active.
No throughput measurements or accepted performance scores come from these runs.
The active GPU executable and its production C17 archive were not rebuilt.
