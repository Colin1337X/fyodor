# Verbose lifecycle diagnostics

Separate diagnostic binaries preserve the active cached-query memcheck's inputs.
Run `python backend/benchmarks/lifecycle-verbose-20261115/build.py` from the repository root on a fresh build directory, then `python backend/benchmarks/lifecycle-verbose-20261115/check.py`.

Each potentially lengthy lifecycle operation emits a flushed stderr begin/end record with wall-clock Unix seconds and the original source line. `build.json` maps those lines to exact statements. An unmatched begin locates an unfinished or failed operation; it does not establish a hang. Inline conditional operations may be skipped, and an inline loop is logged as one stage. These are phase logs, not per-token progress or throughput measurements. Cleanup ends with the original return and has only a begin record.

The existing active executable cannot acquire these logs retroactively. Use the separately named `build-cuda/lifecycle-verbose-v2-20261115/lifecycle-c17.exe` or `lifecycle-c23.exe` for a subsequent diagnostic run, with the same model, prefix and validated environment. Do not run concurrently with another GPU evaluation. No GPU validation of this instrumented version has been claimed.

Both strict optimized C17/C23 builds pass. The check reconstructs the original source after removing logging and exercises early malformed-model rejection in both executables. The first build attempt used `timespec_get`, which this Windows toolchain does not expose; that attempt remains in the first build directory. The successful version uses portable `time(NULL)` with one-second wall-clock resolution, unsuitable for latency measurement.
