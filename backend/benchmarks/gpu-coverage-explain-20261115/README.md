# Coverage rejection explanation

Run `python -B backend/benchmarks/gpu-coverage-explain-20261115/explain.py` from the repository root. This reads the retained paired long diagnostic results and their original coverage audits. It does not collect GPU telemetry or change any acceptance rule.

The report separates capture-wide collector errors, error calls overlapping a measured interval, requested-stop records, retained collector failure flags, and unattributed PIDs. Both captures contain 12 decode intervals. None overlaps a recorded failed collector call, although the original auditor reports a collector error for every interval because it checks the entire capture. Both have requested-stop end records; the original termination label conflates a retained failure flag with unsuccessful stopping. PID 4 is the sole unattributed PID in these audits.

These findings do not prove that the intervals have complete valid samples, that System activity is unrelated to the benchmark, or that measurements are uncontaminated. In particular, counter errors can affect subsequent samples. The report preserves original coverage decisions and never accepts a benchmark. Existing invalid-counter and attribution rejections remain unresolved. No score changes follow from this diagnostic.

The unit check verifies separation of error-call timing, orderly stop records, retained failure state and PID attribution, while preserving rejection. The report records its input hashes. All original collector and audit files remain unchanged so historical evidence remains reproducible.
