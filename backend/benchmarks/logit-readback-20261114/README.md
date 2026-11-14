# Contiguous verification-logit readback (unvalidated GPU experiment)

Inspection of `CUDA/resident.inc` found that the two/four-row batched output
projection synchronizes and downloads each contiguous vocabulary row separately.
This experiment performs one stream synchronization and one contiguous download
for that already-batched path. It retains validation of every value before any
suppression, then applies suppressed-token masks separately to every row.
Single-row and serial-projection paths remain unchanged. Transfer bytes do not
change; the proposed benefit is fewer host/driver calls, not reduced bandwidth.

`readback.inc` is original C. It validates output, row capacity, vocabulary size
and multiplication overflow before calling the backend. `test_readback.c` uses
mock backend callbacks and guarded output buffers to check rows one through four,
NaN/positive and negative infinity in every element (including suppressed IDs),
per-row masking, invalid dimensions, and synchronization/download failures.
Strict C17/C23 `-O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror` builds
and both test executions passed on Windows. These CPU tests do not validate GPU
ordering or actual transfer behavior.

`build.py` creates a separate copy of the FFN input-reuse candidate, checks its
transitive build hashes, inserts the helper and changes only the batched bank
readback call. C17/C23 host compilation and four generation/lifecycle links pass
(`build.json`). The isolated four-token lifecycle harness requires batch-logit
mode, adequate batch capacity, and exactly one download instead of four, while
retaining its numerical, reset/rewind, guard and replacement assertions.

No GPU executable from this experiment has run. The existing FFN-reuse long
context generation job remains first in the GPU queue. Required next checks:
real-model lifecycle/output parity, physical transfer-count evidence, sanitizer
validation and paired decode timing. This candidate is not production-integrated
and has no accepted performance claim. Accepted scores remain 85.61025294% short
and 106.8707255% at 8K versus llama.cpp, below the requested 125% target.
