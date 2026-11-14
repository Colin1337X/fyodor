# Contiguous verification-logit readback (isolated experiment)

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

Short/long GPU lifecycle and generation validation have passed (details below).
Required next checks: full-model sanitizer validation and paired decode timing. This candidate is not production-integrated
and has no accepted performance claim. Accepted scores remain 85.61025294% short
and 106.8707255% at 8K versus llama.cpp, below the requested 125% target.

## CPU sanitizer evidence

`cpu-sanitizer.json` records passing Linux GCC C17/C23 ASan/UBSan executions of
the readback helper tests, with `detect_stack_use_after_return=1` and UBSan halt
on error. Build/run commands, logs and input/output hashes are retained. This
checks host memory handling in the mock harness, not CUDA execution safety.

`run_lifecycle.py c17|c23 257|8192` is prepared for the isolated real-model build.
It validates model identity and build hashes, retains logs and verifies inputs
again after exit. A passing run must satisfy the harness's exactly-one-download
assertion for four-row verification as well as the original numerical/lifecycle
checks. Short-context results are recorded below.

## Completed real-model lifecycle checks

`lifecycle-c17-257` and `lifecycle-c23-257` both exited zero. Four-token
verification passed both reset cycles at prefix 257 with maximum scaled error
2.19546455e-05 against the unchanged 0.0005 bound. Both harnesses required
exactly one download per four-row verification; reset, rewind, replacement,
stale-suffix and output-guard checks remained enabled. All recorded input/output
hashes were independently verified after completion. These are correctness and
dispatch-count checks, not measured throughput gains.

`lifecycle-c17-8192` also passed both reset cycles with maximum scaled error
1.63989979e-05 against the unchanged 0.0005 bound, exactly one download per
four-row verification, and 568 integer calls. All 47 recorded hashes independently
verified after exit zero. This extends the lifecycle evidence to prefix 8192;
full-generation checks also passed as recorded below; sanitizer validation remains pending.

## Completed short-context generation

`generation-short-c17` and `generation-short-c23` each passed all twelve
ordinary/serial2/batch2/joint3 cases in forward/reverse/forward order with
25 prompt tokens and 32 generated outputs. Both processes exited zero, and
each joint round recorded 2556 integer calls. All input/output hashes were
independently verified. `run_generation.py` requires explicit successful build
and short/long lifecycle proofs before execution, then verifies model identity,
process ownership, output parity, mode selection and unchanged inputs.
These correctness runs have no accepted timing claim. GPU memory validation
remains pending.

## Prepared full-model memory check

`run_model_memcheck.py short|long` runs the isolated generation executable under
Compute Sanitizer. It requires explicitly successful build, context-generation,
C23 short-generation and short/long lifecycle proofs, including the single-bank
download assertions. It retains transitive input/output hashes and requires exit
zero, exactly one zero-error summary, all twelve ordered matching cases and
three actual integer-kernel execution counts. Seven existing sanitizer-evidence
regression tests passed against this runner, covering failed exits, malformed
summaries, missing/reordered/wrong-context cases and absent kernel execution.
The short-context full-model sanitizer completed successfully in 2692.05 seconds:
exit zero, exactly one zero-error summary and all twelve matching generation
cases. Each joint round executed 2556 integer kernels. All 68 recorded hashes
independently verified after completion. This instrumented run is correctness
evidence only; none of its timings updates performance scores.

## Completed long-context generation

`generation-long-c17/result.json` records exit zero and all twelve matching
ordinary/serial2/batch2/joint3 cases at 8205 prompt tokens and 32 generated
outputs, in forward/reverse/forward order. Each joint round recorded 2556
integer calls. All 60 recorded input/output hashes independently verified after
completion. These unmonitored correctness timings do not update performance
scores. The full-model short-context sanitizer also passed, as recorded above.

## Prepared monitored comparison runner

`run_monitored.py short|long UNIQUE_LABEL` targets this exact readback binary.
Before GPU execution it requires successful build, completed short-context
full-model sanitizer, short/long lifecycle and short/long generation proofs,
including their transitive hashes. It retains model identity, process/idle gates,
continuous GPU coverage, output parity, actual integer dispatch and post-run
input-integrity checks. Three existing input-integrity regressions passed against
this runner, covering changed/deleted inputs and output inventory separation.
`monitored-short-A` completed all twelve matching generation cases and recorded
2556 integer calls per joint round. All 102 input/output hashes verified and
the runner reported no changed inputs. However, all twelve intervals failed
continuous coverage because of unattributed GPU activity (PID 4, Windows
System); one also contained an invalid target counter. The collector exited
cleanly, but the coverage audit returned failure. All timings are rejected for
performance acceptance. Do not kill System or silently allowlist its activity.
Neither successful monitoring nor generation timing alone updates accepted
decode scores.
