# Borrowed wide attention in full MTP generation

This experiment measures the existing opt-in borrowed wide-value attention path
after target wide-batch attention and greedy early rejection were integrated.
It addresses the approximately 70/103 ms of remaining borrowed-attention work
identified for MTP windows 2/4 in mtp-profile-20261114.

Run `python backend/benchmarks/mtp-borrow-generation-20261114/run.py` from the
repository root in a fresh evidence directory. The runner refuses to overwrite
build evidence. It compiles a unique executable using current production core
and library, with only diagnostic B512 ordinary prompt ingestion patched.
Commands and source/object/library/executable hashes are recorded in build.json.
Production prompt ingestion is unchanged.

The same executable runs borrowed wide attention off and on, at 25 and 8205
prompt tokens, with 32 greedy output tokens. Other optimization flags remain
identical. Each process executes ordinary, serial-window2, batch-window2 and
batch-window4 generation three times, reversing order in round 1. Round 0 is
warmup; both remaining observations are retained with sample standard deviation.
The measured decode loop includes sampling, assistant work and transfers, plus
the first token sampled from prefill logits. It excludes prefill and setup.
CUDA event profiling is disabled so graphs remain available.

Every call must match ordinary text, token count and stop reason. The summary
also requires identical reference text across processes, unchanged binary and
settings apart from the candidate flag, and actual borrowed-wide dispatch for
all nine long-context MTP calls. Short context and ordinary generation are
negative controls. Existing independent float64, linear/ring, rewind and CUDA
memcheck evidence is in mtp-borrow-wide-20261113.

This is an internal diagnostic, not a llama.cpp performance result. The shared
runner records conservative desktop-gate failures and process/resource snapshots.
Retained desktop applications prevent promoting the result to an accepted score.
Run order is off then on; within-process reversal does not eliminate possible
cross-process drift. Any apparent gain needs assessment against the controls.

## Results

All 48 generation calls passed exact parity. All nine optimized long-context MTP
calls reported borrowed-wide dispatch (80 calls for window 2, 116 for window 4);
short-context and disabled controls reported none. The summary checks passed,
including matched binary/settings and absence of CUDA event profiling.

Warmed decode-loop means and sample standard deviations, in milliseconds:

| Context/mode | Off mean +/- SD | On mean +/- SD | On speed / off speed |
| --- | ---: | ---: | ---: |
| Short ordinary | 755.665 +/- 0.048 | 757.845 +/- 0.829 | 99.712% |
| Short serial MTP | 845.919 +/- 1.045 | 847.213 +/- 1.447 | 99.847% |
| Short batch 2 | 822.180 +/- 3.664 | 821.727 +/- 1.752 | 100.055% |
| Short batch 4 | 827.196 +/- 1.012 | 829.658 +/- 0.802 | 99.703% |
| 8K ordinary | 934.645 +/- 0.345 | 934.865 +/- 0.866 | 99.976% |
| 8K serial MTP | 1097.549 +/- 0.367 | 1060.136 +/- 0.525 | 103.529% |
| 8K batch 2 | 1185.288 +/- 7.633 | 1142.879 +/- 2.387 | 103.711% |
| 8K batch 4 | 1154.168 +/- 1.563 | 1102.462 +/- 6.002 | 104.690% |

The candidate saves about 37/42/52 ms per 32-token 8K generation for serial,
batch-2 and batch-4 MTP respectively. Negative controls vary by less than 0.3%.
This supports retaining the independent opt-in optimization, but MTP is still
slower than ordinary generation in every measured case. The feature remains
opt-in; the result does not justify selecting MTP by default.

Both conservative desktop idle gates failed due to retained process identities;
all raw evidence is preserved. These two-observation diagnostics do not establish
clean-room throughput, general prompt/model performance or a llama.cpp win.
Accepted scores remain **85.6103% short / 106.8707% at 8K**, below **125%**.
The next optimization must reduce remaining assistant/verification cost or the
dominant ordinary matrix cost. Serial vocabulary projection in verification is
one measured candidate (71.65/84.46 ms for batch-2/4 in mtp-profile-20261114),
but any batching requires bounded scratch accounting, full-vocabulary parity,
suppressed-logit handling and fallback tests before performance evaluation.
