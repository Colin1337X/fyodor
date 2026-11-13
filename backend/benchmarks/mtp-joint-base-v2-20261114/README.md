# Joint-base MTP v2: validate eligibility after prefill

This isolated C experiment retains the joint-base design described in the original
mtp-joint-base-20261114 directory and its full source patch. It corrects the v1
failure by rechecking resident verification, rewind and borrowing capabilities
after prefill, which can fall back to a host plan. If eligibility was lost, the
joint mode is disabled and the original proposal window restored before decode.
Production core, library and defaults remain unchanged.

C17 and C23 builds passed. Existing architecture tests passed in both standards
under CPU, GPU ring KV, GPU linear KV and forced CUDA failure. Joint selection
is required for ring/linear and forbidden for CPU/failure; result files retain
the commands, flags, exit codes and hashes. Existing sampled host/bridge parity
controls (16 seeds) continue to pass, while the experiment itself only changes
greedy assistant requests with a requested four-token window.

The real Gemma short-context correctness diagnostic completed all twelve cases:
ordinary, serial2, batch2 and joint3 over three rounds, with identical 32-token
outputs. Joint verification was selected in all three joint rounds. No continuous
GPU coverage was collected for this correctness diagnostic; recorded times are
not accepted performance. The long-context correctness run also completed all
twelve cases with identical 32-token outputs and joint selection in three rounds.
Neither correctness run supplies validated performance coverage.

The v1 failed evidence remains intact. run_model.py refuses existing outputs,
checks build and prerequisite hashes, and rejects competing GPU contexts before
launching the model. Never turn the numerical passes into a llama.cpp score.

## Additional stopping-boundary controls

edge_checks.inc compares ordinary and assistant generation at token limits
1, 2, 3, 4, 5, 8, 12, 16 and 17; with each of the eight vocabulary IDs as an
EOS token; and with output capacities 1, 2, 3, 4 and 8. Success requires equal
output bytes, generated counts and stop reasons, or matching errors. The suite
also requires actual error, EOS and context-limit coverage.

Both strict C17/C23 CPU fixtures passed all 22 cases: five matched errors,
three EOS outcomes and two context stops. CPU runs do not select joint GPU
verification. GPU ring, linear and forced-failure fixtures also passed all 22
cases in both standards, including the same error, EOS and context outcomes.

run_edge_memcheck.py ran the real Compute Sanitizer on ring and
linear fixtures in both standards. It requires ordinary edge success, matching
inputs, zero child exit, exactly one zero-error summary, all 22 edge cases, and
joint-mode selection. Five offline validator tests passed. Both ring and linear
runs completed in C17 and C23 with zero reported memory errors. These checks
cover the architecture and stopping-boundary fixtures; the full Gemma model has
not been run under this sanitizer. Production integration remains pending.

## Monitored short-context diagnostic

run_monitored.py requires both real-model correctness results, all CPU/GPU
fixture results, both sanitizer results and their unchanged input hashes. It
uses the shared runner's explicit joint-base mode inventory and continuous PDH
coverage, retaining runner status and evidence hashes even when coverage fails.
Invoke it with `short|long UNIQUE_LABEL`; existing outputs are refused.

monitored-short-A completed all twelve cases with output parity and exactly three
joint selections. The auditor recognized the expected joint-base case order.
Continuous coverage rejected the timings due to unattributed System activity
and invalid counter values. This is rejected diagnostic evidence, not a speedup
claim. All 41 monitor tests passed before the run. The accepted short/8K scores
remain 85.61025294% / 106.8707255% of llama.cpp.
