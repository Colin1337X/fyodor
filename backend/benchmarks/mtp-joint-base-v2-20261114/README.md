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
cover the architecture and stopping-boundary fixtures. The separate full-Gemma
short-context sanitizer described below also passed. Production integration
remains pending.

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

## Full-model memory-safety validation

run_model_memcheck.py runs the unchanged generation binary under actual Compute
Sanitizer, after verifying recorded build, correctness and fixture evidence.
It reuses the monitored run's CUDA settings and requires all twelve cases in
order, 32 generated tokens, matching output, three joint-mode selections, a
zero exit status and exactly one zero-error sanitizer summary. Five offline
validator tests pass, including incomplete runs, bad summaries, wrong workload
and absent dispatch. `python run_model_memcheck.py short` starts the short case
from the repository root when supplied with the script's full relative path.

The short full-model run completed in 2614.43 seconds with exit code zero,
all twelve cases in order, output parity, all three joint-mode selections,
and one Compute Sanitizer summary reporting zero errors. The terminal result
and all nine recorded evidence hashes were verified after completion.
running.json remains only a launch record; result.json establishes completion.
The long full-model sanitizer has not run. Sanitizer timings are not benchmarks.

## Independent reference preparation

run_reference.py adapts the existing aligned-prefix reference runner to this
experiment's completed correctness and full-model short sanitizer evidence.
It accepts `short|long UNIQUE_LABEL`, validates the unchanged native/reference
builds and prompt-token evidence, checks exact generated text against the native
correctness run, and retains continuous GPU coverage. It cannot accept a score.
The existing 14 reference build inputs and 22 prompt-token proof inputs were
rechecked with no changes. Six reference interval tests and four comparison
tests passed. Its GPU exclusivity check refused the active sanitizer before
creating an output folder. After sanitizer completion, reference-short-A failed
before inference because backend discovery did not search PATH for CUDA0. The
unchanged reference binary was then launched from its recorded DLL directory,
matching upstream's executable/current-directory backend discovery. That child
working directory and every DLL hash are retained in reference-short-B.

reference-short-B completed all three rounds with matching prompt token IDs and
exact native output bytes. The collector stopped successfully, but System PID 4
activity invalidated rounds zero and two; round one alone had clear sampled
coverage. No score is accepted and no clean round is cherry-picked. This shows
the unknown activity also occurs in the external reference; it does not establish
ownership or justify relaxing coverage rules. Both attempts are preserved.

reference-long-A also completed three rounds with identical 8205 prompt IDs and
matching 32-token output. System activity invalidated rounds zero and two, and
round zero additionally contained an invalid counter. Round one had clear sampled
coverage. The collector stopped successfully; no long score is accepted.

## Quiet-gap attribution control

build_guarded_reference.py builds a separate C17/C23 reference harness with
two-second Sleep calls immediately before the start timestamp and after the end
timestamp. The decode work and measured endpoints are unchanged. Select it with
the final `guarded` argument to run_reference.py. The gaps isolate sampled decode
brackets from neighboring prefill/cleanup work, but can change GPU clocks; these
runs are rejected by compare_monitored.py for performance comparisons.

reference-guarded-short-A completed all three rounds with matching output and a
healthy collector. Round zero still had System activity and an invalid counter;
round two had an invalid counter, while round one had clear sampled coverage.
This does not establish ownership or clear the environment. The native scores
remain unchanged. Strict C17/C23 builds and comparison rejection tests passed.

The reference runner also re-hashes both external GGUF files with bounded
memory before launch. model_identity.py compares them with the previously
recorded target/assistant SHA-256 values and rejects missing, replaced or
changing files. Tests cover missing files and replacement with equal-length
content. model-identity-observation.json confirms both files still match during
the active sanitizer run; it is not retroactive pre-launch evidence for that run.
The native launchers should adopt this check after the live run completes, so
its hash-bound source inputs remain unchanged while it is executing.

## Comparison gate

compare_monitored.py accepts native/reference evidence folders and a new output
path. It verifies evidence hashes, both idle gates, per-case continuous coverage,
matching contexts, the explicit joint-base case inventory and generated text.
It emits no ratios when coverage is contaminated or incomplete. A successful
comparison remains diagnostic: two warmed observations do not establish reversed
cross-engine order or CPU/thermal equivalence, and cannot replace the accepted
fixed-forward scores. Five shared comparison tests and two coverage-gate tests
pass. comparison-rejected-short-A.json records the expected refusal of the
existing contaminated native run before any reference comparison.
