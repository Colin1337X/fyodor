# Isolated Q4 four-row model candidate

Adds the validated Q4 four-row kernel on top of the Q6 paired-row/Q4 paired-row
candidate. Dispatch requires NYA_CUDA_Q4_QUAD_ROWS=1, successful function load,
integer interleaved execution, batch four, Q4_K type 12, 3840 columns, 15360 rows,
and native 144-byte-block stride. Existing BLAS precedence remains intact.
Only this expansion shape changes; Q6 and other Q4 shapes keep prior dispatch.
The existing provider/resident host sources remain C. Production is unchanged.

Build verifies prior candidate short memcheck and long generation, raw-kernel
numerical/sanitizer proofs, and both direct paired timing orders before adding
the new source. Isolated C17/C23 O3 builds passed. The actual extracted dispatch
predicate passed 3780 cases per standard with strict warnings as errors.

Lifecycle validation retains output tolerances, two resets, four tokens per
reset, graph reuse, single-bank downloads and input hashes. A positive and
well-formed four-row execution marker is mandatory alongside Q6/Q4/QKV markers.
C17 prefix-257 validation passed both resets, with maximum scaled logit error
2.19546455e-05 and 96 four-row capture/direct calls. All 910 hashes
verified. C23 lifecycle, generation, long lifecycle, full-model sanitization
and model timing are pending. No accepted score changes.

C23 prefix-257 lifecycle also passed both resets, with 96 four-row execution
markers counted and maximum scaled error 2.19546455e-05. All 910 hashes verified.
The C17 8192-prefix lifecycle is running. Generation requires all three lifecycle
proofs and exactly three positive four-row trace entries; eight offline marker
cases passed. Its missing-long-lifecycle refusal was exercised before any GPU
launch. The full-model sanitizer runner retains the same gates and requires
four-row enablement/execution; 13 validator tests pass, and its missing-generation
refusal was exercised. These are harness tests, not completed generation or
full-model sanitizer runs.

The monitored on/off wrapper now toggles only NYA_CUDA_Q4_QUAD_ROWS. Q6
paired rows, Q4 paired rows and QKV reuse remain enabled in both variants.
Exactly three positive four-row markers are required when enabled and none
when disabled; malformed entries fail either mode. Eight offline tests passed,
including independent Q6/Q4/QKV requirements. The missing-full-model-memcheck
refusal was exercised before creating a measurement folder or GPU workload.
This preparation does not resolve counter coverage or authorize accepting
unattributed activity. No monitored timing has been collected for this candidate.

C17 8192-prefix lifecycle passed both resets with four verified tokens each,
maximum scaled logit error 1.63989979e-05, 96 four-row capture/direct calls,
and retained graph reuse and rewind checks. All 910 hashes verified. This is
long-context correctness evidence; generation parity is the next gate.

C17 short generation passed all twelve ordered cases across ordinary, serial-two,
batch-two and joint-three modes, with 32 generated tokens matching the ordinary
baseline. Four-row execution markers were [96,96,96], alongside required Q6,
Q4 and QKV evidence and six reused graph plans. All 922 hashes verified. C23
short generation is running; full-model memcheck remains gated on that result.

The pinned llama.cpp reference wrapper is prepared for this candidate. It
retains source revision, prompt-token and output-byte parity, input hashes and
continuous GPU coverage checks. Its missing-memcheck refusal was exercised
before GPU work; no new reference timing is claimed.

C23 short generation also passed all twelve cases with [96,96,96] four-row
execution markers, required companion kernels and six reused graph plans.
All 922 hashes verified. The complete short-model memcheck is the next gate;
no accepted throughput change follows from generation parity alone.

Full-model short memcheck passed all twelve ordered generation cases with
output parity, required independent four-row/Q6/Q4/QKV execution evidence,
six reused graph plans and a unique zero-error sanitizer summary. Exit code
was zero, elapsed instrumented time 2590.5897336 seconds, and all 929 hashes
verified after completion. This is correctness evidence, not throughput.
Monitored short on/off timing is the next measurement; coverage policy remains
unchanged and long generation is still pending.

## First short model timing pair: diagnostic regression

EnabledA and DisabledA each completed all twelve output-parity cases with
unchanged inputs; all 960 hashes per run verified. Four-row markers were
[96,96,96] when enabled and absent when disabled. Q6/Q4/QKV and graph reuse
checks passed independently in both variants.

Joint decode-loop durations for 32 output tokens:

- EnabledA: 589.1529, 581.2256, 580.9656 ms; mean 583.7813667 ms, SD 4.6537004.
- DisabledA: 557.5781, 551.8586, 563.8468 ms; mean 557.7611667 ms, SD 5.9961963.

Enabled mean latency is 4.6651% higher. All joint intervals failed coverage
for unattributed System/PID 4 activity; the first interval in each run also
had an invalid target counter. These results are diagnostic, not accepted
performance. No llama.cpp score changes. Reverse-order measurement is next;
the isolated kernel gain does not justify production integration.

## Reverse-order attempt interrupted by collector failure

DisabledB stopped after 13 completed PDH samples. The next formatted-array
data call returned 0xc0000bba; the collector exited 1 and the runner terminated
its owned benchmark (PID 22984). Only ordinary, serial-two and batch-two cases
in round zero completed. No joint-decode comparison is available. The complete
parity/execution inventory is false, inputs remained unchanged, and all 959
recorded hashes verified. Evidence is retained as an incomplete rejected run.

The reverse pair is incomplete and must not be presented as confirmation of
the initial 4.6651% diagnostic regression. This candidate remains unpromoted:
the existing evidence provides no model-level benefit, and the continuous
coverage problem remains unresolved. Do not retry this measurement unchanged
or claim the isolated kernel gain as an inference throughput gain. Production
was never switched to this candidate, so no production rollback is required.
