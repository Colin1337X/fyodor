# Improved FDPP versus native decode, October 17, 2026

The preceding experiment established a 2.55% short-decode improvement over
the previous FDPP schedule. That is not evidence of a win over native decode.
This experiment directly compares native execution with FDPP plus
NYA_CUDA_FDPP_SHORT_SPLIT=1, using the same executable and model.

The initial comparison uses the existing eight-activity protocol: prefixes
0/1024, 128 decoded tokens, two warmups, five repetitions, both variant orders,
F32 KV, six CPU threads, batch512 and host logits every decoded token.
All observations are retained, including high-variance runs. Native default
execution is not changed by this experiment.

The first short FDPP process reports 34.36 tok/s sample standard deviation.
Before any additional run, a follow-up protocol was chosen: short decode only,
ten warmups and twenty repetitions per process, both variant orders. It adds
system CPU busy counters and GPU P-state, SM/memory clocks and clock-event
reasons to the existing monitor. These are diagnostics, not new or relaxed
acceptance gates. GetSystemTimes supplies cumulative idle/kernel/user times;
kernel includes idle, so busy fraction is one minus idle delta divided by
kernel-plus-user delta. The first sample has no calculated busy fraction.

The monitor samples startup/loading as well as timed decode. Its observations
cannot pinpoint the cause of an individual slow token or repetition. Neither
CPU load nor clock variation alone proves external contamination. The initial
results are not discarded or overwritten by the longer run.

## Contamination discovered and corrected

RobloxPlayerBeta.exe (PID6288) was present before the initial trial. The old
monitor admitted every pre-existing C+G identity as desktop activity, including
this game. Both `clean` (a historical directory name, not an acceptance claim)
and the interrupted `longer-short` attempt are excluded. Their raw results are
retained; excluded-initial-summary.json explicitly records the exclusion.
The benchmark owner was stopped and Roblox was closed under the user's prior
authorization. Before/after process snapshots and contamination-audit.json
record the cleanup. No OS or development process was terminated.

The shared benchmark helper now admits only explicitly named OS/development
desktop surfaces automatically. Unknown graphics processes, games, overlays
and other compute processes are rejected even when present before the trial.
Regression checks cover Windows and slash-separated paths, case folding,
allowed desktop applications, compute-only processes and known/unknown games.
The additional CPU/clock diagnostics remain local to this experiment.

The preceding direct and partition FDPP experiments, and the accepted
llama.cpp reference trial, had empty initial GPU-process lists. This finding
does not invalidate those recorded comparisons. Their original monitoring
still rejects newly appearing identities.

## Post-cleanup result

Both clean protocols use ten warmups and twenty repetitions in each order,
forty observations per variant/context. All eight activities pass the existing
idle and competitor gates. No samples are removed. Runs completed before the
session interruption; the saved outputs and monitor records were validated
on October 26, without rerunning or treating the old process handle as live.

| Prefix | Native tok/s | FDPP short-split tok/s | Difference |
| --- | ---: | ---: | ---: |
| 0 | 352.665 +/- 2.308 | 360.821 +/- 2.597 | +2.31% |
| 1024 | 303.502 +/- 5.796 | 296.422 +/- 1.157 | -2.33% |

Short FDPP is faster in this clean comparison; full-context FDPP still loses
on long decode. Default inference remains native. The next integration should
evaluate selective short-context dispatch while retaining native long decode,
then refresh an equivalent llama.cpp comparison. These self-comparisons do not
replace the accepted 93.18% short / 114.31% long llama.cpp ratios. Neither the
125% target nor Vulkan FDPP, AMD execution or full FDPP pipeline work is done.
