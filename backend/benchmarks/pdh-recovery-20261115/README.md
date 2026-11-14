# Diagnostic PDH recovery

Separate diagnostic collector derived from gpu-monitor-20261114. The original
collector and coverage audit are unchanged. The latest model-q4-batch-gate long
DisabledA capture failed after 175 attempts with formatted-array error
0xc0000bba; this collector permits further diagnostic observations after that
specific error instead of immediately aborting the workload.

Each failed attempt retains its index, QPC timestamps and error in the trace.
Three consecutive matching errors are fatal. A successful collection resets
only the consecutive count. Other exceptions and close failures remain fatal.
Any recovered error permanently sets end.failed, so cooperative process exit
zero does not mean clean coverage. The copied monitor wrapper still rejects
failed termination, and the original coverage audit rejects every error row.
System/PID 4 attribution and invalid counter values remain unresolved.

Run `python backend/benchmarks/pdh-recovery-20261115/test_recovery.py`.
Six tests execute the real collector with a fake provider: clean stop, isolated
recovery plus original audit rejection, three-error termination, successful
sample resetting the consecutive count, unknown error and close failure.
All six pass on Windows. These are offline fault-injection tests, not evidence
of clean GPU monitoring or improved decode performance. Live runner integration
and a new long comparison remain pending.

Live integration now uses run_cost.py and run_monitored.py, with the original
model prerequisites and an independent post-run coverage audit even when the
monitor rejects completion. audit_settings.py verifies both effective variants
against completed C17/C23 generation proofs. The DisabledRecoveryA long baseline
was launched; its outcome is pending and its live artifacts are not committed.

Microsoft documents 0xC0000BBA as PDH_CSTATUS_INVALID_DATA (returned data is
invalid), distinct from PDH_RETRY and buffer-sizing PDH_MORE_DATA. Recovery here
is a diagnostic policy, not a documented promise that a subsequent sample will
succeed. The code alone does not establish a driver defect, another application,
or a counter-instance lifetime as the cause. References:
https://learn.microsoft.com/en-us/windows/win32/perfctrs/pdh-error-codes
https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw

DisabledRecoveryA completed all twelve long cases with parity and validated
execution inventories. All 1011 recorded identities verify. Three joint decode
loops took 812.4364, 794.8721 and 807.5827 ms per 32 tokens (mean 804.9637333,
sample SD 9.0703031). Six isolated PDH errors were retained; the collector
completed cooperative shutdown with failed=true. Every joint interval fails
coverage, also reporting unattributed activity. No accepted score changes.
The enabled counterpart is the next comparison; this is not a measured gain.
