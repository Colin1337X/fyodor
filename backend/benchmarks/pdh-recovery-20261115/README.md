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
