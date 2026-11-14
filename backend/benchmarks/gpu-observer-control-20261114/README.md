# Idle observer control

This original diagnostic compares ten one-second PDH-only samples, ten with
an owned `nvidia-smi -q -x` helper per sample, and ten PDH-only samples after.
It records helper PIDs, exit status, XML output, counter statuses, timestamps,
adapter identity, and source hashes. No inference or GPU kernel is launched.

All 30 samples completed without collector errors. None contained positive
target-adapter activity or invalid target counters, including the owned-helper
phase. Thus this idle experiment did not reproduce the inference-time System
activity or invalid counters. It does not establish causal attribution under
inference load, eliminate observer effects under load, or certify any timing.

The earlier benchmark failures remain rejected. No monitoring threshold or
allowlist changed. `result.json` is diagnostic evidence, not a performance score.
