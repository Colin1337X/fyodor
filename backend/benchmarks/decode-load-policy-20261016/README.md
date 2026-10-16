# Rejected cache-load policy probes, October 16, 2026

Original Fyodor q4/q6 GEMV helpers were probed with cg/cs/ca byte payload loads.
No production matrix kernel was changed. Metadata and half loads stayed intact.
The first script attempt shadowed a CUDA loader callable with a load-policy name;
renaming that local fixed the harness before results were written.

Hot-cache and rotating 64 MiB+ bank cases compare exact outputs for every bank.
Forward/reverse graph timing uses 50 launches, three warmups and five samples
of ten replays per order. The fused gate/up diagnostic checks three shapes.
Snapshots document GPU processes, but these are not continuously monitored
acceptance benchmarks and do not update the scorecard.

Hot Q4 changes were mixed and output projection regressed. Hot Q6 mostly
regressed (up to about 16%). Rotating Q4 gains were at most 1.25% on internal
shapes; large shapes regressed. Q6 cg regressed 3-8%; cs/ca had mixed small
changes with hot-cache regressions. All gate/up policies regressed about 4-6%
on the target 5632x2048 shape. General policy substitution was rejected.
Baseline source, scripts, raw samples and exactness results remain here.
