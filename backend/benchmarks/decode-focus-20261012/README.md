# Decode-only optimization focus — October 12, 2026

The user's current priority is exclusively beating llama.cpp in decode.
Suspend prefill and training optimization until matched decode performance is
above 100% of the pinned reference. Track short and long workloads separately;
a win on one workload does not establish a win on the other. The broader
runtime goal remains active, but this is the current ordering constraint.

The current accepted scores are 69.6% short decode (128 tokens, empty prefix)
and 89.8% long decode (128 tokens, prefix 1024). See ../SCORECARD.md. These use
TinyLlama Q4_K_M, F32 KV, six threads, batch 512 and pinned llama.cpp b10809,
with reference flash attention off. Do not change those settings to manufacture
a greater-than-100% score. Keep strict correctness gates and monitored paired
runs. Prefix preparation for long decode remains untimed.

Fresh diagnostic profiles of the restored production runtime at 4e1867f:

| Workload | Dot matrix intervals | Gate/up Q4_K 5632x2048 | Share |
| --- | ---: | ---: | ---: |
| Short | 1393.654 ms | 598.926 ms | 42.98% |
| Long | 1391.440 ms | 601.805 ms | 43.25% |

The long dot total also includes three initial prefix-final-logit projections.
Vendor GEMM intervals for prefix preparation are excluded from this table.
Profiles include one warmup and two measured sequences. Short has 59,520 dot
calls; long has 59,523 dot and 924 vendor calls. Counts were checked against
22 layers, seven projections per layer, vocabulary projection and 384 decoded
tokens. These event intervals disable graphs and can include host submission
gaps; they are diagnostic evidence, not updated throughput scores.

Next target: shared-input Q4_K gate/up dispatch, preserving each projection's
F32 accumulation order and the existing SiLU result. Compare a fused decode
path against the separate projection/activation path; retain only a measured
win after kernel, model, graph reuse, long-context and fallback validation.
Do not repeat the already rejected alternating two-chain dot accumulation
experiment without new evidence (../INFERENCE_DOT_PROBE_20260926.md).

Reproduce the profile, with no concurrent GPU tests or builds:

```text
python backend/benchmarks/decode-focus-20261012/profile.py NEW_DIRECTORY
```

Raw command/environment, model/executable hashes, per-shape times and aggregate
logs are in baseline/. Other desktop apps remained open. No performance
acceptance or zero-background-activity claim is made from these profiles.
