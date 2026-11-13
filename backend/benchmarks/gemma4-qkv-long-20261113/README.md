# Gemma QKV long-context validation — 2026-11-13

The real-model oracle compares separate and fused Q/K/V projections at total
lengths 4097 and 8193, after batched prefill and two individual decode steps.
Both cases match all logits exactly; the existing 0.001 scaled-error bound is
unchanged. `oracle.log` and `oracle-exit.json` retain results, binary identity
and environment. This uses the original native-weight fallback when the packed
copy cannot fit, with the same F32 KV and wide split attention in both modes.

Two competing KoboldCpp CUDA processes were stopped with prior user authority;
their identities are retained. The matched timing attempt was rejected before
starting any timed process because the strict environment gate found additional
GPU applications. `reference/environment.json` preserves the rejected state.
There are no accepted timing results here. Earlier 4K/8K scores remain unchanged;
no QKV speedup is inferred from numerical equality.

`reference.py` prepares the 4096/8192-prefix, 32-token decode comparison in both
orders, with exactly the same flags as the accepted short/1K QKV experiment.
It must be run in a fresh output directory after environment cleanup. The goal
remains at least 125% of equivalent llama.cpp decode on short and long contexts.
