# Monitored Q4 gate/up unroll recheck, October 26

The current profile puts gate/up at about 40% of matrix time. Rechecked outer
unroll 1/2/4/8 with the current four-block launch bound. All outputs and guards
match the archived native kernel exactly across six shapes and eight rotating
banks per model-sized shape. All three monitored activities pass the idle and
competitor gates. Every sample and resource count is retained.

This reproduces the earlier decode-bound-unroll-20261015 rejection with stronger
environment monitoring and rotating weights. Unroll one loses about 3.4% on
TinyLlama rotating weights; unroll two is essentially tied and four loses 0.3%.
Keep eight. A prepared launch-bound recheck was discarded before execution after
reading decode-registers-20261015: that already rejects tighter bounds. Avoid
further compiler-hint tuning without a new mechanism or hardware evidence.

Scripts compile original diagnostic device code under root optcpp/matvec using
NVRTC compute_120, C++11, fmad enabled. CUDA graph timings have 50 launches per
graph, three warmup replays and five samples of ten replays in both orders.
These are isolated diagnostics, not full-model acceptance. Current whole-model
scores stay 94.70% short / 114.23% long versus llama.cpp; the 125% target is unmet.
Runtime/CMake sources are unchanged, so physical removal remains unaffected.

Run `python backend/benchmarks/decode-unroll-20261026/run.py` in a fresh copy
without its output directories to reproduce. RTX 5060 Ti, driver 617.42;
source hashes/options are in each results.json. Baseline source is archived.
