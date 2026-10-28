# Aligned Q4 gate launch experiments

Current packed-projection profile still identifies gate/up as the dominant matrix kernel (326.23 ms of 845.25 ms matrix time). Profile disables graphs and is diagnostic only. Accepted whole-model performance remains 110.5081% short / 129.4989% long on TinyLlama.

Original Fyodor source is generated from `optcpp/matvec/byte_scale_gate_aligned.cu` in the probes. Block geometry tests use 2/4/16 warps against the accepted 8; all preserve 32 desired resident warps in the launch-bound hint. Register experiments retain 8 warps per block and request 2/3/5/6 blocks per SM against the accepted 4. No production source or dispatch changed.

Both geometry runs and the register run passed process monitoring and bitwise output/guard checks for every rotating weight bank at 2048x257, 2048x5632 and 4096x11008. Each has ten timed samples in forward/reverse order. Raw native bank sizes exclude the additional packed allocation; packed bytes are native bytes * 160/144 and both copies coexist. Packing is outside timing.

The first small-shape baseline phase had a large timing anomaly (approximately 28 us versus 4 us in the reverse phase); all samples are retained. The full repeat stabilized it. At the production 2048x5632 shape, reducing to two warps yields under 1%, four warps less, and sixteen regresses. Register-limit alternatives did not produce an adopted improvement. No runtime integration or end-to-end claim is made. The new user request to test Gemma 4 takes priority over further launch tuning.

See `summary.json`, per-run resources, and monitor files. Generated sources are hashed by each probe; this is diagnostic evidence, not the full portability/correctness matrix.
