# Initial matched timing diagnostics

`timing.py` compares native whole-update CSV timing with synchronized PyTorch
whole-update wall time. Both execute the same random-initialized byte decoder,
corpus windows, optimizer settings, 30 updates, and six CPU workers. First ten
updates and the last diagnostic update are excluded. Four repetitions alternate
native/PyTorch order. Static RoPE and causal masks are cached in the reference.
PyTorch 2.11.0+cu128 uses eager F32 operations with TF32 disabled, double RMS,
RoPE and clipping, and stock fused CUDA AdamW. This is a transparent reference,
not a claim about optimized or compiled PyTorch training generally.

The width64/FF176/two-layer/four-head/two-KV-head/context32 CUDA case passed
all losses and final parameter/gradient/first-moment/second-moment checks at
the unchanged 1e-5 tolerance. Four throughput ratios were 67.12%, 66.52%,
61.78%, and 66.12% (mean 65.38%). These are **diagnostic, provisional timings**:
there was no concurrent GPU activity/resource monitor, and the reference and
native process lifetimes differ. They are not accepted clean competitive scores.
The first attempt stopped on a reference RoPE reshape bug; the corrected
four-run evidence is `timing-64-cuda-diagnostic.json`.

The width256/FF704/two-layer/eight-head/two-KV-head/context128 CUDA case
failed correctness on its first pair. Maximum absolute loss error was
1.0485551e-5 and scaled parameter error was 2.0243995e-5; raw gradients and
both moments passed. Its speed is rejected, recorded as null, and the run
stops immediately. Evidence is `timing-256-cuda-rejected.json`. The cause has
not yet been isolated; do not infer a native or reference defect from this alone.
Further work must localize the first divergence without loosening tolerances.

Raw checkpoints, native CSVs/logs, and exports remain in
`D:/fyodor-validation/training-timing-64-cuda-v2-20261011` and
`D:/fyodor-validation/training-timing-256-cuda-20261011`. Each JSON records the
native binary and harness hashes, configuration, per-step reference timings,
losses, command, and correctness results. No production code changed here.
