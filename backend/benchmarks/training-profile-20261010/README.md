# Training profile and competitive scorecard — October 10, 2026

Completed diagnostic baseline on top of `83719af`. The 18-run public CLI diagnostic uses
TinyLlama Q4_K_M, rank-2 LoRA, contexts 8/32/128, three repetitions and alternating
CPU/CUDA order. Each run performs five updates; the first two are warmup. CSV
phase events, process memory samples, whole-device GPU telemetry, model and
checkpoint hashes are retained. Exports are hashed then removed individually;
checkpoints stay on D:. No model files are committed.

NVIDIA overlays restarted after authorized termination. A PyTorch reference
installation overlaps later runs. This series is diagnostic, not accepted
competitive timing. `diagnostic-summary.json` states that exclusion explicitly.
Its first CPU/GPU pair at each context passes full parameter/gradient/moment/
variance comparison with the existing scaled 1e-5 bound. Maximum errors are
3.64e-12, 2.32e-10 and 6.57e-7 respectively. Same-device repeated checkpoints
match exactly across all three repetitions. All 18 runs completed and the final
diagnostic summary is recorded.

Current phase evidence points to forward execution, followed by backward,
as the main GPU cost; optimizer time is below one millisecond in these runs.
Source inspection shows ordered 32x32 tiled forward dots in CUDA/training.cu.
Short sequences underuse the token tile. This is a hypothesis to investigate
with operation-level measurements, not proof of an accepted optimization.

The independent PyTorch reference lives in D:/fyodor-validation/pytorch-reference.
It is development tooling only, installed from the official CUDA 12.8 wheel
index. The resolver selected torch 2.11.0+cu128 for Python 3.14. Installation completed; both CPU and CUDA tiny oracle runs passed. The tiny oracle starts from a native
step-zero checkpoint, matches byte tokens, GQA, interleaved RoPE, RMSNorm,
SwiGLU, causal attention and AdamW settings, and checks all four state arrays.
The unchanged 1e-5 gate passed on CPU and CUDA. Maximum state errors were
7.27e-7 (CPU) and 1.20e-6 (CUDA); loss errors were 2.26e-9 and 4.75e-7. This is
one complete tiny-decoder update, not multi-step/real-model PyTorch validation
or a throughput comparison.

The user's requested ongoing percentages are defined in scorecard.md and backed
by scorecard.json. Historical inference scores use the later matched September 26
C API comparison, not the older mismatched overview table. They must be refreshed
for current-build claims. PyTorch training has no competitive score yet.


The first oracle attempt failed because the local `profile.py` benchmark shadowed
Python's standard-library module during PyTorch optimizer initialization. It was
renamed to `native_profile.py`; the failed log is retained. The initial one-device
oracle and the final explicit CPU/CUDA runs are all retained. NumPy is absent;
the reference uses standard-library checkpoint decoding and torch tensors, and
its NumPy warning does not affect these executed operations. No tolerance changed.

Next required work is multi-step PyTorch parity, matched timings with native and
framework implementations, real-model reference validation, and operation-level
profiling. The current timings identify likely costs but cannot establish a clean
competitive percentage. Historical inference scores remain unchanged.
