# Native Q4 batch-four gate/up fusion: offline candidate

The current decode profile identifies Q4 batch-four FFN expansion as the largest
matrix group. CUDA/resident.inc fuses Gemma gate/up only at count one; batched
execution uses separate gate/up projections followed by nya_vector GELU.

llama.cpp revision 5266f24da75dc449bd56cbed7addb9c8e4a6a73e was inspected locally:
ggml/src/ggml-cuda/mmvq.cu supports gate fusion and applies the GLU operation
after reducing both projections. Its MIT license was checked. This is an
algorithm reference, not a source transplant: q4_batch_gate.cu derives from
Fyodor's q4_pair_rows.cu and uses the existing nya_vector GELU expression.
No upstream implementation text was copied. Unlike llama.cpp's Q8 activation
dot-product route, this candidate retains interleaved F32 inputs and native
144-byte Q4_K blocks; it introduces no activation quantization contract.

One warp computes the same row from gate and up matrices, sharing each float4
input load across both projections. Per-projection accumulation and warp
reduction order follow the existing kernel. The output is the row-major gated
result for all four inputs. A future dispatch must require compatible Q4_K
gate/up shapes and strides, batch four, and the existing input layout; other
cases must retain fallback. No runtime dispatch or allocation changes exist.

NVRTC compilation and offline SM120 assembly pass. ptxas reports 80 registers,
no stack or spills, and no barriers for the candidate. These are static resource
reports, not measured occupancy or a speedup. No CUDA context was created.
The full-model sanitizer for the separate joint-one scheduling experiment
remains the only active GPU workload.

Required next evidence: compare against two existing native projections plus
the exact GELU operation, independent numerical bounds, guarded partial rows,
shape rejection, sanitizer checks, paired synthetic timing, then full-model
parity and monitored performance before considering integration. Fusion can
regress due to scheduling or activation latency; compilation alone establishes
neither correctness nor performance. Accepted scores remain unchanged.

## Guarded numerical harness prepared

check_kernel.py prepares 222 cases plus eight unsupported-shape checks. Gate and
up use independently generated native weights. Each separate projection must
meet the existing independent C float64 oracle bound; fused output must then be
byte-exact against those projections followed by the existing nya_vector GELU.
All final values must be finite. Inputs, both weight arrays, interleaved-input
guards and output guards are checked. Cases cover warp/block row boundaries,
the full 15360-by-3840 shape, finite half-coefficient extremes and zero, constant,
alternating, subnormal, large and impulse activations. Large activations are
bounded at 2^12 here to keep the gated product finite; this is not a claim of
all-input overflow handling.

The combined reference/candidate harness compiles offline, with all required
entry points present. preflight-refusal.json records actual refusal while the
joint-one full-model sanitizer prerequisite is missing, before importing the
context-creating CUDA helper. These are preparation checks, not numerical passes.
After that sanitizer completes successfully, run check_kernel.py without flags
for raw validation. Sanitizer labels only select separate evidence filenames;
they do not themselves invoke or certify Compute Sanitizer.

run_sanitizer.py invokes the actual Compute Sanitizer executable for memcheck,
racecheck or synccheck, requires the complete passing raw sweep and unchanged
inputs, records the child PID, and checks all 222 cases and eight rejections
again. Both projection bounds, exact fusion output and guards are mandatory.
The result requires one zero-error summary (or zero hazards/errors/warnings for
racecheck) and matching raw/sanitized source identities. Seven offline validator
tests pass. sanitizer-preflight-refusal.json records refusal before launching
the sanitizer because raw validation is still pending. No GPU sanitizer result
is claimed for this fusion candidate yet.

## Paired timing harness prepared

probe.py requires the complete raw result and all three passing sanitizer
results, with recorded input/evidence hashes checked before creating a context.
It compares native metadata projections plus nya_vector GELU with fusion,
including one input interleave in both operations. Six shapes include partial
rows and the full Gemma expansion. Each shape runs separate/fused, fused/separate,
fused/separate, separate/fused ordering, retaining 40 samples per variant, their
means and sample standard deviations. Each sample contains 100 graph operations;
each order has 1,000 warmup operations. Default-stream uploads are fenced before
nonblocking graph execution. Output buffers are reset before each order and must
match the separate-pipeline reference byte-for-byte, including guard bytes.

Syntax compilation and the missing-raw-result refusal pass; no timing execution
has occurred. This compares the native metadata baseline, not the separate
experimental pair-row kernel. Even a later synthetic gain must survive comparison
with relevant stronger candidates and full-model testing. Repeated weight banks
and snapshot-only activity checks make these timings diagnostic, never accepted
llama.cpp performance scores.

## Completed isolated validation and first timing

All 222 numerical cases and eight unsupported-shape cases pass. Both projection
oracles satisfy the unchanged float64 bounds (maximum fraction 0.05542385), and
all fused outputs are byte-exact against the separate native projections/GELU.
Memcheck, racecheck and synccheck each repeat the complete sweep with exit zero
and a unique zero-error/hazard summary. Their input and evidence hashes were
independently verified before timing.

The first six-shape timing completed with output/guard parity throughout:

| Columns | Rows | Separate mean ms | Fused mean ms | Diagnostic latency reduction |
| --- | --- | --- | --- | --- |
| 256 | 17 | 0.003714 | 0.002416 | 34.93% |
| 3840 | 512 | 0.013741 | 0.011310 | 17.69% |
| 3840 | 2048 | 0.038460 | 0.034592 | 10.06% |
| 3840 | 4096 | 0.067837 | 0.062376 | 8.05% |
| 3840 | 8192 | 0.156643 | 0.129620 | 17.25% |
| 3840 | 15360 | 0.291142 | 0.244302 | 16.09% |

All forty samples per variant and their standard deviations remain in timing.json.
This supports testing against the stronger experimental pair-row baseline next;
it does not justify production integration. Full-model correctness and monitored
performance remain pending. Accepted llama.cpp/PyTorch scores are unchanged.

## Completed comparison against paired projections

The independent pair-row comparison also passes all six shapes with byte-exact
outputs and guards. It uses the same four forward/reverse orders, forty samples
per variant, and complete interleave/projection/GELU operation. At the model's
3840-column, 15360-row expansion, two paired projections plus GELU average
0.2578501763 ms versus 0.2437933269 ms fused (5.4500% lower latency).
The complete samples, standard deviations and verified prerequisites are in
timing-pair.json. This remains synthetic diagnostic evidence, not an accepted
model throughput result. An isolated full-model integration is the next gate;
production dispatch and accepted scores remain unchanged.
