# MTP generation cost investigation, 2026-11-13

Diagnostics only, not accepted llama.cpp scores. The unchanged desktop gate
rejects retained GPU identities. Roblox and Powder Toy were stopped before this
investigation; Powder Toy reopened during the older B8 run and was stopped again
(process-cleanup.json). OS, development, remote-access, browser and download
processes remain. Accepted scores stay 85.6103% short / 106.8707% at 8K.

## Current matched B512 decode probes

The standalone C harness loads Gemma4 12B Q4_K_M and its MTP assistant once per
process. It compares ordinary generation, serial MTP window 2, and batched MTP
windows 2/4, with greedy sampling, seed 42, 32 output tokens and F32 KV. Prompts
have 25 or 8205 tokens. Three rounds run in forward/reverse/forward order; round
zero is warmup. Every call checks exact text, token count and stop reason against
ordinary generation. The summarizer additionally requires matching reference
text across revisions, all expected rows and successful process exit codes.

build_probe.py creates two standalone binaries using the current engine library
and a separately compiled replacement llm_cpu object with the exact CMake flags.
Both probes call the existing native batched-prefill routine for ordinary prompts
as well as MTP. This adjustment exists ONLY in the benchmark objects: production
ordinary prefill dispatch is unchanged. Both request NYA_CUDA_BATCH=512; proposal
verification still passes its actual 2/4-token batch to the existing small-batch
kernel. Provider memory limits may reduce prefill capacity as usual. Neither
probe changes the decode algorithm, except that the baseline restores the unused
last assistant forward which the candidate removes. The exact two patches,
commands and source/binary/library hashes are retained in probe-build.json and
*-probe.patch. Both probes use the same high-resolution QPC timer.

Run build_probe.py after building build-cuda/libnya-engine.a, then run.py
b512-baseline build-cuda/mtp-cost-baseline-b512.exe followed by run.py
b512-candidate build-cuda/mtp-cost-candidate-b512.exe. Runners refuse to overwrite
existing evidence. Run summarize.py only after both complete. No compiler or
other GPU evaluation overlaps these timing processes.

Decode timing starts before the first sample from prefill logits and ends after
UTF-8 output validation. It includes sampling, assistant operations and transfers,
but excludes prefill, plan creation and destruction. The first emitted token
uses prefill logits; these numbers are NOT the fixed-forward llama.cpp benchmark
contract. Whole-call timings are probe costs, not production request latency,
because the probe standardizes prompt ingestion. Assistant timings accumulate
whole forward calls including host work and transfers; they exclude the target's
hidden read and target verification. These diagnostics do not prove a speedup
against llama.cpp or the 125% goal.

## Production change and validation

The next MTP round replaces assistant logits/hidden state from its target-derived
base, so the forward after the final proposal in a round is discarded. Skip that
forward. Standalone drafting and target verification remain unchanged. Opt-in
NYA_GENERATION_PROFILE=1 prints decode-loop and MTP assistant timing/counts;
ordinary operation does not take per-step profiling timestamps.

C17 and C23 each pass all nine selected CPU/Vulkan/CUDA generation, architecture,
MTP ring/linear, fallback and speculative tests, with profiling enabled. Existing
MTP checks cover greedy windows 1..8 and sixteen sampled seeds against independent
host execution. The CPU architecture test also passes under ASan/UBSan with
sampled MTP comparisons enabled. Logs are retained. Matched real-model B512 timing and cross-revision parity are complete. All 48
requests (two revisions, two contexts, three rounds, four modes) match their
ordinary reference, and reference text matches across revisions.

## Superseded B8 diagnostic

The root-level stdout/stderr/timing.json preserve the older diagnostic. All twelve
short calls passed; the long run was intentionally stopped after repeated prompt
ingestion took about 209-224 seconds per call. The queued validation script then
correctly refused to continue from a nonzero exit. b8-stop.json records this
intentional change of experiment; it was not a model correctness failure.

Older B8 warmed short decode-loop times: ordinary 766/766 ms; serial MTP 906/906;
batch-2 MTP 1156/1125; batch-4 MTP 1281/1297. First long calls: ordinary 906 ms,
serial MTP 1281, batch-2 1781, batch-4 1813. This did not show a MTP speedup.
The older timer used GetTickCount64, with roughly 15.6 ms effective resolution.
Preserve this limitation; do not mix those incomplete/coarse measurements into
the new QPC/B512 summary. baseline.patch and sources.json describe that older
binary, whereas *-probe.patch and probe-build.json describe the replacement pair.

## Next measured candidates (not implemented in runtime)

Source audit: CUDA/resident.inc cuda_plan_borrow_attention currently dispatches
only p->attention_ring/p->attention. The normal target decode path, in contrast,
selects p->wide_value for long positions at widths 256/512 when enabled.
Consequently NYA_CUDA_WIDE_VALUE=1 in this experiment accelerates target attention
but does not accelerate the assistant's borrowed-KV attention. Keep the current
candidate limited to removing unused assistant work and diagnostic timing;
measure borrowed-attention wide-value dispatch separately afterward.

A conservative next experiment can select the existing unsplit wide-value kernel
for prefix >256, width 256/512, honoring p->use_wide_value and physical-layout
selection. It must retain the explicit frozen prefix (request pointer NULL),
requested window, scale 1, ring slot addressing, retained-range checks, hidden
preservation, failure invalidation, and exact transfer/allocation counts. Do not
reuse the live decode request pointer: that would read the wrong prefix after
the target base token advances. Verify against the existing basic borrow path
at both widths, linear/ring layouts, rewind and long/sliding windows before
promoting dispatch. Split attention is a distinct numerical/scratch contract
and should not be silently included in this first comparison.

The standalone weight-reuse experiment in ../small-batch-reuse-20261113 has
not been compiled or run. It is outside runtime dispatch and the queued candidate
build. Its correctness and profiling jobs must wait until the existing timing
sequence exits; Python syntax checks alone do not validate its CUDA behavior.

## Completed matched B512 results

Mean decode-loop milliseconds, two warmed observations per cell. Initial round
excluded; full observations and sample standard deviations in summary.json.
This is diagnostic desktop timing, not an accepted llama.cpp comparison.

| Context | Mode | Before ms | After ms | Observed speed change |
| --- | --- | ---: | ---: | ---: |
| short | ordinary | 757.027 | 758.072 | -0.138% |
| short | mtp-serial2 | 905.028 | 855.298 | +5.814% |
| short | mtp-batch2 | 1136.854 | 1089.441 | +4.352% |
| short | mtp-batch4 | 1285.692 | 1254.843 | +2.458% |
| long | ordinary | 935.921 | 935.714 | +0.022% |
| long | mtp-serial2 | 1209.234 | 1115.543 | +8.399% |
| long | mtp-batch2 | 1784.089 | 1693.774 | +5.332% |
| long | mtp-batch4 | 1818.948 | 1757.924 | +3.471% |

The unchanged ordinary path is the negative control and stays within 0.14%.
Assistant forwards fall from 33 to 22 for window 2 and 40 to 32 for window 4.
MTP still trails ordinary generation on this workload. Removing discarded work
is useful but does not establish a speculative-decoding win or meet the 125%
llama.cpp goal. Larger sampled/quality studies and uncontaminated comparisons
remain outstanding; two repetitions do not establish broad workload performance.

## Clock header isolation correction

The first profiler included train_clock.h directly from llm_cpu.c. A later full
physical-removal build exposed Windows interface-macro leakage into the training
graph test, which includes that implementation file. The clock is now exposed as
nya_runtime_monotonic_seconds through the plain C runtime header; only runtime.c
includes the platform clock helper. Its QPC/monotonic implementation is unchanged.
The completed probe binaries retain their recorded hashes and timing semantics.

The corrected tree passes all 82 registered physical-removal tests (four expected
unavailable ROCm/MLX hardware skips), plus train/export/infer. Initial failure and
fixed logs are retained in ../small-batch-reuse-20261113/physical-removal*.log.
Three targeted tests each in C17/C23 pass with profiling enabled, including the
training graph interface; clock-api-*.log records them. This correction changes
header isolation, not model math, and no performance score is promoted.
