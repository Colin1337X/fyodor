# Assistant stage wall-time profile

The paired output-batching experiment still leaves MTP slower than ordinary
generation. This profile divides assistant forward execution into stages before
choosing the next optimization. It does not modify production sources: run.py
builds a separately named core object and executable, recording the source patch,
commands, settings and hashes. The only other patch is matched B512 ordinary
prompt ingestion, as in the preceding generation diagnostics.

Each context runs three rounds of ordinary, serial-MTP, batch-2 and batch-4
generation. Round zero is warmup; round one reverses mode order. Exact output,
token and stop parity must hold. The summarizer matches every assistant stage
record to the outer assistant call count and retains both warmed observations.
Output batching, borrowed/target wide attention, weight reuse and greedy early
rejection are enabled. CUDA event profiling is disabled.

Stage definitions:

- `setup`: RoPE tables, scaled target token embedding, recurrent hidden input,
  input projection and pre-layer setup.
- `query`: layer bookkeeping, input normalization, query projection, head
  normalization and query RoPE, summed across assistant layers.
- `attention`: borrowed attention and its synchronous query/result transfers.
- `attn_output`: attention output projection, post-normalization and residual.
- `ffn`: feed-forward normalization, gate/up projections, activation, down
  projection, post-normalization and residual, plus any layer output scaling.
- `head`: final normalization and vocabulary projection/readback.
- `recurrence`: hidden-state projection for the next assistant step.
- `softcap`: final logit softcap, when present.
- `validation`: finite-logit checks and suppressed-token masking.

These are QPC wall-time intervals, including assisted GPU execution and transfers;
they are neither isolated kernel durations nor exclusively CPU time. Timer calls
add overhead. Logging occurs after stage measurement and contributes to the
outer assistant timer; the summary preserves this difference. Target hidden
readback before the assistant call and generation sampling are outside these
stages. Do not use profile wall times as accepted throughput.

The purpose is to locate costs without guessing that softcap, transfers, or one
matrix dominates. Accepted llama.cpp ratios remain 85.6103% short and 106.8707%
at 8K; the 125% target remains unmet.

## Results

All 24 generation calls passed exact parity. Every assistant call has a complete
nine-stage record; ordinary generation has none. The two warmed observations
and stage sample standard deviations are retained in summary.json.

Mean stage totals per 32-token generation, in milliseconds:

| Context/mode | Assistant calls | Feed-forward | Vocabulary head | Borrowed attention | Outer assistant total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short serial MTP | 20 | 33.481 | 20.946 | 7.333 | 90.377 |
| Short batch 2 | 20 | 33.457 | 20.983 | 7.502 | 90.934 |
| Short batch 4 | 29 | 48.927 | 30.440 | 10.633 | 131.095 |
| 8K serial MTP | 20 | 33.409 | 20.991 | 43.181 | 126.200 |
| 8K batch 2 | 20 | 33.792 | 21.014 | 43.518 | 127.199 |
| 8K batch 4 | 29 | 48.336 | 30.531 | 63.050 | 183.647 |

The stage sums exclude about 4.8-5.0 ms per window-2 generation and 6.8-7.3 ms
per window-4 generation that the outer timer includes, principally stage logging
and call overhead. This instrumentation is consequently unsuitable for a speed
claim. Softcap totals are below 0.001 ms in every warmed mean; finite validation
is about 1.0/1.5 ms for 20/29 calls. Optimizing softcap cannot close this gap.

Feed-forward execution is the largest short-context assistant stage. At 8K,
borrowed attention becomes larger; it still includes query/result transfer and
synchronization around the existing wide-value kernel. Vocabulary projection
is the other large context-independent stage. Resident/fused assistant work
should therefore target these paths while preserving the frozen target prefix
and recurrent hidden-state semantics. Stage wall time alone does not separate
arithmetic from transfer/launch overhead, and CUDA's assisted context does not
currently expose context statistics. Do not claim that all of this time can be
removed merely by moving buffers to the GPU.
