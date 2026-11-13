# Rejected multi-row Q6 decode experiment

Original Fyodor CUDA experiment in `optcpp/matvec/q6_row_pair.cu`; no upstream source copied. Each warp computes 1/2/4/8 adjacent output rows, reusing one activation load across them while preserving signed Q6 scale-product arithmetic and each output's accumulation/reduction order. No runtime or build dispatch was added.

Decision: **do not integrate**. Every multi-row variant regresses both measured Gemma shapes. The one-row control is close to the existing kernel. Reusing activations alone is insufficient to improve these projections. The JIT reports 39/40/48/72 registers for 1/2/4/8 rows respectively and zero local bytes for all. These attributes do not establish the cause of the slowdown or measured occupancy.

`check_kernel.py` validates exact outputs, finite results and guards at columns/rows 256/1, 256/7, 256/9, 3840/65, 15360/3840, and 3840/262144. Partial row groups are included. This checks agreement with the existing integer-scale Q6 kernel, not an independent error bound. No real-model or sanitizer validation was performed because isolated timing rejected the candidate.

RTX 5060 Ti SM120. The probe captures ten operations per graph, three warmups, five samples of ten replays, in forward/reverse variant order (ten observations per variant). It repeatedly accesses the same weight bank. Desktop activity is retained and recorded; all timings are diagnostic and acceptance is false. No llama.cpp/PyTorch score changes.

| Columns x rows | Variant | Mean ms | Sample SD ms |
| --- | --- | ---: | ---: |
| 15360 x 3840 | baseline | 0.148476 | 0.004282 |
| 15360 x 3840 | rows1 | 0.149276 | 0.005453 |
| 15360 x 3840 | rows2 | 0.340493 | 0.006142 |
| 15360 x 3840 | rows4 | 0.294041 | 0.007721 |
| 15360 x 3840 | rows8 | 0.378784 | 0.003349 |
| 3840 x 262144 | baseline | 2.231660 | 0.029462 |
| 3840 x 262144 | rows1 | 2.234276 | 0.036693 |
| 3840 x 262144 | rows2 | 3.471187 | 0.011099 |
| 3840 x 262144 | rows4 | 3.981947 | 0.064277 |
| 3840 x 262144 | rows8 | 4.097200 | 0.047212 |

Reproduce with `check_kernel.py`, `probe.py`, and `resources.py` in this directory from the repository root. Raw samples, source hash, environment snapshots and JIT attributes are retained. The source is diagnostic-only and can be physically removed with optcpp without affecting the runtime build.
