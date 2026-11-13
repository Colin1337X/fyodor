# Rejected compact Q4 Gemma gate experiment

Decision: do not integrate. Reading original 144-byte Q4_K blocks instead of 160-byte expanded blocks reduces a gate/up pair from 73,728,000 to 66,355,200 bytes, but all tested compact kernels are slower than the current packed kernel. Scale decoding and generated instruction scheduling are plausible costs, not measured causal attribution. Smaller storage alone does not establish faster execution.

`compact_source.py` reproduces the original Fyodor candidate by transforming the existing `optcpp/matvec/gemma_gate.cu`; no third-party code is copied. The temporary CUDA source was removed after rejection; no dead runtime kernel or build option remains. The generator matches the tested source, including the oracle's original CRLF bytes. Both oracle JSON files retain identical source hashes and outputs. Eight row counts (1/7/8/9/63/64/65/15360) match the separate packed kernels exactly, including canaries and finite outputs. All four compact scheduling variants also match at 15360 rows before timing. There was no sanitizer or real-model validation because the speed result gave no reason to integrate.

RTX 5060 Ti SM120, NVRTC 13.1, F32 activations, 15360 rows x 3840 columns, original synthetic weights. CUDA graphs contain twenty operations; three warmups, five samples of ten replays in each forward/reverse order. Results include ten raw samples per variant. Function attributes are JIT resource queries, not measured occupancy or hardware counters.

| Variant | Mean ms | Sample SD ms | Registers/thread | Local bytes |
| --- | ---: | ---: | ---: | ---: |
| separate | 0.197466 | 0.002606 | - | - |
| u8b4 | 0.194925 | 0.003812 | 64 | 0 |
| u1b4 | 0.189626 | 0.002515 | 55 | 0 |
| compact1 | 0.241650 | 0.002300 | 64 | 0 |
| compact4 | 0.256564 | 0.002978 | 64 | 16 |
| compact8 | 0.250068 | 0.002704 | 64 | 0 |
| compact15 | 0.248051 | 0.005924 | 64 | 0 |

**Not accepted performance evidence:** repeated synthetic weight banks and retained desktop GPU activity. GPU/environment snapshots, source hash, raw samples and resource queries are in `probe-results.json`. No llama.cpp/PyTorch score changes. Accepted Gemma ratios remain 85.61% short / 106.87% at 8K; the separate 143.24% long-decode result remains diagnostic. Neither experiment changes runtime dispatch. Reproduce with `python backend/benchmarks/gemma4-gate-compact-20261113/probe.py`.
