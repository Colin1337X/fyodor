# Fixed-width gate/up diagnostic, October 27, 2026

Original Fyodor device specialization under root optcpp/matvec. Columns and
Q4 row stride become compile-time constants; accumulation, precision, reduction,
launch bound, unroll factor and weight format are unchanged. The ABI retains
ignored column/stride arguments for this diagnostic. Callers MUST select the
matching symbol; this is not a generally valid replacement for arbitrary shapes.
The production runtime and CMake dispatch remain unchanged.

The October 26 profile places gate/up at about 40% of matrix time. Previous
load/layout/compiler-policy experiments did not yield useful gains. This trial
asks whether knowing the entire row width lets NVRTC remove dynamic loop/address
work, without extra weights or new arithmetic. No upstream code was copied.

| Workload | Rows x columns | Throughput change |
| --- | --- | ---: |
| hot | 9 x 256 | +6.32% |
| hot | 5632 x 2048 | +0.44% |
| hot | 11008 x 4096 | -3.97% |
| extended | 65 x 768 | +16.14% |
| extended | 257 x 2304 | -39.19% |
| extended | 28672 x 8192 | -0.15% |
| rotating | 5632 x 2048 | +2.93% |
| rotating | 11008 x 4096 | -3.59% |

The 2048-column specialization wins 2.93% with rotating weights, but only 0.44%
with hot weights. It merits an isolated runtime trial, not adoption or a new
whole-model score. The 4096-column specialization loses in both cases; 2304
columns uses 312 bytes of local memory and loses badly. All others use no local
memory; 256 columns uses 48 registers, other specialized widths 64. Reject a
blanket fixed-width dispatch. The useful next step is a guarded 2048-column
full-model trial with exact baseline comparisons and optional-source removal.

All six hot cases plus two eight-bank rotating cases match baseline output and
guards byte-for-byte, including odd loop counts and partial blocks. This covers
seeded random payload/metadata, bounded input, fixed finite half scales. It does
not substitute for real-model, exceptional-value or integrated lifecycle tests.

All three monitored activities passed idle/competitor gates; no GPU competitor
needed stopping. NVRTC compute_120, C++11, fmad enabled, RTX5060Ti driver617.42.
Each graph contains 50 launches, three warmup replays, then five samples of ten
replays in forward and reverse order. Every sample is retained; summary.json
reports pooled means and sample standard deviations. Rotating weights total
103809024/405798912 bytes. Packing, graph capture and allocations are untimed.

Run run.py with fresh output directories. Source hashes and compiler options
are in each results.json, resource counts alongside them. Process snapshot
trailing whitespace alone was normalized after collection. There is no new
build dependency: removing optcpp still removes only unselected diagnostics.

Latest matched whole-model scores remain 94.70% short / 114.23% long of llama.cpp;
125% on both is unmet. No new llama.cpp comparison or runtime gain is claimed.
