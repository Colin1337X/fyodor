# Native metadata launch-bound-four experiment

`optcpp/matvec/integer_metadata_bound4.cu` differs from the validated native
metadata kernel only in its exported symbol and `__launch_bounds__(256, 4)`
instead of `(256, 3)`. The offline audit in the adjacent integer-metadata
experiment found 61 registers/thread rather than 66, with zero spills for both.
That compiler result motivates a test; it does not establish runtime resources,
occupancy or performance.

NVRTC compute_120 compilation passed without a GPU context. The host metadata
algebra check passed all 65536 byte pairs and 10000 independent metadata blocks;
this is not execution of the CUDA kernel. The production build and the running
bound-three model candidate are unchanged. Removing this optional source has
no effect on the production build.

The prepared raw harness compares baseline, metadata bound-three and metadata
bound-four against the existing Q4 projection oracle in the same 207 cases.
It preserves input/output guards and independent float64 checks for small cases.
Its prerequisite gate refused to run while the current long-generation proof
was incomplete, before importing any context-creating module.

`run_memcheck.py` repeats the raw sweep under Compute Sanitizer. `probe.py`
requires completed raw and memory evidence, then compares all three kernels on
eight shapes in forward/reverse order with identical interleave work, warmup,
ten samples per kernel/shape and byte-identical output checks. This synthetic
diagnostic does not have continuous activity coverage and cannot update accepted
decode scores. Completed GPU results are recorded below.

Raw parity and Compute Sanitizer each passed all 207 cases after long generation
finished. The sanitizer exited zero with exactly one zero-error summary. The
three-way diagnostic retained byte-identical outputs and completed all eight
shapes. Compared with metadata bound-three, bound-four mean latency increased
by 9.33%, 3.74%, 2.47%, 2.21%, 0.16%, 2.67% and 0.11% on the first seven shapes
in the recorded order. Only the 15360-column/3840-row contraction improved:
0.138007904 ms versus 0.140811872 ms, a 1.99% decrease. This is a diagnostic,
not an accepted performance result or proof of an occupancy-based cause.

Do not replace the bound-three kernel globally. A shape-specific follow-up
would require a controlled repeat and real-model validation; the current
bound-three candidate continues to full-model memory checking. Lower register
allocation alone was not sufficient to improve most shapes. Accepted scores remain 85.61025294%
short and 106.8707255% at 8K of llama.cpp, below the requested 125% on both.
