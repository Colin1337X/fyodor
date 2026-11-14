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
decode scores. No GPU execution or timing has occurred for this variant yet.

The next action is raw parity, memory checks and the three-way diagnostic once
the existing long-generation workload has finished. Only a useful result can
justify another isolated real-model build. Accepted scores remain 85.61025294%
short and 106.8707255% at 8K of llama.cpp, below the requested 125% on both.
