# Selective paired-row full-model experiment

This isolated C17/C23 O3 build extends the validated QKV-input-reuse model
candidate. It embeds the validated native `optcpp/matvec/q4_pair_rows.cu`
kernel and leaves production sources, archives and dispatch unchanged.

`NYA_CUDA_PAIR_ROWS=1` requests optional function lookup. A missing function,
disabled switch or ineligible dispatch retains the existing kernel. Selection
requires four input rows, Q4_K type 12, enabled integer interleaved dispatch,
native 144-byte block stride, and exactly one measured (columns, rows) pair:
(3840,15360), (8192,3840), or (15360,3840). Other shapes retain the old path.
Existing vendor-BLAS precedence and QKV input-reuse behavior are preserved.

The selected launch uses 256 threads and sixteen output rows per block.
The new trace counter records capture/direct calls, not graph replay calls.
It is separate from the existing logical operation statistics; no extra
weight packing, scratch allocation or data transfer is introduced.

The build verifies QKV build, short-model memcheck and long generation;
paired-row raw numerical and three sanitizer proofs; and repeated timing
gains for exactly the three selected shapes. Both C17 and C23 strict-warning
builds passed. The actual extracted C dispatch predicate passed 3780 cases
per standard, including missing function, disabled mode, wrong batch/type,
nonselected dimensions and incorrect strides. This control test uses no GPU.

The lifecycle runner additionally requires positive paired-row selection
markers while retaining reset/rewind/replacement, one-download, graph reuse,
QKV reuse and the unchanged 0.0005 scaled-logit bound. Its initial attempt
stopped before launching a workload because inherited tool provenance
includes an absolute Python path outside the checkout. Manifest key handling
now retains such absolute paths instead of requiring every tool to be local.

Model lifecycle/generation, full-model sanitization and end-to-end timing
remain pending. The raw kernel's 5.9–6.2% expansion latency reduction does
not imply the same model-level gain. Accepted llama.cpp scores remain
85.61025294% short / 106.8707255% at 8K, below the 125% target.

Both C17 and C23 short lifecycle checks subsequently passed both reset cycles
at prefix 257 and four verified tokens. Each recorded 128 paired-row capture/
direct calls, retained the graph/readback and reset/rewind guards, and reported
maximum scaled logit error 0.0000219546455. All 754 hashes per result were
independently verified. Long lifecycle, generation, full-model sanitizer and
paired end-to-end performance are still pending; selection is not a speed claim.

Generation and full-model memcheck runners are prepared for this isolated
binary. Both require exactly three well-formed, positive paired-row execution
markers, along with the existing twelve-case output inventory, graph reuse,
integer-path markers and QKV reuse. The sanitizer validator has eleven passing
CPU tests, including missing/extra/zero/malformed paired-row markers. These
are validator tests, not a completed model sanitizer run. The generation
runner refused the missing long-lifecycle prerequisite, and the sanitizer
runner refused missing generation evidence without launching a GPU job.
The current 8K lifecycle process remains active; it has not been restarted.

The 8K lifecycle subsequently passed both resets with maximum scaled logit
error 0.0000163989979 and 128 paired-row capture/direct calls; all 754 hashes
verified. Short generation then passed all twelve ordered cases in both
C17 and C23 builds, with exact output parity, graph reuse and paired-row
markers [128,128,128]. All 766 hashes per generation result independently
verified. Full-model memcheck, long generation and end-to-end performance
remain pending. These are correctness results, not updated speed scores.

`run_monitored.py short|long LABEL [enabled|disabled]` is prepared for the
same-binary paired-row comparison. Only `NYA_CUDA_PAIR_ROWS` changes between
variants; QKV input reuse remains enabled and requires its three positive
markers in both. Paired-row mode requires exactly three positive markers
when enabled and none when disabled, rejecting malformed markers either way.
Six CPU tests pass. The pending-memcheck refusal was exercised without
starting another GPU workload. The unchanged continuous activity collector
can still reject or abort runs on invalid counters; no evidence requirement
has been relaxed and no new timing has been measured with this wrapper.
