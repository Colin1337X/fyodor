# Q6 paired rows with four interleaved inputs

Original isolated candidate in `optcpp/matvec/q6_pair_rows_four.cu`. Each warp
computes two adjacent rows and shares a float4 input load across them. Native
Q6_K blocks, signed-zero scale-product handling, per-output accumulation order
and warp reduction order follow the existing batch-four Q6 integer kernel.
An odd final row uses the preceding valid row as an unobserved dummy input.

This differs from the rejected single-input experiment in
`gemma4-q6-rows-20261113`; that rejection remains valid and motivates caution.
The decode profile identifies batch-four Q6 down projections as material cost.

NVRTC compilation for SM120 passed, without creating a GPU context. Compilation
does not establish numerical correctness or speed. Required next checks are
compiled resource inspection, exact baseline equivalence plus independent
numerical bounds, tails/guards, sanitizers, then forward/reverse timing. No
production dispatch or accepted performance score changed. The source remains
optional under root optcpp and is not required by portable C builds.

SM120 offline assembly reports 80 registers, zero stack/spill bytes and zero
barriers for the candidate. The first numerical harness attempt reached all
comparisons but failed provenance serialization because an input array shadowed
the hash dictionary. Its source and partial output are retained as failed
evidence; it is not counted as a pass. The corrected harness rerun passed
367 cases with byte-exact baseline agreement and output/interleave guards.
This includes odd-row tails, partial block groups, several block counts and
model projection shapes. 359 single-block cases additionally passed an
independent Python float64 bound; maximum bound fraction was 0.01663237.
Large shapes currently have byte-equivalence checks only, not an independent
double oracle. All 42 compile/assembly/oracle hash checks verified. Sanitizers,
large-shape independent bounds and timing remain pending.
