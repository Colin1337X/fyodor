# Q4 offline instruction audit

Both kernels were compiled in the same NVRTC module with the existing
compute_120/C++11/fmad/FAST_MATH options and assembled by ptxas 13.1.115 for
SM120 at O3. cuobjdump/nvdisasm 13.1.115 produced `kernels.sass`. NVIDIA
redistribution archive SHA256 and sizes were verified against the existing
CUDA 13.1.1 manifest before extraction into local SDK tools. The first
disassembly attempt failed because cuobjdump also needed nvdisasm; that log
is retained, then the matching nvdisasm package was installed and used.

The current metadata kernel has eight 128-bit input-load sites, four quantized
byte-load sites and 40 FFMA sites. The partial-unroll kernel has two, one and
ten respectively inside its shorter inner loop. Static counts are not dynamic
work: the inner body repeats four times. Partial unrolling also introduces
six equality-predicate sites and predicate-controlled MOV selections in the
inner loop (see offsets 0x530 onward), versus no equality-predicate sites in
the current kernel. This is concrete additional selection/control machinery,
consistent with the observed regression, but does not isolate its timing cost.

This audit also identifies 64-bit induction/address instructions in the fully
unrolled version. The separate bounded-index experiment retains full unrolling
and narrows only validated column induction. Its runtime benefit is unknown.

These are offline SASS instruction sites, not runtime driver-JIT output,
measured dynamic counts, stalls, occupancy or throughput. No GPU context was
created. `ptx-inspection.json` records sources, tool identities and hashes.
Production dispatch and accepted scorecards remain unchanged.

Tool reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
