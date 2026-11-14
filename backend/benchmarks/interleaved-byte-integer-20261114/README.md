# Packed Q4 integer batch-four candidate

Original Fyodor CUDA experiment in `optcpp/matvec/interleaved_byte_integer.cu`.
It applies the current eight-row/bound-three integer batch-four arithmetic to the
existing lossless 160-byte Q4_K metadata layout from `CUDA/q4_byte_pack.h`.
Scale and minimum codes are loaded as bytes instead of being unpacked from
three words for each group. Input layout, four accumulators, weight expression
and warp reduction order follow `integer_launch.cu`.

The profile identifies Q4 batch-four FFN expansion as the largest matrix group.
`plan_byte_weights` already retains selected packed gate/up pairs when configured
and memory permits. A future dispatch must use only a complete packed matrix
returned by `plan_byte_tensor`; otherwise it must retain the native integer path.
This experiment adds no production allocation or dispatch and must not assume
all layers have a packed copy. It must not repack weights during decode.

Potential tradeoff: 160-byte blocks consume 11.11% more weight bytes than native
144-byte blocks, so removing metadata instructions may regress bandwidth-bound
shapes. Neither source inspection nor successful compilation establishes a win.

`compile_only.py` compiles the candidate with NVRTC for compute_120 using the
current FP contraction options and verifies its entry symbol. `compile.json`
records options and source/PTX hashes. Compilation creates no CUDA context and
executes no GPU work; the ongoing FFN-reuse sanitizer remains isolated.

Pending: native-versus-packed raw byte-exact parity over partial rows, multiple
column counts and finite coefficient extremes; independent numerical oracle;
output guards and Compute Sanitizer; paired kernel timing; real-model selection,
parity, lifecycle and monitored decode timing. No production integration or
accepted speed claim. Removing this optional source leaves the production build
unchanged. Accepted llama.cpp scores remain 85.61025294% short and 106.8707255%
at 8K; the 125% target is not achieved.

## Prepared raw parity harness

`build_pack.py` compiles the existing production C packer through a small test
ABI in C17/C23. Each build passes 256 independently decoded metadata blocks,
checks unchanged half-coefficient/payload bytes and zero padding, and verifies
short-capacity rejection without altering guarded output. Commands and hashes
are recorded in `pack-build.json`; no CUDA context is used.

`check_kernel.py` prepares 207 GPU cases using that C packer, native Q4 outputs,
byte-exact comparison, guarded interleaved inputs/outputs, partial rows, eleven
matrix shapes, finite half-coefficient edge pairs and an independent float64
bound on small cases. The isolated `common.py` includes the candidate and native
kernels. These scripts have parsed successfully but have not executed GPU work.
They refuse to create a GPU context until the active FFN-reuse sanitizer has a
completed passing result. The `--memcheck` label alone is not sanitizer evidence;
retain the actual Compute Sanitizer exit and summary when running that mode.
