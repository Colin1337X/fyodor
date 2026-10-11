# Resident operation profiling

`NYA_TRAIN_PROFILE=1` now enables diagnostic operation profiling for resident
training graphs as well as the existing CPU graph profiler. Default execution
remains asynchronous. No device interface, kernel, precision or optimizer
semantics changed. This is original Fyodor C code.

Resident records use the separate `train_profile_resident` prefix and explicitly
label `timing=synchronized_host_diagnostic`. Every executed forward operation
and every differentiable backward operation is counted by operation kind.
Timing drains previous work before sampling and includes host dispatch, finite
checks, execution and a completion fence. Input/leaf entries therefore measure
their finite checks, not matrix computation. Pending work is not attributed to
the next operation. Asynchronous failures use the existing graph error path.

These samples perturb scheduling and include synchronization overhead. They
must not be treated as CUDA kernel times or accepted competitive throughput.
Use them to locate bottlenecks, then verify candidate improvements with normal
profiling-disabled repeated end-to-end runs.

## Evidence and findings

The reproducible driver is `training-torch-20261011/resident_profile.py`.
It runs a width256, FF704, two-layer, context128 random decoder for three
updates with profiling off and on. Checkpoints and exported models are exactly
identical by SHA-256. It then profiles three TinyLlama Q4_K_M rank2 LoRA updates
at context32 with six CPU workers. Raw logs, CSVs, checkpoints and exports are
retained in `D:/fyodor-validation/resident-operation-profile-20261011`.
The committed `resident-operation-profile.json` records commands, source/binary
hashes, output hashes and all operation summaries.

TinyLlama diagnostic per-step dense forward times were 281.5944, 280.7345 and
281.6716 ms, versus frozen mapped-forward times 54.5934, 54.8222 and 54.8303 ms.
Dense backward was about 41 ms and mapped backward about 64 ms. There are 310
dense forward calls and 155 mapped forward calls per update. Thus the LoRA
dense forward path is the next concrete optimization target. Source inspection
shows the general 32x32 output tile is used even for rank2 projections; this is
a hypothesis for the cost, not yet a verified optimization. Preserve ordered
arithmetic and test both narrow and wide shapes when evaluating alternatives.

## Validation

The CPU training subset passed all 12 tests. The initial CUDA training subset
passed all 81 tests with profiling disabled. Enabling profiling exposed five
test failures because the old graph assertions prohibited *all* synchronization.
Those failed logs are preserved. The assertions now explicitly require no
additional synchronization normally and additional synchronization in diagnostic
mode, while keeping upload/download/allocation invariants unchanged.
A dedicated `training-graph-cuda-profile` CTest runs the full tiny-model
100-update parity, checkpoint/resume and export lifecycle with profiling enabled.
No numerical tolerance was changed.

The final CUDA training subset passes all 82 tests including that new test.
All 20 selected graph/validation checks also pass with profiling enabled,
including Gemma variants and injected failures.
Evidence logs are committed alongside the operation summary. The larger
PyTorch F32 trajectory mismatch remains unresolved; this profiler does not
change or supersede its rejected correctness result.
