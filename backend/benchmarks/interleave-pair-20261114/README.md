# Adjacent gate/up input conversion reuse (experimental)

The joint-base profile identifies matrix work as about 85% of recorded GPU kernel
time. Its largest group is batch-four Q4 feed-forward expansion. The original
`CUDA/resident.inc` converts `p->norm` separately for gate and up, although both
read the same input and use the same scratch layout.

This isolated host-C candidate skips the second conversion only for adjacent
projections that both select an interleaved kernel, at batch two or four, without
cuBLAS dispatch. Aligned prefix batches are excluded. The reuse flag is cleared
immediately after the up call, before error handling. No scratch validity is
inferred across layers, requests or graph builds. Projection arithmetic is intact.

Production source and dispatch remain unchanged. No C++ dependency is introduced.
`build.py` transforms an isolated host translation unit; `build_executables.py`
links C17/C23 lifecycle and joint-base generation executables. Both verify the
original validated build inputs. Compile/link manifests alone are not admission.

## Completed checks

- `lifecycle-c17-257` and `lifecycle-c23-257`: both reset cycles pass four-token
  verification with ring KV. Maximum scaled error is 2.19546455e-05 against the
  original 0.0005 bound. Original reset, rewind, replacement, stale-suffix and
  output-guard assertions remain in the lifecycle harness.
- `lifecycle-c17-8192`: both reset cycles pass at prefix 8192, with maximum scaled
  error 1.63989979e-05, 568 integer calls and exit zero. All 33 recorded hashes
  independently verified.
- `inventory-257`: baseline performs 658 conversions and 658 projections;
  candidate performs 562 conversions, reuses 96 inputs and retains 658
  projections. Both lifecycle runs pass; all 47 hashes independently verified.
  This removes 14.59% of conversion dispatches in this test, not 14.59% of decode
  time. Counts include host dispatch and graph construction, not GPU replays.
- `generation-short-c17`: all twelve ordinary/serial2/batch2/joint3 cases pass
  output parity in forward/reverse/forward order, with 25 prompt tokens and 32
  generated outputs. All three joint rounds execute the integer kernel; all 41
  input/output hashes independently verified. Unmonitored timing is not accepted.

## Reproduction and remaining gates

Run from the engineering worktree root with `PYTHONDONTWRITEBYTECODE=1`:

```
python backend/benchmarks/interleave-pair-20261114/run_lifecycle.py c17 8192
python backend/benchmarks/interleave-pair-20261114/run_inventory.py
python backend/benchmarks/interleave-pair-20261114/run_generation.py long c17
python backend/benchmarks/interleave-pair-20261114/run_model_memcheck.py short
```

Runners refuse to overwrite existing evidence. Long-context generation has passed
all twelve cases (details below). The full-model short-context sanitizer is active;
its result is not yet available. Its seven evidence-validator regression tests pass (using the
existing integer-launch sanitizer test cases against this runner). It requires
successful build, context-generation, short/long lifecycle and dispatch-inventory
proofs, validates transitive hashes before and after execution, and requires
exactly one zero-error sanitizer summary plus all twelve matching generation
cases. Runner tests do not establish GPU memory safety.

Remaining gates include full-model sanitizer validation and controlled paired
throughput comparisons. Review the net decode benefit
before production integration. Accepted llama.cpp scores remain 85.61025294%
short and 106.8707255% at 8K. No result here establishes the requested 125% target.

## Completed long-context generation

`generation-long-c17/result.json` passes all twelve cases at prompt length 8205
and 32 generated outputs, in forward/reverse/forward order. The process exited
zero; all three joint rounds recorded 2556 integer calls. All 41 input/output
hashes were independently verified after completion. These unmonitored timings
remain excluded from performance acceptance. Full-model short-context Compute
Sanitizer validation is now the active GPU check.


## Monitored throughput runner

`run_monitored.py short|long UNIQUE_LABEL` is prepared for this exact FFN-reuse
binary. It requires successful compile/link, dispatch inventory, short/long
generation and full-model sanitizer proofs before launching GPU work. It retains
the existing model identity, idle/environment, continuous GPU monitoring,
transitive input hashes, output parity and dispatch-count checks. Three existing
input-integrity regression tests pass against the new runner. It has not run;
the sanitizer prerequisite is still pending. Monitoring alone never marks a run
as an accepted benchmark, and contamination remains grounds for rejection.
