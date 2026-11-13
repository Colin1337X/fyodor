# Bounded batch verification interface, 2026-11-13

This checkpoint adds `nya_llm_session_verify` and optional `nya_compute_plan_verify` behind the native C interface. Output is `[count, vocabulary]`: row i is the distribution **after consuming token i**. The distribution used to verify the first proposal is the session's existing logits before this call. Callers own the output buffer and pass its capacity in floats. Input/output/session buffers must not overlap.

CUDA batches the transformer within its configured plan batch size, then runs each position's output norm/head/softcap and readback separately using existing scratch. It does not allocate a vocabulary-by-batch device buffer. Count-one calls preserve the ordinary graph path; normal decode and final-logit-only prefill retain their behavior. Other providers use serial session execution when the optional callback is absent. The new field is appended to the internal backend interface; C interfaces expose no driver or C++ types.

All token IDs, capacity and product/byte overflow are checked before dispatch. Invalid inputs leave the session position/logits intact. A runtime failure can leave partial output; it must be discarded. The session becomes invalid, drops its failed device plan, and can reset onto the CPU reference path. No partial GPU cache is reused as CPU cache.

## Validation

- GCC C17: fourteen registered tests pass, including CPU/Vulkan fallback, CUDA batches, tiny ring wrapping, two-layer shared KV, graphs disabled, injected failure/reset, and five existing resident/graph/replay regressions. The initial shared-KV test accidentally used a one-layer trained fixture; `initial-c17.log` records that harness failure. The final test generates its own deterministic two-layer fixture.
- GCC C23: nine verification/fixture tests pass.
- Clang ASan/UBSan: CPU fallback and empty-prefix verification tests pass with fixture setup.
- CUDA compute-sanitizer memcheck: shared KV plus ring wrapping pass with zero errors (`memcheck.json` retains the command/environment).
- Physical removal: source copy without optcpp/CUTLASS/CuTe completes all 82 C-only tests, with four expected ROCm/MLX hardware skips, and the training/export/inference lifecycle. The copy tested the new API before later test-only shared-fixture improvements; runtime source is identical.
- Real Gemma4 12B Q4_K_M: 17 verification tokens in chunks of 8 plus continued single-token decode, two resets, prefix sizes 7, 255, and 1020. Every output position is checked, not just the final row. Maximum scaled logit differences are 0.000383745074, 0.0000663417304, and 0.000067389802 against serial CUDA. The fixed bound is 0.0005 times `(1 + abs(reference))`; it was not relaxed. Suppressed negative infinities must match and all other logits must be finite. Guard canaries and last-logit consistency pass.

The model runner commands, environment and binary hash are in `model-results.json`. After those runs, the permanent harness gained shared-fixture and transfer-count assertions; the model checks above used the recorded earlier binary. Reproduce current checks with the `nya-test-verify` CMake target and `ctest -R inference-verify`. The real-model command is `nya-test-verify MODEL.gguf PREFIX` with the recorded environment; only the fixture-size tests should set `NYA_TEST_VERIFY_SHARED`, `NYA_TEST_VERIFY_RING`, or failure injection.

## Scope and remaining work

No inference timing or accepted score changes are claimed. Accepted Gemma ratios remain 85.61% short / 106.87% at 8K; the separate 143.24% 8K result remains diagnostic. This API is not yet used by speculative acceptance and does not make the MTP assistant resident. Batched output-head execution, rejection rollback integration, shared device hidden/KV state, probabilistic acceptance validation, and end-to-end timing remain. See `../../docs/gemma4-mtp-execution-audit.md`.
