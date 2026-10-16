# Portable attention foundation and FA4 audit, October 16, 2026

Original F32 tiled online-softmax kernels under root optcpp provide CUDA and
Vulkan compatibility paths. They are not FA4. The pinned FA4 SM120 source uses
an SM80-style pipeline, so blanket SM120 incompatibility is incorrect. Its
FP16/BF16 contract and CuTeDSL runtime/export dependencies require a separate
integration from the accepted F32 KV benchmark. Audit hashes and revisions are
in optcpp/attention/upstream-audit.json; no unlicensed artifact was copied.

CUDA: six double-oracle cases, all four sanitizers and real TinyLlama logits
pass. Compilation covers compute75/80/86/89/90/100/120; actual execution is
SM120 only. Vulkan: the existing independent 2e-6 attention oracle and full
TinyLlama inference pass. The Vulkan implementation is synchronous and uploads
host KV on every attention call; it is not a resident performance pipeline.

Initial build failed because the optional function lookup could jump to cleanup
before start_weights initialization; moving the lookup below initialization
fixed it. The first removal invocation predated target reconfiguration; retry
passed. All failed logs are retained. Combined removal validates 157 tests,
with five expected provider skips; the final post-integration removal is in
../flash-decoding-pp-20261016. SPIR-V is reproducible with the optional generator
under optcpp/attention; no shader compiler is required for normal C builds.

Flash-Decoding++ implementation and subsequent validations are in the adjacent
flash-decoding-pp-20261016 directory. AMD execution and FA4 AOT remain pending.
Accepted performance scores are unchanged by these correctness results.
