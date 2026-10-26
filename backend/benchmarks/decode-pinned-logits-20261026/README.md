# Pinned logits download experiment, October 26-27, 2026

Rejected original C host-side prototype; production CUDA sources are restored.
The candidate is retained as candidate.patch against e755dbe. No device kernel,
precision, tolerance, model layout or graph topology changed.

The existing decode path synchronizes its stream then downloads all logits.
The prototype allocated vocabulary_size*sizeof(float) pinned host bytes per
session (128000 bytes for TinyLlama), queued cuMemcpyDtoHAsync on the execution
stream, synchronized once, then memcpy'd to the caller buffer. Finiteness and
suppressed-token handling stayed intact. Allocation failure used the original
path with a diagnostic. The experiment's NYA_CUDA_PINNED_LOGITS setting is NOT
a supported production option after restoration.

The API design follows NVIDIA's memory-management documentation:
https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html
No upstream implementation was copied. This measures pinned staging plus stream
ordering together; it does not isolate their individual cost or rule out every
possible host-buffer ownership design.

C17 and C23 each passed all six selected lifecycle tests, including reset,
prefill/decode interleaving, short/long topology, prefix replay, disabled graphs
and unavailable BLAS fallback. Logs and exits are retained. A first C17 build
caught a declaration after a label, fixed before testing. No fresh real-model
CPU-logit oracle was run: the full-model trial regressed and was rejected.
The oracle.py helper is prepared but unexecuted and is not validation evidence.

The October 27 first attempt was refused BEFORE any timing because Epic Games
Launcher PID6840 and EOSOverlayRenderer PID21848 were GPU competitors. Their
identities were checked before stopping them under the user's prior authority.
paired-20261027 retains that gate failure. All eight subsequent activities in
clean-20261027 passed continuous monitoring and the unchanged idle gate.

TinyLlama Q4_K_M, F32 KV, full GPU, six CPU threads, batch512, 128 decoded tokens,
prefix0/1024, selective FDPP in both variants; full host logits every token.
Each process uses four warmups and eight repetitions, forward/reverse order,
16 pooled observations per cell. No builds/tests overlapped timing. Driver617.42,
RTX5060Ti; exact binary/model hashes and commands are in metadata.json, source
and C17/C23 build configuration in candidate-manifest.json. Every summary and
raw process log is retained. No allocation-fallback warning occurred.

| Prefix | Existing tok/s | Pinned tok/s | Change |
| --- | ---: | ---: | ---: |
| 0 | 365.302 +/- 1.264 | 362.300 +/- 1.287 | -0.82% |
| 1024 | 308.244 +/- 0.870 | 306.698 +/- 0.969 | -0.50% |

Both orders favor existing downloads. The extra staging copy and queued transfer
did not provide a useful gain. Restore the original source and rebuild both C
standards; retain the patch only as diagnostic evidence. No new optional C++ or
physical-removal dependency was introduced.

No new llama.cpp run occurred, so do not divide these timings by an older
reference to claim a new score. The latest matched score remains 94.70% short /
114.23% long versus llama.cpp, below 125% on both. Summarize with summarize.py.
To reproduce the prototype, apply candidate.patch to its recorded base, build,
and run paired.py with a fresh output directory; default production ignores
the experimental setting after restoration.
