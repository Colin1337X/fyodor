# Transfer controls with a large resident allocation

Prepared follow-up to gpu-transfer-attribution-20261114. That small-buffer run
did not reproduce System Copy-engine activity. This worker keeps 13 GiB of
initialized device buffers allocated while repeating the same seven five-second
idle/upload/download/device-copy phases with 8 MiB transfer buffers. It refuses
to allocate unless CUDA reports enough free memory to retain at least 1 GiB of
headroom. Initialization and synchronization complete before phase recording.
All allocations are released at exit; no persistent device settings change.

The runner and summarizer reuse the original Fyodor control implementation in
a separate evidence directory. Run run.py, then summarize.py from the repository
root using their full relative paths. The original small-buffer evidence is
preserved. Whole-interval coverage and unknown-process activity rules are unchanged.
This diagnostic cannot establish performance acceptance or prove System ownership.

Syntax checks pass. The GPU exclusivity preflight rejected the currently active
Gemma sanitizer before creating output. The residency control has not executed;
no claim about its behavior on the GPU is made at this checkpoint.
