# Diagnostic runs

`baseline/` captures the unmodified `aa298f6` executable before optimization.
The event profile includes warmup and untimed prefix preparation, disables
graphs, and can include host submission gaps. It is a bottleneck diagnostic,
not an isolated-kernel benchmark. Other desktop applications remained open.

`split-probe/` is the first on/off experiment using one modified executable.
All samples are retained. Its final run, `1024-3-0`, overlapped a build and the
start of the expanded attention test. This entire directory is exploratory
and excluded from performance acceptance. Earlier samples motivated further
correctness checks and a separate controlled before/after comparison.

Idle GPU Engine counters identified `dwm.exe` (PID 1920, roughly 10.5% engine
utilization) and `ChatGPT.exe` (PID 18352, roughly 5.8%) as active. These are
per-engine counters and must not be added to or equated with `nvidia-smi`'s
whole-device utilization. Other apps remained open; none was terminated.

Correctness matrix and GPU instrumentation may overlap one another; none is
a performance measurement. Acceptance benchmarks run only after all of them
have exited, with no builds, tests, package work, or other agent benchmarks.

`accepted-desktop/` was the first acceptance attempt, despite its directory
name. It is **excluded**: a new `D:\models\koboldcpp.exe` process appeared
during the 512-token runs, GPU memory climbed above 12 GiB, and the next idle
gate stopped the driver. All records remain intact. The user explicitly
authorized stopping competing GPU activity; process 11628 was identified and
stopped, returning memory use to about 2.1 GiB. The full rerun is stored
separately in `accepted-clean3/`. Its sampler additionally rejects pure CUDA
compute processes and GPU identities absent from the initial desktop snapshot,
except the measured child. Idle checks use the same process guard.

`accepted-clean/` contains no benchmark samples. The first process guard used
`--query-compute-apps`, which reports ordinary C+G desktop clients on WDDM and
therefore rejected the desktop itself. The corrected guard uses the XML process
type and initial graphics-process identities. Existing desktop clients remain
the disclosed source of interactive-desktop variability.

`accepted-clean2/` is also excluded in full: the corrected guard caught a newly
appearing Thorium GPU process, PID 11956, after `512-after-a` and stopped.
Thorium was then closed under the user's competing-GPU authorization. Its
process identities are retained in `stopped-thorium.json`; `accepted-clean3/`
starts fresh after that closure.
