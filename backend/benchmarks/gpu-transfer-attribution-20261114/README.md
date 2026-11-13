# GPU transfer attribution controls

The two short Q4 generation diagnostics passed idle gates and output parity,
but failed continuous coverage. PID 4 activity was predominantly on Copy engine
13, with occasional small CUDA and other Copy-engine values. This does not
establish which workload requested the activity. PID 4 remains unattributed.

WPR's GPU capture could not start: error 0xc5585011, failure to enable the system
profiling policy. A subsequent status check confirmed WPR was not recording.
A separate uniquely named DxgKrnl trace returned Access denied. No trace was
created, no existing recording was stopped, and no policy was changed.

The prepared worker compares seven five-second phases: idle, pageable upload,
pageable download, pinned upload, pinned download, device copy, and idle.
It uses 8 MiB buffers and explicit synchronization. This is an attribution
diagnostic, not a bandwidth benchmark. The runner requires no competing GPU
contexts, retains continuous PDH evidence, and checks collector liveness.
Its preflight rejected the live long-context Gemma run before creating output.
The subsequent controls run completed all seven phases with a healthy collector.
Each phase contained four complete sample intervals and no invalid counters.
No positive System or other-process engine activity appeared within those
intervals. Uploads used the worker's Copy engine 3, downloads Copy engine 4,
and device copies CUDA engine 10. These small-buffer controls do not reproduce
the Gemma System Copy-engine activity or establish its cause.

The summarizer includes only complete sample intervals within a phase. It
reports invalid counters, missing coverage, and System activity separately;
it never treats correlation as proven ownership. Four offline tests passed
for interval boundaries, invalid values, collector failure and bad clocks.
No benchmark acceptance rule has changed.

Microsoft references:

- https://learn.microsoft.com/en-us/windows-hardware/drivers/display/device-paging-queues
- https://learn.microsoft.com/en-us/windows-hardware/test/wpt/built-in-recording-profiles
- https://learn.microsoft.com/en-us/windows/win32/perfctrs/about-performance-counters

Windows documents asynchronous device paging queues and GPU recording profiles.
Performance counters are not intended for high-frequency profiling; the existing
one-second collection interval is retained here.
