# Active working-set attribution control

The prior 13 GiB residency control initialized its retained allocations once,
then transferred only 8 MiB buffers during its sampled phases. Its negative
System-activity result does not test repeatedly accessing the retained working
set. This follow-up fills all 52 retained 256 MiB allocations and, in a separate
phase, copies between 26 disjoint pairs. Both phases access the full 13 GiB
address range. A fill writes 13 GiB per sweep; a copy reads 6.5 GiB and writes
the other 6.5 GiB. Neither allocates within a phase. Idle phases separate them,
followed by the original small pageable-download control and a final idle phase.

Each phase lasts at least five seconds, synchronizes completed operations and
records QPC-compatible bounds. The summarizer considers only complete sampled
intervals, retains invalid counters and continues to label PID 4 unattributed.
The worker refuses insufficient memory headroom. The runner waits for the live
batched-graph model sanitizer to complete before checking for competing GPU
contexts. It does not terminate System, change permissions or waive coverage.

Five CPU summarizer tests pass, including rejection when the active phase did
not execute or reports only a small working set. GPU execution completed with
exit zero and a healthy collector; 183 input hashes and six evidence hashes
verified. Each of seven phases had four complete sampled intervals. System
activity appeared during idle and active phases (two to four positive samples
per phase). Fill and copy each had an invalid target counter. This run does
not isolate System activity to active working-set access or establish its
ownership. Raw counters remain retained; no clean timing or attribution proof
is claimed.
This is a causal-control experiment to narrow a hypothesis, not a throughput
benchmark or proof that System activity belongs to the worker. Even if the
activity appears only during sweeps, further attribution is needed before any
benchmark acceptance rule changes.

## Source audit

Microsoft's system-paging documentation describes allocation transfers, fills,
page-table operations and translation-cache work in a separate system paging
context. Device paging queues can perform mapping work even for allocations
already resident. Therefore staying below a boundary memory-budget snapshot
does not rule out paging-engine work. GPUView distinguishes scheduler paging
and preemption packets from ordinary process packets; packet-level tracing is
the more direct attribution path, but prior WPR/DxgKrnl attempts lacked access.
These general facts do not establish what PID 4 did in the rejected Fyodor run.

- https://learn.microsoft.com/en-us/windows-hardware/drivers/display/system-paging-process
- https://learn.microsoft.com/en-us/windows-hardware/drivers/display/device-paging-queues
- https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-hardware-queue

`D3DKMTQueryStatistics` was also considered, but its public page marks it reserved
for system use and supplies no supported application attribution contract. No
private structure layout or guessed packet attribution is introduced here.

- https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/nf-d3dkmthk-d3dkmtquerystatistics

Accepted decode remains 85.61025294% short and 106.8707255% at 8K of llama.cpp.
