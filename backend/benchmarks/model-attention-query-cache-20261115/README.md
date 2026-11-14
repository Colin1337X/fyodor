# Isolated model integration of cached-query attention

Builds on the validated model-q6-pair-20261114 experiment, retaining Q4/Q6 paired
rows and QKV input reuse. The unpromoted gate/up fusion is not included.
The portable production archive and default dispatch are unchanged.

NYA_CUDA_CACHED_QUERY=1 resolves two optional kernel entries. Missing entries
fall back independently to existing linear/ring attention. Dispatch requires
existing split eligibility, non-reference Gemma execution, batch two or four,
16 query heads, and either width 256/KV heads 8/window 1024 or width 512/KV
head 1/no window. Scratch layout, combine launch and arithmetic are unchanged.
A trace counter records capture/direct calls, not logical graph replays.

The first strict build rejected a shadowed local variable; its failure record
and generated directory are retained. The corrected isolated v2 build passes
C17/C23 with warnings as errors. The actual extracted C eligibility predicate
passes 8064 cases per language standard, including absent optional kernels,
reference mode, non-Gemma models and unsupported shapes/batches.

Raw geometry validation, sanitizer evidence and fenced synthetic timing are
prerequisites verified by build.py. They are documented under
../attention-query-cache-20261115. Lifecycle and real-generation validation are
pending. No accepted inference performance or production promotion is claimed.

The first C17 prefix-257 lifecycle exits zero and both resets meet the original
0.0005 numerical bound (observed 2.19546455e-05). Graph capture/replay and existing
Q4/Q6/QKV inventories pass. However, there are no cached-query calls: the
existing batch-four split dispatch keeps full attention unsplit below 8192,
and this short workload does not exercise the new path. The wrapper correctly
reports passed=false because positive execution evidence was required. This
result is fallback evidence only, not cached-query lifecycle validation.
All 897 build identities, six control identities and 907 short-run identities
verify independently. The eligible C17 prefix-8192 lifecycle is running; its
result remains pending.

check_lookup.py compiles the actual optional symbol-loading block with a fake
CUDA lookup function. All 192 cases per C17/C23 standard pass: every combination
of available linear/ring baseline and cached entries, reference mode, unset/empty
and malformed flags, and the exact enabling value. Missing cached entries leave
the original pointers intact and do not select an unavailable function. This
is a host loading/fallback check; the live 8K lifecycle outcome is still pending.

The C17 prefix-8192 lifecycle now passes. Both resets report maximum scaled
error 1.63989979e-05 against the unchanged 0.0005 bound; capture/replay is 1/2.
The trace records eight cached-query capture/direct calls, proving the candidate
ran. Reset, rewind, invalid-input, replacement-token and download assertions
remain in the original C lifecycle. All 907 result identities verify independently.
This is numerical/lifecycle evidence, not throughput. A separate fallback-only
runner now requires absence of cached calls at prefix 257 and labels that scope
explicitly. Its C23 run is pending.

Both explicit prefix-257 fallback runs now pass in C17 and C23. Each retains
the full numerical/reset/rewind/guard/download checks, confirms zero cached-query
calls, and verifies 907 identities. These are deliberately labeled fallback-only.
run_generation.py requires both fallback proofs and the positive C17 8K lifecycle.
Short generation must report no cached-query calls; long generation requires
six positive execution records across the three batched-two and joint rounds.
The C17 short generation run has started; its result is not yet established.

Short generation now passes all twelve ordered cases in both C17 and C23.
Output parity, existing Q4/Q6/QKV execution inventories and graph reuse pass;
cached-query calls are absent as required at this prefix. All 920 recorded
identities per run verify independently. These are fallback-generation results,
not evidence of a short-context gain. C17 long generation is now running and
must prove positive cached-query execution as well as all twelve output matches.
