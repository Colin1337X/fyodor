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
