# Isolated joint-base MTP experiment

This experiment moves an assistant round's already-emitted base token into the
same target verification batch as up to three proposed tokens. The existing
runtime separately executes the base forward, then verifies proposals. Combining
these could avoid one target weight traversal per round, but no speedup or GPU
correctness claim is established yet. Production core and defaults are unchanged.

`joint_base.py` transforms an isolated copy of the current C core. It requires
an assistant, greedy sampling, requested window four, and resident verification/
rewind support with a batch bound of at least two. Effective proposals reserve
one slot for the base (three proposals for a four-token batch). Other requests,
non-greedy sampling and unsupported providers retain existing execution.

The assistant receives the previous target hidden state and reads the frozen
KV prefix before any base advance. Verification starts at the base position,
counts all evaluated target positions, uses the base output to accept/reject the
first proposal, then shifts the remaining logits to the existing acceptance
loop's indexing. A rejection uses the existing rewind/replacement path. The early
first-proposal shortcut is disabled only for joint rounds because base logits
are not yet available. This sacrifices early rejection to reduce separate target
passes; low acceptance could regress. No numerical tolerance is relaxed.

build_generation.py retains the exact core patch, C17/C23 commands and hashes.
It also retains the earlier isolated B512 prompt-ingestion override. The fourth
generation mode is named mtp-joint3; ordinary, serial2 and batch2 stay available
for output comparisons. The generation executable has NOT run. Existing four-mode
coverage parsers deliberately do not accept its changed mode inventory; a matching
validated parser/launcher is required before any timing acceptance.

build_fixture.py builds the existing architecture suite against the experimental
core with strict C17/C23 warnings. CPU executions passed both standards, including
the host MTP generation/window and sampled comparisons. They do not select the
joint GPU path. fixture-cpu preserves logs, hashes and the exact v1 launcher used;
the current launcher additionally rejects competing GPU contexts. Its ring launch
was refused while the separate CUDA removal suite remained active, without
creating a fixture-ring directory.

Pending: GPU ring/linear/failure controls, full-model greedy parity across contexts
and stop boundaries, integrated sanitizer, correct interval coverage/parser, and
paired short/long decode timing. Do not integrate or count this as performance.

## GPU findings: superseded by v2

Ring and linear GPU architecture fixtures passed in C17/C23, selecting joint
verification. The forced CUDA failure control failed: prefill fell back to host
execution after joint eligibility had been computed. The candidate then skipped
the base forward despite losing resident verification. fixture-failure retains
the actual failure; do not run this original prototype as a valid implementation.
The isolated v2 experiment rechecks capability after prefill and restores the
original proposal window when falling back. Production code was never changed.
