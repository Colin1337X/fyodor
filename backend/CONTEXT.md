# Context permissions and assembly

For the later executed-request persistence and native CLI generation work, see
[request receipts](REQUEST_RECEIPTS.md). The sections below retain the earlier
milestone evidence and its limitations at that time.

The native application service now implements persisted exact resource grants,
authorized reads and retrieval, and deterministic bounded context assembly.
`include/fyodor_context.h` defines the C interface. This is a context foundation;
it does not yet execute generation or persist request-level provenance.

## Identity, authorization and migration

Principals are canonical non-nil UUIDs. An authenticated application/transport
must supply the principal; package data and model text cannot establish one.
Each grant names exactly one principal, namespace and resource URI. Access is
denied by default. Parent resources, local ownership and remote-node identity
do not imply context grants. Grant replacement and revocation are transactional.

Schema 2 adds the grant table. Opening an exact schema-1 database validates it
before migrating in one transaction. Resources, revisions and tombstones remain
unchanged; the grant table starts empty. Unknown or modified schemas still fail.
Older schema-1 binaries cannot open the upgraded database. Resource-package
schema 1 remains unchanged and does not carry grants.

READ=1 and SEARCH=4 are enforced by the new read/search/assembly operations.
Search requires both bits. WRITE=2, APPEND=8 and DELETE=16 now protect the
corresponding existing-resource mutation APIs. EXECUTE=32 remains reserved;
no general execution API is enabled by that bit. Trusted local
administrative resource APIs retain direct access; do not expose those APIs or
permission setters as authenticated remote operations without an admin policy.
SQLite files still require filesystem protection and are not encrypted.

## Permission-checked mutations — 2026-09-30

`fyodor_context_write`, `fyodor_context_append` and `fyodor_context_delete`
require the exact corresponding grant and a nonzero expected revision. They
acquire the SQLite writer transaction before reading the grant, and retain it
through the revision check, data change and history insertion. A concurrent
revocation cannot interleave between the permission check and commit. Errors
roll back resource/history changes and preserve caller output values.

WRITE replaces an existing resource's title/content/metadata/provenance.
APPEND adds a UTF-8 suffix while preserving all other fields. DELETE preserves
the normal tombstone/history and live-parent checks. None implies READ, SEARCH
or another mutation permission, creates resources, changes namespace/identity,
or edits grants. Append-only callers receive a revision without old content.
Authorization precedes revision checking. Higher-level resource services must
still enforce their own structured content schemas.

The CLI adds `context append PRINCIPAL URI EXPECTED_REVISION TEXT` and
`context delete PRINCIPAL URI EXPECTED_REVISION`, following the same store and
namespace options. These call scoped APIs; existing `resource` commands remain
trusted local administrator operations. Remote authentication is unfinished.

The native test exercises all 64 permission masks, principal/namespace denial,
independent READ permission, append field preservation, stale revisions,
revocation and original history. Injected history failures roll back write,
append and delete without a new revision. The CLI process test proves append
without READ, stale-append rejection, denied deletion, then explicitly granted
deletion. The rebuilt C17 CPU suite passes **58 tests with 4 hardware skips**
(`build-cpu/scoped-mutations-tests.log`). The application selection passes **8
CTest entries** under ASan/UBSan and C23, including the trained-model fixture.
These are functional results, not performance measurements. No schema or
dependency change was required.

## Retrieval and bounded assembly

Search is a literal substring query over live title/content, with ASCII case
folding and exact non-ASCII byte matching. Title matches precede content-only
matches, with canonical URI byte order breaking ties. Returned summaries are
limited to 100. Schema 4 now uses a [persistent trigram index](CONTEXT_INDEX.md)
for queries of at least three Unicode scalars and scans permitted rows for
shorter queries. Ranking and literal semantics are unchanged. No embedding
retrieval or large-corpus latency claim is made.

Explicit assembly accepts up to 64 distinct resources and a budget up to 1 MiB.
The budget counts UTF-8 content bytes and inserted newlines, **not model tokens**.
The caller chooses source order. Truncation preserves complete UTF-8 scalars.
Attribution records URI, exact revision, output offset, included length and
original length, including omitted sources. It does not yet insert source
labels into prompt text or persist this inspection result.

Every explicit source must be authorized, including sources beyond an exhausted
budget. Any denial fails the entire operation without returning partial text.
Permission checks and content reads share one read transaction. Revocation
affects subsequent snapshots; it cannot retract content already disclosed.
Denied and absent resources have the same principal-read result. Authorized
reads target live heads; deletion prevents subsequent context access even when
a historical grant remains stored.

## Native CLI

After the existing `--store PATH --namespace NAME` options:

```text
context permissions PRINCIPAL URI [MASK]
context inspect PRINCIPAL URI
context search PRINCIPAL QUERY
context explain PRINCIPAL BYTE_BUDGET URI...
```

`permissions` without a mask inspects the stored effective mask; zero revokes.
Mask 5 grants READ plus SEARCH. This command is a **local administrator** command,
not proof of authentication. Inspection emits title/content/revision JSON;
search emits JSONL; explain emits the bounded text and attribution JSON. Access
denial exits 6 with stderr diagnostics and no stdout. Existing exit codes remain.
`explain` currently previews assembly; it does not claim these resources entered
an executed model request.

## Validation — Windows, 2026-09-29

The complete C17 CPU suite passed **57 tests**, with **4 unavailable ROCm/MLX
hardware skips** (`build-cpu/context-tests.log`). The six application tests passed
under ASan/UBSan. After adding CLI commands, the updated real-process CLI test
also passed under ASan/UBSan. No new external dependency was introduced.
All six final application tests also passed under GCC C23 in the CUDA-enabled
configuration; these tests perform no GPU work.

Coverage includes persisted grants/revocation, principal and namespace isolation,
READ-only/SEARCH-only denial, unauthorized search exclusion, deterministic title
ranking, budgets across multibyte boundaries, exact source offsets/revisions,
denial after budget exhaustion, duplicate/oversized input rejection, deleted
resource denial, v1 migration preserving original content/history without grants,
and real native CLI grant → inspect/search → bounded preview → revoke → denial.

Remaining at that checkpoint: model-token budgeting, generation integration, persisted request
provenance, indexed retrieval, context layers, principal-scoped mutations,
Desktop/TUI context interfaces and authenticated remote principal mapping.
No complete Unified Context acceptance or concurrent revocation stress result
is claimed yet.

## Native generation bridge

`fyodor-inference` now links the application service to the existing engine;
the resource-only CLI still links only the application library. The public
`fyodor_generate_with_context` API authorizes and assembles sources before
calling native inference. `nya_generation_prompt_info` exposes the exact native
tokenizer's count and the loaded model's context capacity without running a
forward pass. No tokenizer or inference implementation is duplicated.

The final engine input is context text, two newline bytes when context is
nonempty, then the unchanged user prompt. All requested output tokens are
reserved. Oversized context is repeatedly halved at UTF-8 boundaries and
retokenized until it fits. This is a deterministic conservative policy, not
optimal token packing. If the user prompt and requested output alone cannot
fit, the request fails before inference. Caller-specified byte and output limits
remain enforced. Draft and soft-token requests are not supported by this bridge.

The result owns the exact prompt, included-source byte ranges and revisions,
native output, token counts and model context capacity. Shortening context
updates attribution, including omitted sources. Authorization applies at the
assembly snapshot; subsequent revocation cannot retract that already-disclosed
snapshot. The caller must serialize model use and retain its lifetime.

Validation: the complete rebuilt C17 CPU suite passed **58 tests**, with the
same **4 unavailable hardware skips** (`build-cpu/context-generation-tests.log`).
The new integration test and its trained-model fixture passed under ASan/UBSan
and GCC C23 (**2 CTest entries each**). The C23 configuration includes CUDA,
but this test explicitly selects CPU. The fixture executes the real tiny trained
transformer, verifies nonzero native generation work, compares output and token
counts against a direct engine request using the exact assembled prompt, forces
a 65,536-byte source through token-budget reduction, checks attribution and
rejects impossible token budgets and revoked access. It is not a language-quality
or real-large-model performance claim. An initial strict compilation caught
misleading indentation in the test; that fixture formatting was corrected.

Still unfinished: durable request-level provenance, client generation commands,
Desktop/TUI integration, streaming/cancellation, model registry identities and
the other context layers. The existing CLI `context explain` remains a preview,
not an inspector for a stored executed request.

## Layered assembly and generation

`fyodor_context_assemble_layers` and `fyodor_generate_with_layers` implement
explicit → workspace → session → retrieved → global priority, preserving order
within each layer. The existing explicit-only APIs are wrappers. Every listed
source is checked in one snapshot; retrieved sources additionally require
SEARCH. Duplicate URIs are rejected across layers. The same total byte budget,
UTF-8 truncation and native tokenizer reservation apply. Sources and executed
receipts retain layer names even when omitted by the budget.

See [the local client contract](CONTEXT_HTTP.md#context-layers) for CLI/TUI,
HTTP and Desktop controls, verified behavior and remaining automatic-provider
work. Layers are caller-supplied policy, not implicit grants or database dumps.
