# Desktop authorized context generation

Desktop's Context workspace calls the same `fyodor_generate_with_context`
service used by the CLI. This closes the local selection → permission check →
bounded assembly → native inference → persisted receipt → inspect workflow.
It is a dedicated context workspace; ordinary Chat/Playground requests are not
silently changed to include resources.

## Local HTTP contract

These POST routes use JSON and require the existing local bearer token and
allowed browser origin. Like resource routes, they require `FYODOR_STORE_PATH`,
127.0.0.1 binding and remote mode disabled. The bearer token authenticates the
local administrator. The requested principal UUID selects whose stored grants
are evaluated; it is **not remote identity authentication**. No pairing or
remote permissions are claimed by this API.

All routes require `namespace` and a canonical non-nil `principal` UUID:

| Route | Additional input | Result |
| --- | --- | --- |
| `/api/v1/context/search` | `query`, 1–128 UTF-8 bytes | Up to 20 summaries requiring READ and SEARCH; no continuation cursor |
| `/api/v1/context/permissions` | `uri`; optional integer `permissions` 0–63 | Exact stored grant mask; when supplied, replaces the mask |
| `/api/v1/context/generate` | Loaded `model_id`, `resources` array, optional parallel `layers` array, `prompt`, `byte_budget`, `max_tokens` | Saved `receipt_id`, actual `text`, prompt/generated token counts and context byte count |
| `/api/v1/context/receipt` | `receipt_id` | Exact `prompt`, `output`, `metadata_json`, `sources_json` |
| `/api/v1/context/receipts` | Optional `after` receipt UUID | Up to 20 authorized receipt summaries and `next` cursor |

Generation accepts at most 64 distinct resource URIs. With no `layers`, all
are explicit and retain caller order. When provided, `layers` must have exactly
one valid layer name per resource; ordering follows the shared policy below. Byte
budget is 0–1 MiB; token limit is bounded by server configuration. All listed
sources need READ, even those omitted by a zero/exhausted budget. Native
tokenizer checks can reduce context further to reserve all requested output
tokens; the user prompt is never truncated. Invalid or denied requests do not
proceed to native generation. The execution lock pins the model mapping while
the registry lock is released during computation, matching ordinary generation.

Sampling currently uses greedy temperature 0, top-p 1 and seed 42. Streaming,
in-flight cancellation, drafts, soft tokens and remote context sources are not
implemented here. Successful return requires durable receipt storage; inference
cannot be rolled back if receipt storage subsequently fails. Response bodies
are bounded by server output limits. Receipt response allocation accounts for
JSON escaping and is rejected if it exceeds that configured limit.

Receipt metadata and sources are returned as JSON **strings**, preserving the
stored numeric text exactly for inspection. A read requires both the original
principal and current READ on every source. Revocation/deletion hides later
receipt reads; it cannot recall output already disclosed. Receipts preserve the
executed snapshot when a source is edited afterward.

## Desktop behavior

Select resources from paged local lists, then explicitly choose Grant read only,
Grant read and search, or Revoke all grants. These replace the selected resources' grants for the
displayed identity; selection alone grants nothing. Multiple grant changes are
separate transactions, and partial failure reports how many completed. This
administrative list may show resources the chosen principal cannot read; the
generation service still enforces that principal's READ grants.

Choose a loaded model in the top bar, enter the prompt and budgets, and generate.
The result displays a receipt ID, output, exact prompt, source revisions/byte
ranges, and model/sampling metadata. History lists accessible receipts after
reload; click one to reopen it, or paste an ID directly. The browser remembers its context identity in local storage; the
Workspace and identity control accepts an existing CLI principal UUID. Identity
labels contain no credentials. Grants and receipts live in SQLite. Selections,
prompt drafts and receipt contents are not persisted in browser storage.

## Evidence and open work

- Real HTTP test loads the trained tiny native model, verifies denial before
  grants (including zero byte budget), rejects duplicate/65-source requests,
  executes generation, checks exact `abc\n\na` input and source revision,
  reads that receipt through the CLI, then verifies generation and receipt
  denial after revocation. Other-principal receipt reads also fail.
- Real browser test creates/saves a document, selects it, verifies denied
  generation, grants READ, generates two real CPU tokens, checks its exact
  context/prompt, reloads and reopens the receipt, revokes READ and verifies the
  receipt disappears and cannot be read, then explicitly regrants for a CLI
  receipt interoperability check. Screenshot `frontend/qa/themes/context.png`
  was inspected. The fixture is a correctness test, not a quality/performance
  benchmark.
- Full CPU CTest suite: **63 passed, 4 hardware skips**. Affected HTTP/compatibility
  tests plus the trained-model fixture pass under C23 and ASan/UBSan. Frontend
  production build and all 16 unit tests pass. No new Rust changes in this step.
- Remaining work includes token-aware retrieval/ranking/context layers,
  per-device authentication, ordinary Chat and
  specialized workspace integration, streaming/cancellation, installer/UI
  platform coverage, and the broader product brief.

## Receipt history

The C `fyodor_receipt_list` service returns 1–100 summaries per page, each from
one SQLite snapshot. It shares the receipt-read visibility predicate: exact
principal and namespace plus current READ on every source. Denied receipts do
not contribute IDs, timestamps, previews or totals. Summaries contain only ID
and creation time; reading prompt/output performs a fresh authorization check.

Pages use canonical UUID byte order and an exclusive UUID cursor, **not date
order**. Separate pages can see concurrent changes. HTTP and CLI page size is
20; CLI emits JSONL and accepts the last returned ID as its optional cursor:

```text
fyodor --store PATH --namespace NAME context receipts PRINCIPAL [AFTER_ID]
```

The same command is available through the TUI dispatcher. No schema migration
was needed. Desktop refreshes history after generation and explicit grant
changes. Revocation removes inaccessible rows; grants may make old receipts
visible again. Clearing browser storage changes the default identity; enter
the original identity to recover its history.

Receipt-history validation: all 15 selected application/HTTP/compatibility
tests pass in C17. Generation/history and HTTP tests plus their model fixture
pass under C23 and ASan/UBSan. Direct C tests cover two-page cursor traversal,
end-of-list, invalid limit and revocation. HTTP checks owner isolation,
exclusive cursors, CLI listing and revocation. Browser QA reopens a receipt
from history after reload and verifies revoked receipts disappear. Frontend
production build passes.

## Permission-filtered search

Desktop can switch from the administrator resource list to the shared C
`fyodor_context_search` service through `/api/v1/context/search`. Results require
both READ and SEARCH for the selected identity and namespace. Queries are
literal ASCII-case-insensitive substrings of title/content, with title matches
first and URI order within each group. Percent and underscore have no wildcard
meaning. A [persistent trigram index](CONTEXT_INDEX.md) serves queries of
at least three Unicode scalars; shorter queries use a scan. This bounded
first-20 result set has no continuation page, semantic ranking or snippets. Selecting a match adds it to ordered context
selection; it does not modify permissions. Changing identity or namespace
clears prior results and selections before loading the new scope.

HTTP tests cover missing grants, READ-only exclusion, READ+SEARCH inclusion,
ASCII case matching, literal percent, owner isolation and revocation. Browser
checks cover read-only exclusion, explicit search grants, selected matches and
revoked matches disappearing. The affected HTTP/compatibility tests and model
fixture pass under C23 and ASan/UBSan (four each). All 16 frontend unit tests,
real browser checks and production build pass.

## Context layers

Each desktop resource has an explicit/workspace/session/retrieved/global layer
selector. New selections from search default to retrieved; administrator browse
selections default to explicit. Selections and layer choices remain in memory.
The shared C service orders sources by that layer priority and then preserves
caller order within each layer. The total byte budget and native token budget
apply to the resulting ordered text. Duplicate resource URIs across layers are
rejected; zero-length omitted sources retain their layer and revision.

Every source requires READ, including omitted sources. Retrieved sources also
require SEARCH in the same database snapshot. Choosing a layer never creates a
grant or automatically enumerates a workspace/session/global corpus. This is a
caller-supplied layered assembly policy; automatic feature-specific context
providers and persisted layer memberships remain unfinished.

HTTP accepts an optional `layers` array parallel to `resources`; unknown names,
non-string values and unequal lengths are rejected. CLI and the TUI command
prompt accept optional `LAYER=URI` source arguments for `context explain` and
`context generate`. Bare URIs retain their previous explicit semantics:

```text
fyodor --store PATH --namespace NAME context explain PRINCIPAL 4096 workspace=fyodor://writing/documents/UUID session=fyodor://writing/notes/UUID
```

Executed receipts add the winning source's `layer` beside its exact revision
and byte ranges. Existing stored receipts remain readable and may lack that
field. Receipt reads continue to require current READ on all their sources;
SEARCH governs new retrieval inclusion, not previously saved receipt access.
No database migration or new runtime dependency was introduced.

Validation covers all five priorities, stable order within a layer, budget
omissions, duplicate/invalid layers, identity/namespace denial and retrieved
SEARCH revocation. Native generation preserves global-layer attribution through
real tokenizer truncation. HTTP verifies layer shape validation, retrieved
authorization and receipt attribution; a separate CLI checks layered preview.
Browser QA selects session context, generates, verifies the receipt layer and
reopens it after reload. The 15 selected C17 application/API tests, six affected
C23 and sanitizer tests, all 16 frontend tests and production build pass. The
additional truncation/stable-order checks pass in all three native builds.
