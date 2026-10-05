# Local resource API and desktop library

The native server now exposes the shared `fyodor-app` store to Desktop. Tauri
creates its application-data directory and supplies `workspace.db` through
`FYODOR_STORE_PATH`; Rust does not implement or proxy persistence. Standalone
servers opt in with that environment variable (UTF-8 conversion of the Windows
wide environment is supported). Parent directories must exist. The server
validates the store at startup and rejects workspace-enabled remote or
non-127.0.0.1 binding. Without the variable, resource access returns 403.

Every resource request passes the existing bearer authentication and browser
origin checks. These endpoints are **local administrator operations**, not
principal-scoped context APIs. A namespace is a storage partition, not an
authorization credential. Remote resource access remains disabled pending
paired-device authentication and authorization. No requested body field can
select a database path. Errors do not include paths, credentials or content.

All routes below use POST with `Content-Type: application/json`. Fields are
UTF-8 JSON; `namespace` is required. Revisions are canonical decimal **strings**
to avoid JavaScript integer rounding. The normal server request-byte ceiling
still applies (default 1 MiB including headers); a resource package can exceed
that ceiling even when it fits the store's package limit. CLI import remains
available for larger packages. Server configuration can raise the HTTP ceiling.

| Route | Additional fields | Success response |
| --- | --- | --- |
| `/api/v1/resources/list` | Optional `after` URI, `scope` (`all`, `writing`, `projects`, `lore`), or `folder` (project URI; empty = unfiled) | `ok`, up to 20 `resources` summaries, `next` cursor |
| `/api/v1/resources/move` | `uri`, `expected_revision`, `folder` | `ok`, `revision` |
| `/api/v1/resources/lore` | `uri`, `expected_revision`, `resources` (0–32 lore URIs) | `ok`, `revision` |
| `/api/v1/resources/get` | `uri`, optional `revision` (`"0"` = live head) | `ok`, `revision`, `deleted`, `resource` package from one snapshot |
| `/api/v1/resources/put` | `resource` package, `expected_revision` | `ok`, `uri`, `revision` |
| `/api/v1/resources/update` | `uri`, `title`, `content`, `expected_revision`; optional `receipt_id` with `principal` | `ok`, `revision` |
| `/api/v1/resources/export` | `uri`, optional `revision` | Version-1 package JSON directly |
| `/api/v1/resources/history` | `uri`, optional `before` revision (`"0"` = newest) | Up to 20 revision/time/deleted summaries, `next` revision (`"0"` = end) |
| `/api/v1/resources/delete` | `uri`, `expected_revision` | `ok`, `deleted` |

Put with revision `"0"` creates; nonzero revisions compare-and-swap an existing
resource. Update changes only title/content while preserving metadata/provenance
inside C storage. Concurrent updates still use the shared store's revision
check. Delete is a history-preserving tombstone and rejects live children.
Conflicts return 409, missing resources 404, malformed input 400, store busy 503.
Each request owns its SQLite connection, so worker threads do not share a
transaction or handle. Responses are bounded; no model registry lock is held.

When update includes `receipt_id`, the native Writing save service validates
receipt ownership, current source access and the original target revision,
then commits text and receipt provenance together. Denied access returns 403;
the operation remains a local administrator edit. See
[accepted-edit provenance](WRITING_GENERATION.md#accepted-edit-provenance).

Desktop's Resources workspace creates documents/notes, browses namespaces and
pages, edits title/content, imports/exports packages and deletes saved resources.
Conflicts retain the editor draft and instruct the user to reload; there is no
silent overwrite. Metadata/provenance are not exposed as editable form fields.
Imports transmit validated original JSON text; exports download server JSON
without converting metadata numbers through JavaScript. Drafts are retained
across in-app navigation but not browser/process restart. Save before exporting.
This library is not yet the complete Writing, Dataset Studio or Explore UI.
The separate [Context workspace](CONTEXT_HTTP.md) now selects saved resources
for permission-checked native generation and receipt inspection.

## Verification

- Real HTTP tests: missing token, denied origin, disabled workspace, invalid
  JSON/revisions, create/read/list/delete, CLI interoperability, concurrent
  revision conflict, process restart, and metadata integer `9007199254740993`
  preservation during title/content update and export.
- Full CPU suite at initial integration: **63 passed, 4 hardware skips**.
  Final resource HTTP tests pass after adding update/export. The HTTP/API
  selection (including the trained-model fixture) passes under C23 and
  ASan/UBSan. No GPU claim follows from these checks.
- Headless Edge drives Resources → New document → Save, reloads the page,
  reopens the persisted document, verifies inert markup, and exports that same
  database record through a separate native CLI process. Screenshot:
  `frontend/qa/themes/resources.png`. Run
  `node frontend/tests/resource-browser.mjs` from the project root.
- Frontend production build and 16 frontend unit tests pass. Tauri `cargo check
  --locked` passes with the installed VS 2022 14.44 compiler and Windows SDK
  10.0.18362.0; automatic toolchain selection initially chose incomplete newer
  installations. Installer packaging and an actual Tauri window are not yet
  revalidated for this feature.

## Writing and historical revisions

The Writing page reuses the same editor and C store, with a native filtered list
of writing documents, notes, characters and projects. Filtering happens before
pagination. It creates documents, notes and plain-text character descriptions;
character UUIDs are shared resources, not copies stored in browser state.
Writing package imports accept only Writing URIs. Shared projects now support
[nested folders and revision-checked moves](WRITING_FOLDERS.md). Explore reuse
workflows remain unfinished.

Writing and Resources show revision history newest first. Selecting a revision
shows escaped historical title/text without changing the current editor.
Copy title and text to editor creates a dirty draft and retains the observed
head revision. Save uses the existing compare-and-swap update; it appends a new
revision, preserves current metadata/provenance and cannot silently overwrite
a concurrent edit. Dirty text must be saved or discarded first. Old history is
immutable. This is text restoration, not metadata restoration or resurrection
of a deleted identity. History includes tombstone summaries and explicit old
reads remain available to the trusted local administrator by URI.

The native C history API supports limits 1–100 and an exclusive revision cursor.
Each page is one snapshot; pages can see concurrent changes. Missing-resource
history is empty. CLI and the TUI command prompt expose the same operations:

```text
fyodor --store PATH --namespace NAME resource history URI [BEFORE_REVISION]
fyodor --store PATH --namespace NAME resource export URI [REVISION]
```

CLI history emits up to 20 JSONL summaries; pass the last revision to continue.
Export defaults to the live head. Explicit historical packages contain the
original title/content/metadata/provenance and retain the resource identity;
a tombstone package is content, not a deletion command. No schema migration was
needed: the existing immutable revision table supplies these operations.

Validation covers 25-revision pagination, namespace isolation, invalid limits,
tombstones, historical package content and filtering before a list limit.
HTTP tests check revision-string bounds, CLI history/export parity, exact large
metadata integers and refusal to update deleted resources. Browser QA edits a
document, reopens the first revision, copies its text, saves a third revision,
then generates from that document through Context. It also creates a Writing
character and verifies it remains listed after reload; writing.png was visually
inspected. Seven selected C17 tests, five affected C23/sanitizer tests each,
all 16 frontend unit tests and the production build pass.

Writing now has [native Generate/Rewrite/Continue previews](WRITING_GENERATION.md)
with context selection, explicit grants and executed receipts. Readable parent
projects and [linked lore](WRITING_LORE.md) now supply permission-checked workspace
context. Resource listing also accepts `scope: "lore"`. Accepted edits retain
their receipt references transactionally in resource provenance.
