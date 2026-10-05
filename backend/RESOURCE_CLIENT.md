# Native resource commands and packages

The client now also supports [context generation and durable request
receipts](REQUEST_RECEIPTS.md). Engine-independent application services remain
available separately; the command dispatcher now lives in `fyodor-client`.
The counts and binary size below describe the earlier resource-only checkpoint.

Build target `fyodor` now provides local resource operations in pure C. It calls
the shared `fyodor-app` service in process, with no child commands, WebView,
model engine or language runtime. `fyodor_command_run` in `fyodor_client.h`
accepts caller-owned input/output/error streams; later native interfaces can
reuse its command behavior alongside the typed application APIs.

The current executable is a resource client foundation, not the full CLI in
the product brief. Generation, models, training, remote nodes, context, agents,
TUI rendering and Desktop integration remain unfinished.

## Commands

```text
fyodor resource id document
fyodor --store workspace.db --namespace local resource list
fyodor --store workspace.db --namespace local resource export URI
fyodor --store workspace.db --namespace local resource import [EXPECTED_REVISION]
fyodor --store workspace.db --namespace local resource delete URI EXPECTED_REVISION
```

`id` emits a new canonical URI for a top-level resource kind, without opening
a database. Nested resource identities must currently be supplied in packages.
Import reads one complete package from stdin, bounded to 8 MiB. Omit the revision
or use zero for create-only; pass the revision previously observed in `list`
for an update. Delete also requires that observed revision. Conflict never
silently overwrites current data or revives a deleted identity.

List emits JSONL, one live resource per line, with URI, title and revision.
Other successful commands emit JSON; empty lists emit nothing. Diagnostics go
to stderr. Exit codes: 0 success, 1 storage/allocation/I/O error, 2 usage or
invalid input, 3 missing resource, 4 revision/parent conflict, 5 busy workspace.
Windows uses a UTF-16 entry point converted to UTF-8 and binary stdin/stdout.
Only the explicit path and namespace are used; defaults do not silently select
a different workspace. The parent directory must already exist.

## Version-1 package

```json
{
  "schema": 1,
  "uri": "fyodor://writing/documents/01234567-89ab-4cde-8f01-23456789abcd",
  "title": "A beginning",
  "content": "The first line.\n",
  "metadata": {"tags": ["draft"]},
  "provenance": {"source": "local"}
}
```

These six fields are required, with the exact types shown. Unknown fields,
future versions, duplicate decoded keys at any depth, embedded NULs, malformed
UTF-8/JSON and record-limit violations are rejected before mutation. Escaped
spellings of duplicate keys are also rejected. Namespace and expected revision
are chosen by the caller, never supplied by the package. Parent resources must
exist in the target namespace before importing children.

Packages carry user content and untrusted metadata/provenance. They neither
grant permissions nor prove source authenticity. They omit store namespace,
revision history and application credentials; they do not redact secrets a
user has explicitly placed in document content or metadata. Metadata such as
an arbitrary `permissions` property has no authority. Export/import preserves
JSON values, not insignificant whitespace. Imports use the same transactional
history and conflict checks as direct application writes.

The listing API returns at most 100 summaries, sorted by canonical URI bytes,
with a URI cursor for the next page. Each page uses one read snapshot. Multiple
pages are not a single snapshot while another process modifies the database.
The command streams these bounded pages rather than loading every document.

## Validation, Windows 2026-09-29

- Complete GCC C17 CPU suite: **56 passed, 0 failed, 4 ROCm/MLX hardware skips**.
  Raw CTest output: `build-cpu/resource-cli-tests.log`.
- Final application tests: **5 passed** under LLVM-MinGW ASan/UBSan and **5
  passed** under GCC C23. The C23 build enables CUDA but these tests use no GPU.
- Package tests cover cross-namespace round trips, explicit revision conflicts,
  exact-size unterminated input, Unicode and JSON escaping, maximum 1 MiB content
  expanding beyond 6 MiB, duplicate/unknown fields, embedded NUL and oversized
  input rejection, and 105-resource pagination with deleted entries excluded.
- Actual CLI subprocess tests cover UTF-8 Korean/emoji database paths and titles,
  piped stdin, parsed stdout/JSONL, stderr/exit codes, create/update/delete,
  failed stale updates, malformed/oversized packages, and retained content after
  failures. They invoke the native executable, not an alternate implementation.
- The GCC C17 executable is **1,590,679 bytes**, importing only `bcrypt.dll`,
  `KERNEL32.dll`, and `msvcrt.dll`. No new runtime dependency was introduced in
  this checkpoint. No startup-performance claim, POSIX execution, remote-client
  functionality or full-product completion is implied.
