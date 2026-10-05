# Application resource store

The current database is schema 4, adding a transactional
[context search index](CONTEXT_INDEX.md) after execution receipts in schema 3
and context grants in schema 2. See [request receipts](REQUEST_RECEIPTS.md) and the validated,
transactional migration described in [context services](CONTEXT.md). The resource
and history format below is unchanged from schema 1. The validation counts in
this report describe the original storage checkpoint.

`fyodor-app` now stores typed resources through the C API in
`include/fyodor_store.h`. It is an in-process persistence service, independent
of the model engine, HTTP server and Desktop shell. It is not yet wired into
the existing interfaces and is not an authorization boundary.

## Persistent behavior

Open an explicit UTF-8 database path whose parent directory exists. The store
creates a schema only in an empty SQLite database. A non-Fyodor application ID,
unknown schema version, unexpected table/index/trigger/view or changed schema
definition is rejected. Version 1 is the first supported schema; there is no
older Fyodor database migration to execute yet. Browser storage is unchanged.

Resources are keyed by `(namespace, canonical URI)`. Each holds a title, UTF-8
content, JSON-object metadata, JSON-object provenance, revision, creation and
modification timestamps, and deletion state. Tags can be stored in metadata;
they are not separately indexed yet. Namespace names are bounded ASCII tokens.
Content is limited to 1 MiB, each JSON object to 64 KiB, titles to 1,024 bytes.
The service rejects malformed UTF-8 and title controls before beginning a write.

Creation requires expected revision 0. Updates and deletes require the live
revision the caller observed. A competing update produces a conflict rather
than overwriting newer data. Each successful operation changes the current row
and appends its complete history record in one transaction. SQLite serializes
writers across connections and processes, with a one-second busy timeout.
Callers must serialize operations on a single store handle.

World lore, world saves and agent runs require a live parent in the same
namespace. A parent with live children cannot be deleted. Deletion creates a
tombstone, preserves history, and prevents identity reuse. Explicit history reads
can inspect tombstones; normal head reads hide deleted resources. History is not
automatically pruned. Deletion is therefore not secure erasure.

Timestamps use the SQLite VFS wall clock, in milliseconds since the Unix epoch.
An update does not decrease its prior modification timestamp when the clock
moves backward. Revisions, not wall-clock values, establish mutation order.

Rollback-journal transactions use `synchronous=EXTRA`; loadable extensions are
compiled out, defensive mode is enabled, and untrusted schema features are
disabled. Only fixed SQL statements with bound values are executed. These
settings do not encrypt files or protect against an actor with local write
access to the workspace. Place the database in the user's private application
directory. POSIX new-file mode is configured as 0600; Windows inherits directory
ACLs. Network-filesystem locking and power-loss guarantees have not been tested.

The SQLite dependency is vendored, pinned and documented in
`vendor/sqlite/README.md`. Existing engine, trainer and server executables do
not acquire a SQLite dependency. The Windows C17 store test executable is
1,594,427 bytes including the test harness and statically linked SQLite; this
is not a shipped CLI size claim. GCC emits an upstream `sqlite3Strlen30`
`-Wstringop-overread` diagnostic in the unmodified amalgamation. Fyodor's own
code builds with warnings treated as errors. Clang sanitizer tests report no
finding on exercised paths; that does not prove the upstream warning harmless.

## Executed validation — Windows, 2026-09-29

- Complete C17 CPU CTest suite: **54 passed, 0 failed, 4 hardware skips** for
  unavailable ROCm/MLX. Log: `build-cpu/resource-store-tests.log`.
- Final focused identity, store and recovery tests: **3 passed, 0 failed**.
- Final LLVM-MinGW C17 ASan/UBSan application tests: **3 passed, 0 failed**.
- GCC C23 application tests: **3 passed, 0 failed** in the CUDA-enabled build;
  these tests perform no GPU work.
- Store behavior covers reopen, original-content history, tombstones, conflicting
  revisions, namespace isolation, child/parent lifecycle, UTF-8 and size limits,
  malformed JSON, lock contention, foreign/future schema and added-trigger
  rejection. Injected history failure rolls back both update and deletion.
- The process fixture uses separate native executables to create, reopen, crash
  with an uncommitted spilled transaction, and recover. It verifies a recovery
  journal exists after the intentional exit, then verifies unchanged head/history
  and journal cleanup. Two competing native processes updating revision 1 yield
  one success and one conflict, then a fresh process verifies revision 2.
- That process sequence runs for ASCII and UTF-8 Korean/emoji filenames, inside
  a Unicode directory. It tests the C storage API's UTF-8 path behavior, not
  Windows CLI argument conversion. Python is only the optional test driver.

The recovery fixture terminates itself with `_Exit(23)`; it does not terminate
unrelated processes. No GPU benchmark or process-environment modification was
performed for this milestone. POSIX execution, storage-device failure, resource
package import/export, authorization and Desktop integration remain unverified
or unimplemented. A SQLite file is not an accepted untrusted resource package.
