# Persistent context search index

Schema 4 adds a SQLite FTS5 trigram index to the shared C resource store. Normal
resource creation/import registers searchable title/content automatically.
CLI, TUI and Desktop/HTTP keep using `fyodor_context_search`; no client-specific
index or new network/runtime dependency is introduced.

## Literal search policy

Queries with at least three Unicode scalars use an escaped FTS phrase over
ASCII-folded title/content. The trigram tokenizer is case-sensitive so
non-ASCII characters keep the existing exact matching semantics. Quotes are
doubled, and user text cannot introduce FTS operators or column expressions.
One- and two-scalar queries use the original permitted-row scan. Query size
remains 1–128 UTF-8 bytes; control characters remain invalid.

Candidates join live resources and exact namespace/principal grants in one
SQL statement. READ and SEARCH must both be present. A final literal substring
check preserves the old semantics. Title matches still precede content-only
matches, followed by canonical URI order; limits are applied after filtering.
There is no corpus-frequency score that could change with inaccessible data.
No denied IDs, counts, snippets or source text are returned.

The implementation follows SQLite's documented [trigram tokenizer and
contentless-delete index](https://www.sqlite.org/fts5.html#the_trigram_tokenizer).
The bundled SQLite target enables FTS5 at compile time; loadable extensions
remain disabled. This is indexed literal retrieval, not semantic search,
embedding retrieval, automatic query expansion or a measured large-corpus
latency guarantee. Very common terms can still produce many candidates.

## Storage and transactions

`resource_search_keys` assigns stable integer IDs to namespace/URI pairs.
These IDs survive VACUUM and do not rely on implicit resource rowids. The FTS
table stores derived index entries, not another complete title/content copy.
Keys remain for tombstones; deleted resources have no searchable FTS row.
Historical revisions and previously written SQLite pages are not erased by
logical deletion. The entire database still needs filesystem protection.

Index replacement/deletion is part of the same writer transaction as head and
history updates. This includes trusted administrator writes/imports and scoped
WRITE/APPEND/DELETE APIs. Failed index or history writes roll back all changes.
Grant changes need no reindex because every search checks current permissions.

Opening a validated schema-3 database creates keys and indexes live content in
one transaction. Older supported schemas first follow their existing migrations.
Resources, history, grants and receipts are preserved. Exact schema validation
includes the FTS virtual table and its four shadow tables; unexpected objects
and changed definitions are rejected. Schema-3 binaries cannot open schema 4.
Resource-package schema remains 1. Direct database mutations bypassing the C
service are unsupported and may invalidate derived search data.

## Verification

`application-context-index` compares indexed results against an independent
literal SQL scan, including ordered IDs and revisions at limits 1, 7 and 100.
The corpus covers title/content ranking, ASCII case, non-ASCII case distinction,
Korean, emoji, quotes, FTS operator text, percent/underscore, whitespace and
128-byte escaped-query boundaries. It checks READ-only exclusion and namespace
isolation, scoped update/append/delete, revoked grants, rollback after both
index and history failures, index integrity, VACUUM, reopen and populated v3
migration. Existing v1/v2 migration tests now remove v4 index objects when
constructing those historical fixtures.

The full C17 CPU suite passes 64 tests with four hardware skips. The eight
affected store/context/generation/HTTP/compatibility tests pass under C23 and
ASan/UBSan. Real browser search, grants, generation, receipts and CLI interchange
pass. An initial concurrent CPU/browser run collided on a mapped model fixture;
the browser harness now loads a private copy, and the complete CPU rerun passed.
The existing vendored SQLite GCC string-overread warning remains unchanged;
no exercised sanitizer failure was reported.
