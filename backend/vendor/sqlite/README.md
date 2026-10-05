# SQLite 3.53.4

Unmodified `sqlite3.c` and `sqlite3.h` from the official amalgamation:
https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip

Archive SHA3-256, verified before extraction:
`628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e`

Extracted SHA-256:
- `sqlite3.c`: `b1dd5d74ec7f29055a6684fa06fb3c2f6821c87dd38f9a458dfd2e8a1db28189`
- `sqlite3.h`: `919e7f2e8ed1d8f56ac17b412b8971c76aa5d1a879752cc6058f75e7d5910e1d`

SQLite's deliverable code is dedicated to the public domain:
https://www.sqlite.org/copyright.html
The original notices remain in both source files.

This dependency provides application-resource transactions, concurrent access,
crash recovery and JSON validation. Reimplementing those facilities in ad hoc
files would add substantial correctness risk. It is pure C, statically linked,
and adds no language runtime, network service or frontend dependency. Source is
vendored to keep clean builds offline and reproducible. Engine/server/trainer
targets do not link it unless they explicitly consume application storage.

The separate target omits loadable extensions and uses SQLite's own warning
policy rather than patching upstream for Fyodor's conversion warnings. It still
uses the selected C standard and sanitizer configuration. The storage interface
does not expose SQLite handles or permit caller-provided SQL.

FTS5 is enabled in the static SQLite target for the schema-4 context trigram
index. The upstream amalgamation already contains this code; source hashes
remain unchanged and no extension is loaded at runtime. See
[context index](../../CONTEXT_INDEX.md) for semantics and migration validation.
