# Writing projects and folders

Projects are shared Writing resources that also act as folders. Documents,
notes, characters and other projects can have one parent project in their
namespace. Moving never changes a resource URI or grants access to it.

The reserved Writing metadata field `writing_parent` is either absent (unfiled)
or a canonical `fyodor://writing/projects/UUID` string. Null, empty strings,
non-project references, missing/deleted parents and cycles are rejected.
Existing unrelated metadata and provenance survive a move, including integer
values outside JavaScript's exact range. This uses the existing metadata and
history schema; there is no database migration. The field is reserved only for
Writing kinds; older arbitrary data using that name must satisfy the contract
before it can be written again.

All native put/import/scoped-write paths check membership inside the resource
writer transaction. The move service reads the current record, checks its
expected revision, changes only the parent metadata, and commits its index and
history update together. Failed validation, contention or history insertion
leaves the prior record intact. Output revisions are unchanged on failure.
Deletion refuses live Writing children; move or delete them first. Historical
revisions retain their original membership. Copying historical text in the UI
does not restore old membership.

Single-resource packages carry membership through metadata. Import ancestors
before their children into the target namespace. Folder/subtree bundle export
and batch moves are not implemented. Folder membership is not an authorization
inheritance mechanism. The shared [Writing provider](WRITING_GENERATION.md)
automatically selects readable ancestor-project text during generation; denied
parents stop traversal. Siblings are not automatically enumerated.

## Interfaces

The local administrator HTTP API adds `POST /api/v1/resources/move` with
`namespace`, `uri`, `expected_revision` (decimal string), and `folder` (project
URI, or empty string for unfiled). Success returns the new revision string.
The resource list accepts `scope: "projects"` or optional `folder` to page
direct Writing children. A folder filter implies Writing scope; omitted folder
means the existing all-resources/scope behavior. Filtering precedes pagination.
A nonexistent folder has an empty listing. These queries currently scan
metadata, while title/content context retrieval retains its separate index.

CLI and the TUI command prompt use the same C services:

```text
fyodor --store PATH --namespace NAME resource move URI REVISION PROJECT_URI_OR_ROOT
fyodor --store PATH --namespace NAME resource folder PROJECT_URI_OR_ROOT [AFTER_URI]
```

Use the literal `root` for unfiled resources. Folder lists return up to 20 JSONL
summaries; use the last URI as the next cursor. Creation uses resource import
or Desktop's New project / folder action.

Desktop can create projects, filter by folder, create resources inside the
current folder and move saved resources. Save/discard text changes before a
move. A successful move clears stale generation previews, refreshes history and
keeps the resource open even when it leaves the displayed folder. Project
selectors currently list names without a full ancestor breadcrumb. Browser
restart preserves saved membership, but not the selected folder or drafts.

## Verification

Native checks cover nested projects, direct-child paging, self/ancestor cycles,
stale revisions, deletion guards, namespace isolation, package interchange,
metadata precision/size limits, history, rollback and reopen. HTTP/CLI tests
verify the same folder and revision results. Browser QA creates a project,
moves a character, filters the folder and reloads; a separate native CLI export
verifies both membership and the character's accepted-generation receipt.

Twelve selected C17 application tests pass; seven affected C23 and ASan/UBSan
tests each pass. The final metadata size-boundary check also passes in all
three configurations. These tests use CPU execution on Windows.
