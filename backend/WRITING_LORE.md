# Shared lore references in Writing

Writing documents, notes, characters and projects can reference existing world
lore by its shared URI. The lore content remains in the Explore resource; no
copy or new identity is created. World/lore creation currently uses native
resource import or the local API; the full Explore editor is still open work.

Writing metadata reserves `writing_lore` as an ordered array of up to 32 unique
canonical world-lore URIs. Omission or an empty array means no links. All linked
resources must be live in the same namespace. Native put/import/scoped-write
paths validate the field. Invalid types, duplicates, non-lore references and
missing targets are rejected before any mutation commits. A live Writing link
blocks deletion of that lore; unlink it or delete the referring resource first.
Historical links do not block deletion.

`fyodor_store_set_lore` replaces links in a revision-checked transaction. It
preserves content, other metadata and provenance, including large JSON integers.
Metadata, search index and history commit together. Clearing links removes the
field. Packages and historical revisions retain the original links; import the
world and lore before referring Writing resources. No schema change is needed.
Older arbitrary metadata using this reserved name must satisfy the new contract
before the resource can be written again.

## Context behavior

The Writing provider checks the target's lore after caller-selected sources,
then each readable parent project's text and lore, nearest project first.
Discovered sources enter the workspace layer; existing manual selections keep
their selected layer and reserve source capacity. Duplicates are omitted.
Discovery, permissions and content assembly share one snapshot. READ is required
for each lore entry; a link never grants it. Denied lore is skipped without
adding a source to the prompt or receipt, while discovery continues to other
links. Denied project parents stop ancestor traversal.

The existing 64-source limit and byte/token budgets apply. Receipts identify
original lore revisions and actual included byte ranges, even when a budget
omits all bytes. Later revocation hides receipts that used that lore. Editing
lore does not rewrite previously saved prompts or receipt attribution.

## Interfaces

Local administrator HTTP:

```text
POST /api/v1/resources/lore
{namespace, uri, expected_revision: "N", resources: ["LORE_URI", ...]}
```

Success returns the new decimal-string revision. `resources: []` clears links.
`/resources/list` supports `scope: "lore"` with existing pagination. These are
local administrator operations, not remote context authorization endpoints.

CLI and the native TUI command prompt:

```text
fyodor --store PATH --namespace NAME resource lore URI REVISION [LORE_URI...]
```

Desktop's Lore references section browses paginated saved lore, links/unlinks
entries, and reloads persisted links and titles. Save or discard text edits
before changing links. Linking adds a revision and clears a stale generated
preview. Allow model access separately replaces READ grants for the current
resource and its direct lore links; partial failures report the count changed.
The render-only component is `frontend/src/writing-lore.js`; native C owns the
relationship and permission rules.

## Evidence

Native tests exercise malformed/duplicate links, limits, history, metadata
precision, deletion guards, project lore, manual-source deduplication, revocation
and injected history-write rollback. Real HTTP inference verifies exact lore
text in the executed prompt; native CLI export/unlink agrees with HTTP state.
Browser QA links lore, explicitly grants access, generates, checks the receipt,
reloads and verifies the saved link/title. A separate native CLI export confirms
the same identity and retained generation provenance.

Seven affected CTest entries pass in C17, C23 and ASan/UBSan configurations;
browser QA, all 16 frontend unit tests and the production build pass. These are
Windows CPU checks. Explore gameplay, richer lore editing/search, link reordering
in Desktop and subtree package transfer remain unfinished.
