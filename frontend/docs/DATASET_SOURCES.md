# Dataset source selection and provenance

Status: 2026-10-06. This is the implemented **frontend** source composer. It uses
the existing authenticated local administrative resource API and local chat
state. It does not implement the separately owned native Context ingestion or
dataset materialization service. General Studio formats, persistence and
Training handoff are documented in [Dataset Studio](DATASET_STUDIO.md).

## User flow

1. Open Datasets → Create from Fyodor sources.
2. Browse Writing, Explore or Local chats. Choose up to 32 sources; selections
   can span libraries/pages and retain their selection order.
3. Review the selected list. Remove individual sources or clear the selection.
4. Choose pretraining corpus, CPT corpus or SFT records. For SFT, enter the prompt
   for resource completions and choose an explicit delimiter policy.
5. Create dataset draft. Reads, mapping, size checks and hashes finish before the
   current dataset editor is replaced. A dirty editor or unsaved split copies
   block replacement; save or discard them explicitly.
6. Inspect/edit the resulting text, then Save. Creation is not an implicit save.
   Export and Training retain their existing explicit file workflow.

The picker is a disclosure above the editor so ordinary dataset editing remains
compact. It uses native controls and shared semantic clay tokens; no generated
source content is inserted as executable markup. Selected cards remain visible
when browsing a different library or page.
Selection/mapping actions restore their initiating control after the parent's
repaint so keyboard users can continue through the picker. Async completions
do not move focus into a different application view.

## Available sources and bounds

| Library | Included | Selection snapshot |
| --- | --- | --- |
| Writing | Saved documents, notes, characters and projects | Namespace, canonical typed URI, exact decimal-string revision and display title. |
| Explore | Saved worlds and world lore | Same revision reference; world saves/gameplay state are excluded. |
| Local chats | The frontend's current saved message arrays | A copied role/text array, window-local ID, title and creation timestamp. |

Unsaved Writing/Explore editor text is not ingested. Save it first. Selecting a
project copies that project's own text, not its descendants or linked lore.
Selecting a world copies its text, not all its lore; choose lore explicitly.
No automatic traversal, retrieval, permission grant or source mutation occurs.

Resource catalog requests retain opaque cursors, filter typed IDs and stop after
25 requests or at least 20 matches. An empty sparse page may still have a Next
page. Repeated cursors are errors. The Writing catalog uses the existing writing
scope; Explore uses the all-resource catalog until a scoped contract is available.

Selections persist in window memory across view changes. Changing the Dataset
namespace clears the source selection/catalog. A full reload loses selections
and drafts. Chat text is copied when selected, so a later chat edit does not
silently change the prepared source snapshot. Unselect/reselect to take a new
snapshot; browse again and reselect to use a newer saved resource revision.

Limits: 32 sources, at most 2,000 messages per chat, at most 1 MiB total raw text
per selected chat, at most 1 MiB mapped output, and at most 64 KiB provenance.
Role labels, separators and delimiter replacement can change output size. Native
HTTP request limits include JSON/header overhead, so a draft near the store's
1 MiB content cap can still fail Save; it remains available in the editor. Large
or streaming ingestion requires native storage/job contracts.

## Mapping rules

### Pretraining/CPT corpus

Resource content is copied exactly, including CRLF, tabs and Unicode. Sources
are joined with two LF characters; resource titles are not inserted. A chat
becomes its ordered messages, each rendered as `role:\ntext`, joined with two
LF characters. User, assistant and tool messages all appear with their role.
The system prompt, unfinished composer draft, metrics and credentials do not
enter the output. There is no invented chat template or model-specific token
count. The native tokenizer determines actual training windows.

### SFT

Each Writing/Explore source creates one `prompt<TAB>completion` record. The
explicit shared resource prompt supplies the first field; exact saved content
supplies the second field before any selected delimiter conversion. Empty prompts
are allowed by the trainer. Empty resource text is rejected.

Each chat contributes only adjacent user → assistant messages. User → tool →
assistant is excluded; later assistant messages, orphan messages and tool output
do not become fabricated training pairs. Multiple adjacent pairs preserve order.
A chat without any eligible pair fails the whole preparation. Corpus mode is
available when a role-labelled transcript is desired.

The native TSV parser has no multiline quoting or backslash-escape grammar.
Reject inside fields is the default delimiter policy. If a field has a tab, CR or
LF, preparation fails without replacing the editor. Replace with spaces is an
explicit alternative: CRLF is replaced with one space; individual tab/CR/LF
characters become spaces. Existing spaces are not trimmed or collapsed. This is
a lossy mapping recorded in provenance; it never alters the source text. NULs,
empty completions and other structural errors still fail.

Partial/interrupted assistant output, if saved in a chat, is ordinary source text.
The composer does not label it correct, complete or high quality. Review the draft
before training. DPO remains available in the ordinary Studio editor/importer;
this composer creates no preference labels or chosen/rejected pairs automatically.

## Revision reads and failure behavior

At creation the composer calls `/api/v1/resources/get` with the selected
nonzero revision string, never the moving `"0"` head. It checks the response's
revision, URI, deletion flag and text shape. Changed heads therefore do not change
selected output. Missing/unavailable revisions, failed requests, invalid mappings
or exceeded bounds stop the operation and preserve the existing editor and source
selection. Successful reads before a later failure do not persist anything.

Each selected resource is read independently. The result is a reproducible set of
explicit revisions, **not a single database-wide transaction snapshot**. Historical
reads use the native resource service's existing administrative visibility rules.
Creation issues no resource writes; Save later creates one fresh dataset UUID with
expected revision `"0"`. Neither source revisions nor chat arrays are modified.

## Provenance representation

New drafts retain version-1 opaque frontend provenance under `dataset_studio`:

```json
{
  "dataset_studio": {
    "version": 1,
    "operation": "source-ingestion",
    "output_sha256": "<SHA-256 of original mapped output UTF-8 bytes>",
    "mapping": "resource-prompt-content;adjacent-user-assistant",
    "delimiter_policy": "spaces",
    "resource_prompt": "Explain",
    "sources": [
      {
        "kind": "writing",
        "namespace": "workspace",
        "uri": "fyodor://writing/documents/<uuid>",
        "revision": "2",
        "content_sha256": "<SHA-256 of exact saved content UTF-8 bytes>",
        "first_unit": 0,
        "unit_count": 1,
        "byte_start": 0,
        "byte_length": 100
      }
    ]
  }
}
```

Corpus mapping is `resource-content;role-labelled-chat`, with delimiter policy
`preserve` and no resource prompt. `first_unit` is a zero-based mapped-piece index:
one piece per resource or corpus chat, one piece per SFT chat pair. `byte_start`
and `byte_length` refer to the newly mapped output, excluding preceding source
separators and including internal separators between a chat's pairs.

A local-chat origin has `kind: "local-chat"`, `local_id`, `created_ms`,
`message_count`, and `snapshot_sha256` instead of URI/revision/content hash. The
snapshot hash is SHA-256 of UTF-8 `JSON.stringify` of the ordered narrow array
`[{"role":...,"text":...},...]`. SFT adds zero-based `message_pairs` index arrays.
No native chat URI, resource revision, receipt or permission is invented. Chat IDs
are regenerated on session restoration; the snapshot hash is the byte identity,
while the ID is only a reference to that window's chat. Provenance alone does not
archive a full chat or reconstruct a deleted source.

Hashes and ranges describe **creation-time output**. Subsequent dataset editing
preserves source provenance but can invalidate those ranges/output equality. A
consumer can compare current content's hash to `output_sha256`; it must not assume
the current editor still equals the source mapping. Saved package export preserves
the provenance. Later Studio transformations refer to the source dataset revision
and its immutable history rather than expanding every prior source recursively.

## Native boundary and remaining contracts

This is an explicit local human authoring operation. Resource reads are local
administrator API calls; visible resources are not proof of principal/model read
permission. The picker does not use Context selections or claim permission-scoped
ingestion. Context's permission query followed by an administrative read would
introduce a revocation race, so it is not used as a substitute for native enforcement.

Backend-owned work still includes a native atomic permission/revision read or
dataset-ingestion service for selected Unified Context resources, durable native
chat identities/snapshots, large jobs/cancellation, lineage/run records, tokenizer
validation and direct dataset-revision Training materialization. Document exact
schemas, permission/revocation rules and migration of this versioned annotation
before wiring those services. No direct URI training is claimed here.

## Evidence and maintenance

`src/dataset-sources.js` owns typed filtering, snapshots, mappings, bounds and
hash/provenance generation. `src/dataset-source-picker.js` owns source/mapping
controls. Its parent supplies API, chat provider, busy wrapper and draft acceptance;
`dataset-workspace.js` guards and owns editor replacement/persistence.
`src/resource-text.js` shares native UTF-8 title bounds with Writing imports and
derived datasets. Implementation rationale stays in these sources and docs.

Six domain tests cover exact revision strings beyond JS integer precision,
Unicode/CRLF corpus bytes/ranges/hashes, explicit delimiter policy, chat snapshot
immutability/pair exclusion, source failures/size/provenance bounds, sparse paging
and complete-code-point title bounds. The selected frontend suite has 35 checks.

The Windows browser flow selects Writing revision 2 plus Explore lore revision 1,
advances the Writing head independently, verifies revision-2 output, checks dirty
replacement guards and actual corpus downloads, imports a real JSON chat, verifies
delimiter rejection/recovery and eligible SFT pairs, and confirms unchanged chat
messages. Independent CLI exports verify stored sources/hashes. The actual corpus
and chat downloads complete one native CPT and SFT update respectively against the
tiny fixture. These are execution checks, not convergence/model-quality evidence.
Light/dark/390px source-picker screenshots are in `frontend/qa/themes`.

Run the commands in [Architecture](ARCHITECTURE.md). Actual Tauri window/file
pickers, other platforms, large-source performance and a full accessibility audit
remain unverified.
