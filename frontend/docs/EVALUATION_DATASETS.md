# Reusable evaluation datasets and source mappings

Updated 2026-10-06. Test datasets are persisted collections of explicit prompts
and correctness checks. They use the existing generic native resource store;
native evaluation jobs, stable model lineage and loss remain separate contracts.
Definitions/results and measurement limits are documented in `EVALUATIONS.md`.

## Authoring and reuse

Open Evaluations under Run. **New test dataset** creates an unsaved collection.
Add/edit/remove cases, choose literal/JSON/human-review checks and Save. The
editor shows 20 cases per page, preserving absolute case indices and every edit
when changing pages. A collection supports up to 512 cases and 1 MiB content.
Generation settings belong to a definition, not to a reusable dataset. Test
datasets cannot be run directly; choose up to 32 cases for a definition.

The library selector switches between Evaluations/results and Saved datasets.
The dataset catalog includes training text/TSV and test collections because the
native catalog identifies typed dataset URIs but does not expose their opaque
format annotation. Opening a known test collection edits it. Opening training
data selects its saved revision in the mapping panel, preserving the current
editor. Unknown formats report an error and remain available through Resources.
Changing catalog pages/filters does not replace the current editor.

The collapsible **Create tests from a saved dataset** panel is above the editor:

1. Browse and choose a saved dataset. The selection pins its exact revision.
2. Choose an Evaluation definition or reusable test dataset as the destination.
3. Choose a consecutive range or seeded sample and an explicit number of cases.
4. For SFT, choose Human review, Exact completion text or Contains completion text.
5. Create the draft, inspect the actual prompts/checks and Save explicitly.

Draft creation requires a clean editor and no unsaved results. It creates a new
UUID and never overwrites the source. On success the panel collapses and focuses
the new title; on failure the prior editor/selection remain. Selection/mapping
controls remain in window memory across view changes, not across reloads.
Changing namespace clears selection. Refreshing the catalog does not advance an
already selected revision; select again to use a newer head.

The new definition copies settings from the current definition when one is open;
otherwise it uses Evaluation defaults. The selection seed and generation seed
are different settings. Review settings, save the new definition, choose models
and run through the ordinary actual-output Evaluation flow.

## Explicit mappings

| Saved source | Prompts and checks produced |
| --- | --- |
| Known test dataset | Copies selected IDs, labels, raw prompts and existing checks without reinterpretation. |
| SFT `prompt<TAB>completion` | First field becomes raw prompt. Human review ignores completion; Exact/Contains uses its literal text as expected output, only when explicitly chosen. |
| Pretraining/CPT corpus | Each nonempty line becomes a raw prompt for human review. Blank lines are omitted; a terminal CR is removed using Studio's line inspector. |
| DPO `prompt<TAB>chosen<TAB>rejected` | Only the prompt becomes a human-review case. Preference text is excluded; chosen does not become ground truth. |

Training input uses `inspectDataset()` and its existing structural grammar:
required tab counts, nonempty completion fields, NUL rejection, content/line
bounds. It does not replace native tokenizer/context validation. Corpus mapping
is an explicitly chosen line-to-prompt operation, not a claim that the native
corpus trainer treats lines as independent records. Source bytes remain unchanged.

Generated cases from training rows get fresh UUIDs and `Record <physical line>`
labels. SFT/DPO skipped blank lines therefore leave gaps in labels. Stored
reference indices use physical line numbers for training input and zero-based
case indices/IDs for test collections. Individual prompt/check/label bounds from
Evaluation still apply; one invalid or oversized selected case rejects the whole
draft. The mapper never truncates, skips a failing selection or normalizes text
to make a test pass. SFT expected text remains literal and case-sensitive.

## Range and seeded sample

A range uses a **one-based position among available nonempty records/cases**,
then takes the requested consecutive count. It is distinct from a physical file
line number. Both start and count must fit available records. Count is limited
to 32 for definitions or 512 for test datasets. Empty sources cannot produce a
test draft. Counts are validated, never silently clamped during preparation.
Selecting a source initially proposes up to 32 cases for convenience.

A sample uses `seededShuffle()` from Dataset Studio, then takes the count without
replacement. The recorded algorithm is `mulberry32-fisher-yates-v1`; selection
seed is an integer 0–4294967295. Same source revision, seed and count repeat the
same ordered records. For training rows, fresh case/resource UUIDs still differ
between preparations; reproducibility describes selected row text/order, not
byte-identical new package identities. Existing case IDs remain unchanged when
sampling a saved test collection. Generation uses its separate saved settings.

## Storage and versioned format

Test collections are native schema-1 resources at
`fyodor://datasets/<canonical-lowercase-uuid>`. Metadata:

```json
{"evaluation_dataset":{"version":1}}
```

Content is a JSON string representing:

```json
{"schema":1,"kind":"evaluation-dataset","cases":[{"id":"11111111-2222-4333-8444-555555555555","label":"Continuation","prompt":"a","check":{"type":"exact","expected":"bc","pointer":""}}]}
```

The collection has no generation settings, model selection, output or metrics.
Top-level/case/check fields are narrowly validated; IDs are unique canonical
UUIDs. Unknown content fields or versions fail closed. Extra unrelated native
metadata/provenance remains opaque and is preserved by title/content CAS updates.
An unknown `evaluation_dataset` annotation never falls back to treating its JSON
as ordinary training corpus, even if another training annotation is present.

This is a frontend-owned convention over native generic storage. The native
service does not validate test semantics or attest its creation. Dataset Studio
does not treat these JSON collections as trainer text/TSV; edit them in Evaluation
or inspect/export their generic package in Resources. A future native Evaluation
dataset schema must reconcile/migrate this version explicitly.

## Source revision and provenance

Selection builds `{namespace, uri, revision, title}` from a saved catalog item;
revision remains an exact decimal string, including values above JavaScript's
safe integer range. Initial inspection and preparation both read
`backend.resource(namespace, uri, revision)`. The requested URI/revision must
match the returned record, and deleted/unavailable revisions reject. A source
head changing after selection does not change the prepared cases. A generic
administrative read is not a principal-scoped Context grant or atomic multi-write
transaction. Mapping does not modify native permissions or resource history.

New mapped resources record `provenance.evaluation_studio` with:

- `version: 1`, `operation: "dataset-mapping"`;
- source namespace/URI/exact revision and SHA-256 of the complete retrieved source
  content's UTF-8 bytes, including original line endings/blank lines;
- source format: `evaluation-dataset`, `pretrain`, `cpt`, `sft` or `dpo`;
- explicit mapping such as `preserve-cases`, `prompt-completion-exact`,
  `prompt-completion-contains`, `prompt-completion-manual`,
  `nonempty-line-prompt-manual` or `prompt-only-manual`;
- range start/count or sample seed/count/algorithm;
- ordered source records: physical line numbers or original case indices/IDs;
- `mapped_cases_sha256`: SHA-256 of compact `JSON.stringify(cases)` at preparation.

The mapped-case hash describes the initial construction snapshot. Subsequent
editor changes, even before first Save, can change content without changing this
origin record. It is not a claim that current content still equals that hash.
Native updates preserve provenance, so later manual revisions retain their origin.
To verify an edited definition/run, inspect its exact saved content/revision;
run results copy actual prompts/checks and reference the executed definition.
Source hash/reference proves which input was selected, not correctness, native
execution attestation, model identity or perpetual source availability.

Provenance is bounded to 64 KiB and resource content to 1 MiB. Native HTTP framing
and nested JSON escaping can still exceed transport limits. Preparation/Save
failures retain the prior draft. No disk materialization or managed Training
launch is added by this mapping workflow.

## File interchange

**Download test cases** exports current definition/dataset cases as compact UTF-8
JSON in the format above, named `<title>.cases.json`. It validates the current
cases but does not require Save or mutate storage. Compact output keeps valid
case content within the import cap. It omits generation settings, source metadata,
provenance, native URI/revision, credentials and model observations.

**Import test cases** accepts a UTF-8 `.json` case collection up to 1 MiB. Strict
UTF-8 decoding rejects malformed bytes; an initial UTF-8 BOM is removed. JSON,
known format, case IDs, individual text/count and serialized-content bounds are
validated before replacing the editor. Invalid/future/oversized imports leave
the old editor intact. Import requires clean state and creates a new test-dataset
UUID, retaining the imported case IDs/labels/checks. It does not silently merge
cases into a definition, execute them or resolve external references.

Import provenance records `operation: "case-file-import"`, bounded filename,
original byte count and SHA-256 of exact raw file bytes, including a BOM if present.
This original file hash does not attest later edits. Importing a downloaded file
therefore preserves cases but creates a separate local collection and provenance.

**Export package** remains the native saved schema-1 resource export, including
URI/content/metadata/provenance. It requires a saved clean editor and exports the
current native head. Import that package through Resources; it is a different
format from `.cases.json`. No TSV or training-data download is inferred for a
JSON test collection.

## Maintenance and validation

`evaluation-datasets.js` owns format validation, strict import, exact source reads,
selection/mapping and hashes. It imports existing training grammar/shuffle helpers
without changing trainer behavior. `evaluation-editor.js` is render-only shared
case/settings markup; `evaluation-dataset-picker.js` owns window-local selection
and delegates requests/editor replacement to the parent controller. Source and
case text are always escaped. Repaint replaces nodes, so change listeners capture
state immediately and restore focus; case removal/pages also restore a meaningful
control. Keep protocol mutations and model generation outside these renderers.

Six domain checks in `tests/evaluation-datasets.test.mjs` cover known/unknown
formats, strict UTF-8, IDs/count/text/content bounds, exact large revision strings,
source hashes/physical lines, deterministic sampling, invalid ranges, explicit
corpus/DPO mapping, unavailable/oversized records, preserved checks/IDs and inert
paged markup. They run with the selected frontend suite in `ARCHITECTURE.md`.

The private browser/native harness adds an actual SFT source with 40 records,
selects revision 1, advances its head independently and prepares the old revision
into a saved test collection. It edits/reloads the second case page, downloads and
imports cases, rejects a future format without replacing the editor, selects the
imported revision, advances every live prompt and builds a seeded definition from
the old snapshot. That definition executes real two-model generation and saves
results. Independent native CLI exports check records, hashes, exact references,
original mapping versus manual edit, selected case identities and actual tokens.
Light/dark/390px authoring/selection images are retained as
`frontend/qa/themes/evaluation-datasets*.png` and
`frontend/qa/themes/evaluation-dataset-selection*.png`.

This does not establish all accessibility/platform or large-scale behavior.
Actual Tauri-window/installer, screen-reader, zoom/high-contrast and realistic
large-model audits remain open. Native jobs/cancel/events, evaluation loss,
immutable models/training lineage and a canonical native dataset format remain
backend contracts. The full original product goal remains broader than this step.
