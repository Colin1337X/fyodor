# Dataset Studio: frontend behavior and native handoff

Status: 2026-10-06. Dataset Studio authors persistent typed dataset resources
through the existing local resource service. Native trainer source is inspected
read-only. This milestone does not add a dataset job/materialization backend.

## Implemented workflows

- Create a dataset with pretraining, CPT, SFT or DPO format.
- Import strict UTF-8 `.txt`/`.tsv` files into new unsaved resources.
- Select saved Writing/Explore revisions and local chat snapshots, map them to
  corpus/SFT drafts and preserve provenance; see [Source ingestion](DATASET_SOURCES.md).
- Edit title/content; inspect format issues and the first 50 filtered records.
- See bytes, record/nonempty-line counts and exact duplicate counts.
- Filter, sample, shuffle and remove exact duplicates into a new derived draft.
- Prepare separate training/validation drafts and save each independently.
- Persist datasets, reload them, recover stale edits and export saved packages.
- Export current training data as `.txt` or `.tsv`.
- Export and open Training with the chosen mode; explicitly select the downloaded
  file in the native Training form.

Permission-scoped Unified Context ingestion, automatic native file
materialization and persisted run/evaluation linkage remain open. They are
requirements of the full product brief, not implied by the current dataset editor.

## Files and ownership

`src/dataset-data.js` owns structural inspection, strict decoding, deterministic
transforms, typed identity and bounded catalog filtering. It is independently
testable without a DOM or server. `src/dataset-workspace.js` owns view state,
requests, draft lifecycle and downloads. `main.js` mounts the view and handles
the explicit export-to-Training navigation. Shared clay/layout styles remain
in `clay.css`; no new framework or editor dependency was added. Source mapping
and picker ownership are documented separately in `DATASET_SOURCES.md`.

The current namespace, draft, editor revision, mode, filter, seed, sample size,
split share and split drafts live in memory. The app persists the selected view,
but editor drafts/split preparations do not survive a full window reload. Dirty
drafts or unsaved split copies request the browser's unload warning where the
host supports it. Saved resources persist in the native store.

## Resource representation

Dataset IDs use the existing native identity:

```text
fyodor://datasets/<uuid>
```

New resources use schema-1 packages with plain corpus/TSV `content` and this
frontend-owned format annotation:

```json
{"dataset_studio":{"version":1,"mode":"sft"}}
```

The mode values are `pretrain`, `cpt`, `sft`, `dpo`. The native generic store
preserves this metadata as opaque JSON; the native trainer does not consume it.
Dataset Studio validates the annotation version/mode before treating it as a
saved format. Older/external dataset packages without it open for inspection;
Create copy establishes an explicit supported format. Changing a saved format
requires a new copy rather than rewriting arbitrary loaded metadata.

New packages use `/resources/put` with expected revision `"0"`. Existing title
and content updates use `/resources/update` with the loaded decimal-string
revision. This preserves unknown metadata/provenance natively rather than losing
numeric precision through JavaScript. A failed stale write keeps the draft.
Reload saved version is the explicit replacement action. Packages export as raw
native JSON; import existing packages through Resources' package importer.

Studio datasets/imports have a **1 MiB** limit because that is the existing
resource-content contract. The CLI's standalone file cap is 64 MiB; the Studio
does not pretend to offer that larger capacity. Large/streaming dataset storage
needs a separate native contract.

## Validation matches structural trainer rules

The authoritative implementation inspected is `backend/core/train_main.c`,
specifically `read_data`, `dataset_parse` and `encode_record`.

| Format | Studio structural rule | Native check still required |
| --- | --- | --- |
| Pretraining/CPT | Nonempty UTF-8 corpus without NULs, within Studio size bound | Model tokenization must produce at least two tokens; context creates actual training windows. |
| SFT | Each nonempty line has exactly two tab-separated fields; completion is nonempty | Concatenated prompt/completion tokenization, context fit and supervised boundary. |
| DPO | Each nonempty line has exactly three tab-separated fields; chosen/rejected are nonempty | Both tokenized branches, context fit, model support and frozen-reference preparation. |

Empty prompts are structurally permitted, matching the native implementation.
Whitespace-only completions are nonempty and are not rejected on style grounds.
An empty structured line is skipped; a whitespace-only line lacking tabs is an
issue. A final CR is removed from each parsed line for CRLF compatibility.
There is no quoted TSV or backslash escape grammar: tabs/newlines are delimiters.
Do not suggest JSONL/CSV or multiline quoted fields as accepted native inputs.

The structured parser's line allocation counts `1 + newline_count`; values
over 100000 are rejected, including the empty line after a trailing newline.
The UI matches this bound and displays line-specific format errors (first 30).
Record inspection renders the first 50 filter matches in an escaped table.

The format success message explicitly reserves model tokenization/context
validation for training. Character/byte/line counts are not token estimates,
training-window counts, loss or performance measurements. Model-dependent
acceptance cannot be proved by this frontend alone.

Strict file decoding rejects invalid UTF-8 and NULs, strips a UTF-8 BOM and
otherwise preserves import text. Browser textareas normalize displayed line
endings on editing. No incompatible source file or loaded dataset is overwritten
as a side effect of import.

## Transforms and provenance

Transforms require a saved, clean source with no structural issues. They use
structured TSV records for SFT/DPO and nonempty text lines for pretraining/CPT.
Corpus line units are an explicit authoring choice, not the native trainer's
token windows. Transform serialization joins complete units with LF and no final
newline; it does not preserve blank lines or CRLF separators. Original resources
remain unchanged. Corpus structure-sensitive users should export/edit directly
instead of applying line transforms.

Filter uses case-insensitive substring matching across the entire unit. Sampling
uses a shuffle without replacement. Shuffle/sample/split use a documented fixed
32-bit seeded generator and Fisher–Yates shuffle. Seeds range from 0 to 4294967295.
Deduplication compares exact parsed unit text and retains the first occurrence.
It is not semantic similarity or cross-source deduplication.

Split shuffles units with the chosen seed and takes the rounded validation share,
clamped to at least one and at most `N-1`. It requires at least two units and a
share from 1 to 99%. Training and validation positions do not overlap. Identical
duplicates in the source can still appear in both partitions: deduplicate first
when that would cause data leakage.

Each derived resource records descriptive provenance:

```json
{"dataset_studio":{"version":1,"operation":"deduplicate","source":{"namespace":"workspace","uri":"fyodor://datasets/<source-uuid>","revision":"1"},"parameters":{"query":"","seed":42,"count":10,"validationPercent":20},"units":"records"}}
```

`parameters` captures the current transform controls; only applicable parameters
affect an operation. References identify an exact source revision. Manual edits
are available through native revision history; this annotation is not an
authenticated generation receipt or an exhaustive edit log. Imported file
provenance records its filename and operation, not a local path or access grant.

Non-split operations replace the editor with a fresh unsaved derived draft.
Split prepares two separate resources while retaining the source editor. Each
Save training/Save validation action persists one copy and records its returned
revision. Saved copies remain saved if the other fails. This is deliberately not
an atomic two-resource operation. Navigation/reload guards block losing pending
split drafts; Discard editor discards remaining unsaved copies but does not delete
copies already saved. A lost HTTP success response can require catalog inspection
to reconcile a retry conflict; there is no native idempotent split/job API yet.

## Existing API and training boundary

The view uses existing `/api/v1/resources/list|get|put|update|export` authenticated
local endpoints. The generic catalog has no dataset scope, so a compatibility
helper scans at most 25 summary pages per action, retaining the opaque server
cursor and detecting repeats. It stops after at least 20 matches or exhaustion;
empty sparse pages can still have a Next page cursor.

Export training file downloads current editor text after a format check. It may
include unsaved edits; its filename extension follows the selected format.
Export package uses the saved resource and requires a clean draft. The browser
never turns a dataset URI into a filesystem path by guessing.

Export & open Training downloads the text, sets Training's mode and clears prior
data/validation paths to avoid reusing a file of the wrong format. The user must
select the exported file. This is a useful explicit handoff, **not direct native
resource training**. It does not launch a job, claim an output file or use a fake
path. The desktop's existing Training flow still requires base/output/settings
and its actual native trainer.

Native contracts still needed:

- Typed/scoped dataset catalog and large/streaming storage.
- Dataset-revision materialization into managed training/validation files, exact
  provenance, cleanup/lifetime, permission checks and failure reporting.
- Validation against the selected native tokenizer/context/architecture.
- Permission-scoped ingestion from selected Context resources, durable native
  chat snapshots and source/lineage services. The current human-selected
  Writing/Explore/chat frontend mappings are documented in `DATASET_SOURCES.md`.
- Atomic/recoverable derived dataset jobs, split linkage and job/run identities.
- Training/evaluation launch from a dataset revision, run history and reproducibility.

Keep the frontend annotation versioned and agree migrations before a native
service begins interpreting it. Do not silently grant source access or modify
source resources when adding ingestion.

## Verification

Five unit tests cover native structural grammar edge cases, bounds, import
decoding, mode annotation, deterministic sampling/splits, exact dedup/filter and
sparse catalog paging. Six source-domain checks bring the selected suite to 35.

The Windows headless browser flow creates/saves an SFT dataset, derives an exact
deduplicated copy, verifies its source remains unchanged, saves both split copies,
checks actual exported TSV bytes, reloads format metadata, rejects invalid export
and recovers the saved version. The Training handoff selects SFT and clears old
training/validation paths. Independent CLI export checks stored content and
source revision provenance. The exported TSV is passed to the existing native
trainer for a successful one-update SFT run against the tiny trained fixture.
This proves this fixture's export is executable; it is not convergence evidence
or proof that every dataset/model combination fits.

Light/dark/390px screenshots are in `frontend/qa/themes/datasets*.png`. Current
checks do not establish actual desktop-window/file-picker behavior, platform
installers, large-dataset responsiveness, full accessibility or the outstanding
direct training/ingestion backend. Run instructions are in `ARCHITECTURE.md`.
