# Evaluation authoring, execution and maintenance

Updated 2026-10-06. This is model-quality evaluation through existing local
generation and resource APIs. It adds no backend implementation or native job
service. Benchmarks remain a separate measured-report view, and Training retains
its actual loss/validation telemetry. The broader product brief remains open.

## User workflow

1. Open Evaluations under Run, choose a namespace and create a definition.
2. Add labelled cases with raw prompts and a supported correctness check. Set
   temperature, Top P, Top K, maximum new tokens and an exact nonnegative seed.
3. Save the definition. Choose one to four loaded local generation models and
   run the saved cases. Unsaved definition changes prevent execution.
4. Inspect each actual output, check result, request latency and native token
   counters. Errors appear as errors; they do not turn into scored empty output.
5. Save results explicitly as a separate resource. A definition is not replaced
   by its results. Open either resource later from the Evaluation library.
6. For human-review cases, select Pass, Fail or Needs review. Add notes to any
   row and save a new revision. Export the saved native resource package.

Every selected model receives the same definition revision, raw prompts and
settings. Cases run sequentially in definition order, with models in registry
order. A fresh generation call is made for each case. There is no implicit chat
template, system instruction, Writing draft, Context retrieval or remote provider.
Those additions would change the test and require an explicit future format.

Run details are saved even when individual requests fail. Stop prevents later
requests after the current synchronous request completes. It does not abort
native inference, manufacture outputs for skipped cases, or save automatically.
Stopping after the final request can still produce a completed run because no
requests remain. A stopped run can contain zero observations and still be saved.

## Module responsibilities

| Module | Owns | Does not own |
| --- | --- | --- |
| `src/evaluation-data.js` | Versioned formats, correctness checks, bounds, model snapshots, sequential scheduling and filtered catalog paging | DOM, native inference, permissions, database or native jobs |
| `src/evaluation-workspace.js` | Library/editor state, save/reload/copy/export, model selection, review controls and live results | New endpoints, loss calculation, model hashes or durable autosave |
| `src/api.js` | Existing authenticated resource/model/runtime/generation transport | Evaluation business rules |
| `tests/evaluation.test.mjs` | Behavior boundaries through injected generation and clocks | Evidence of actual native inference |
| `tests/evaluation-browser.mjs` | Evaluation flow in the private Edge/native harness | User browser state or production data |

The shell only registers the view, navigation icon and allowed persisted view
key. Clay material, colors and spacing come from existing semantic CSS tokens.
Do not embed implementation commentary or unsupported success claims in product
screens. Keep rationale and integration boundaries in these docs/source comments.

## Existing native API contracts

The controller uses the current administrative local resource interface:

- `backend.resources(namespace, cursor, 'all')`: bounded native catalog page.
- `backend.resource(namespace, uri)`: current saved resource and exact revision.
- `backend.saveResource(namespace, resource, '0')`: create a new typed resource.
- `backend.updateResource(namespace, resource, revision)`: compare-and-swap save.
- `backend.exportResource(namespace, uri)`: native schema-1 resource package.
- `backend.models()`: loaded model observations including `generation_supported`.
- `backend.runtime()`: observed model/compute selection before launch.
- `backend.generate(request)`: completed local native generation response.

Generation sends exactly:

```json
{"model_id":1,"prompt":"a","max_tokens":2,"temperature":0,"top_p":1,"top_k":0,"seed":42}
```

The response's `model_id`, `text` and `seed` must match the request contract.
Reported safe nonnegative `prompt_tokens` and `generated_tokens` are retained;
absent/invalid counters become `null`. Native HTTP errors or mismatched responses
become error rows. The next case still runs unless Stop or the result-size bound
ends scheduling. The backend's existing tokenizer/context rules are authoritative;
the frontend prompt byte bound does not guarantee a prompt fits a model.

Namespaces organize the local administrative store; they are not authentication
or Context-principal grants. Evaluation does not read Context receipts or acquire
source permissions. Remote paired evaluation will require separate authorization,
capability and event contracts. Credentials are never included in saved records.

## Persisted formats

Both definitions and results are native generic resources at
`fyodor://evaluations/<canonical-lowercase-uuid>`. They use native resource schema
1 with JSON serialized into the resource `content` string. Annotation:

```json
{"evaluation_studio":{"version":1,"kind":"definition"}}
```

The same annotation with `kind: "run"` identifies results. This is a versioned
frontend-owned convention over opaque native metadata/content. It is not a
native evaluation receipt, execution attestation or promise of CLI/TUI editing.
Resources can export/import the generic package; Evaluation supports opening its
known format. Unknown versions, kinds, fields or malformed shapes fail closed,
leaving the prior editor intact. No silent migration strips future fields.

A definition content object has `schema: 1`, `kind: "definition"`, `settings`
and `cases`. Each case contains a unique canonical UUID `id`, `label`, `prompt`
and a check `{type, expected, pointer}`. Empty definitions can be saved; running
requires at least one case. There are no native token counts during authoring.

A result content object has:

- `schema: 1`, `kind: "run"`;
- `definition: {namespace, uri, revision}` with the saved revision as an exact
  decimal string, never coerced to a JavaScript number;
- a copy of settings and selected model snapshots;
- `started_ms` and `finished_ms` from the browser wall clock;
- `status: "completed" | "stopped"`, `stop_reason: null | "requested" |
  "result-size"`, `planned` and actual `results`;
- per-row case ID/label/prompt/check, model index, saved text, truncation flag,
  original output byte count, measured latency, reported counters, derived
  observed rate, native stop reason, verdict/reason and review note.

Model snapshots contain `process_id`, `path`, `file_size`, `format`,
`architecture` and `compute_at_start`. Unavailable optional observations are
`null`. IDs are process-scoped; paths and sizes are not content hashes. Files
may change and model/compute observations are separate requests, not atomically
pinned to execution. Unload/provider changes and other API callers can intervene.
The browser does not lock native registry state or claim immutable model lineage.

New run provenance is
`{evaluation_studio:{version:1,definition:{namespace,uri,revision}}}`. Definition
copies record their source reference and whether the source draft was edited.
Copying creates a new UUID and an unsaved draft. Copying a new unsaved definition
can record source revision `"0"`; this is an editor-origin hint, not a saved source.
Existing title/content saves preserve native metadata/provenance, including opaque
fields beyond the Evaluation annotation. Generic imported records may be edited
or fabricated; format validation is not execution attestation.

## Correctness checks

| Check | Actual behavior |
| --- | --- |
| Human review | Output remains unreviewed until a human selects Pass/Fail. Error rows cannot be converted to successful human verdicts. |
| Exact text | Entire returned string must equal expected text, including whitespace and line endings. |
| Contains text | Case-sensitive literal substring; expected text must be nonempty. |
| Valid JSON | Entire returned string must parse as bounded JSON. Fenced JSON/prose wrappers fail. |
| JSON value at pointer | RFC 6901 pointer resolves own properties only; the value must deeply equal the expected JSON value. Object key order is ignored, array order is retained. |

JSON pointers support empty root and `~0`/`~1` escapes. Array indices must be
canonical nonnegative decimal indices; inherited properties and array `length`
are inaccessible. Structured values are limited to 32 nesting levels and 10000
visited values. JSON numbers use JavaScript JSON semantics, including its numeric
precision limits and last-value behavior for duplicate object keys. These checks
are unsuitable for exact large-integer validation without a future format change.

No arbitrary regex, code evaluation, script expression or model-generated judge
is executed. Automatic verdicts are fixed; notes may supplement them. Human
verdict changes update both the row and comparison counts immediately and restore
focus to that verdict control. Comparison reports counts by model, separating
checked, pass/fail, unreviewed and error rows. It does not silently average errors
into correctness scores or invent statistical significance.

## Measurement and reproducibility limits

`latency_ms` uses `performance.now()` around the complete awaited request. It
includes local transport/queueing, tokenizer, inference and response decoding.
`observed_tokens_per_second = generated_tokens * 1000 / latency_ms` only when a
safe reported counter and positive elapsed time exist; otherwise it is `null`.
It is not engine-only throughput, time to first token, isolated prefill/decode
timing or a controlled native benchmark. Zero reported tokens remains zero.

Start/end dates use a different wall clock and can move backwards after clock
adjustment; they do not determine latency. There are no warmup loops, cache
resets, concurrent workloads, statistical distributions, hardware attestations
or fair benchmark guarantees. Sequential model order and unrelated callers may
affect timing. A saved seed/settings/reference helps rerun a test but does not
guarantee bitwise output across providers, model bytes or engine versions.

Loss/perplexity are absent because generation does not expose them. Training's
measured validation loss remains in Training. Stable model identities/hashes,
training-output lineage, durable native evaluation jobs and native quality/loss
services belong to the backend handoff; do not compute plausible-looking values
or treat path/process identity as a substitute.

## Bounds and partial results

| Item | Bound |
| --- | --- |
| Resource content | Native 1 MiB; scheduling reserves 4096 bytes below that bound |
| Cases/models/requests | 32 cases, 1–4 models, at most 128 requests |
| Case label/raw prompt | 512 / 16384 UTF-8 bytes |
| Expected text/JSON pointer | 8192 / 512 UTF-8 bytes; pointer ≤32 segments |
| Saved output/reason/review note | 65536 / 2048 / 8192 UTF-8 bytes per row |
| Temperature/Top P/Top K | 0–5 / greater than 0 through 1 / integer 0–1000000 |
| Output token limit/seed | Integer 1–4096 / integer 0–Number.MAX_SAFE_INTEGER |
| Catalog scanning | At most 25 native pages per action, stopping at 20 matching records; repeated cursors rejected |

Output beyond 64 KiB is stored as a whole-code-point prefix with its original
UTF-8 byte count and an explicit warning. The check uses the complete returned
text before truncation. No full-output digest or retained tail is provided;
therefore a check on truncated output cannot be independently reproduced from
that saved prefix alone. Do not claim the saved prefix is the complete response.

If adding a completed row would exceed the content budget, scheduling stops with
`result-size`. That final completed request is not retained in the bounded result
resource; earlier rows remain. The planned/recorded counts expose incompleteness,
but do not distinguish dropped from unexecuted requests. A future native streamed
result store should remove this bound without breaking the versioned semantics.

The native HTTP request limit also includes JSON framing/escaping and resource
fields. A record below the content cap can still fail transport or metadata
limits. The reserve is not a guarantee for every title/provenance/escaped output.
Save failures preserve pending observations/reviews. Review notes can increase
size after execution and fail save; shorten or remove notes before saving.
Current package export requires successful Save and exports the native stored
head, which may have advanced independently since the editor loaded it. There
is no current draft-result export or historical Evaluation package selector.

## State, concurrency and recovery

Editor/library/model choices live in module memory, not durable browser autosave.
Saved definitions/runs live in the native store and survive a browser reload.
View switching retains drafts, with mount-generation guards preventing old
callbacks painting another screen. During a run, returning to Evaluations mounts
the current state and Stop control. The live result section updates without
rebuilding definition fields for every result. A standard before-unload warning
is requested for dirty drafts, pending results or active execution; host support
varies. Reload, power loss or forced exit can lose unsaved observations.

Busy guards disable editing and duplicate writes. Save uses exact revision CAS;
conflicts leave reviews/drafts unchanged. Explicit Reload saved version discards
editor edits and adopts the current saved head. New, namespace/library changes
and rerun block when dirty/pending; Discard editor/results releases those drafts
without deleting saved resources. Library refresh does not replace the editor.
Saving an edited definition while old pending results exist does not change their
already-copied definition reference; Save results still points to the run's source.

Persisted run fields are shape-validated, with unique model IDs and case/model
observations, nonnegative bounded measurements and consistent saved output bytes.
Completed records must have all planned rows; stopped records need a reason.
Live `running` objects are memory-only and not accepted as persisted results.
Known native request errors become rows. Unexpected controller/callback failures
may leave an unsavable partial live object; recovery currently requires explicit
Discard. There is no resumable job, retry-only-failed-cases or crash recovery API.

All titles/prompts/expected/output/errors/notes are escaped when rendered.
Imported package data is untrusted. Native controls and readable status text are
used; manual verdict changes preserve focus. Complete screen-reader, zoom,
high-contrast, keyboard-order and actual Tauri-window audits remain open.

## Backend and broader frontend handoff

The next contracts/features include saved evaluation datasets and explicit
dataset-to-case mappings, native tokenization validation, immutable model and
training-run identities, permitted exact revision materialization, native
execution/cancellation/events, large result paging, reproducible settings/version
attestation and genuine native loss/perplexity support. Quality definitions/runs
must remain separate from inference benchmarks and training validation records.
No new API is inferred from this format. These conventions should be reconciled
with a native Evaluation schema through an explicit migration, not overwritten.

## Validation

Run the production build and selected frontend suite listed in `ARCHITECTURE.md`,
including `tests/evaluation.test.mjs`. Its seven behavior checks cover text/JSON
checks, invalid definitions, exact raw generation payloads/revision strings,
reported versus missing metrics, stop/error semantics, Unicode output prefixes,
multi-model comparisons, paging, resource-size stopping and malformed records.

`node frontend/tests/resource-browser.mjs` owns a temporary native database,
two copies of the tiny trained fixture, local backend/Vite processes and private
headless Edge profile. The Evaluation scenario runs exact/manual/overflow cases
on both loaded copies. Identical fixture bytes are intentional; this validates
multi-model scheduling, not a claim of different quality. It checks actual native
tokens/errors, saved reload, human verdict counts/focus, independent CAS conflict,
recovery, inert review text and downloaded native package. Separate native CLI
exports inspect definitions, referenced revisions and reviewed result bytes.
Light/dark/390px images are retained in `frontend/qa/themes/evaluations*.png`.
Do not treat synthetic fixture timing as production model performance.
