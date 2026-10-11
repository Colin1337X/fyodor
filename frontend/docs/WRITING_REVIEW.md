# Writing generation review

Implemented 2026-10-11. This extends the editor-first Writing studio with a
separate prose review panel, proposed-text comparison, readable native Context
attribution, actual request counters and explicit preview download/discard.
Backend code, endpoints and persistence formats are unchanged.

## Author workflow

Save a document, note, character or project, choose a loaded local generation
model and configure Creative controls. Set read grants deliberately for the
target/additional sources and linked lore where needed. Generate, Rewrite and
Continue still execute the existing native Writing provider. Saving before
generation is required; the model reads saved resources, not an unsaved editor.

The returned prose appears below the editor in a clay panel. It stays visible in
Focus on text mode. The panel identifies the operation and exact source revision,
shows the full bounded output, and reports actual prompt/generated token counts
and included context bytes. Empty model output is explicitly identified.

Open **Compare proposed text with saved text** to review the changes. Generate
and Rewrite propose replacing the source text; Continue proposes its exact
concatenation with the returned text. No separator, newline, whitespace cleanup
or Markdown interpretation is inserted. The comparison's baseline is the exact
saved editor snapshot at generation start. It does not drift with later typing,
rename the preview as a saved revision, or imply that anything has been saved.

**Append preview to editor** or **Replace editor text with preview** copies the
proposed text into the unsaved editor. Save still creates the native revision and
accepted receipt provenance. Once copied, the same preview cannot be applied
again. Authors may edit the copied prose before saving; the native service
records whether accepted text was edited.

**Download preview (.txt)** downloads only the raw generated output, independent
of editor edits or receipt availability. It is not the proposed entire document
or a resource/receipt package. The shared portable filename helper appends
“preview” to the original title. Text is UTF-8 without a BOM. **Export draft**
continues to export current editor text and **Export package** exports a clean
saved native resource.

**Discard preview** removes the review output/loaded receipt and leaves all
editor text, dirty state and any already accepted receipt association intact.
Thus discarding a copied preview does not turn its following Save into an
unattributed ordinary edit. Discard editor, reload, namespace/resource changes
and new generation retain their existing distinct behavior.

## Modules and ownership

| File | Responsibility |
| --- | --- |
| `src/writing-generation.js` | Response/baseline validation, receipt binding, exact proposed text, acceptance guards, review markup and lazy disclosure/input listeners. No network or writes. |
| `src/resources.js` | Native requests, preview adoption, permission-checked receipt refresh, explicit download/discard/apply, dirty state and accepted receipt CAS save. |
| `src/writing-studio.js` | Moves the review node into the wide editor column while moving Creative controls into the tools column. |
| `src/revision-comparison.js` | Shared exact bounded line comparison and escaped line presentation. Historical and generated comparisons keep separate labels/lifetimes. |
| `src/context-receipt.js`, `src/context-receipt-view.js` | Exact numeric revision parsing, UTF-8 byte slicing and readable native included/truncated/omitted source presentation. |
| `src/writing-files.js` | Existing text download bytes and portable filename policy. |
| `src/clay.css` | Shared semantic material, readable wrapping prose, comparison labels and responsive review/source layout. |

The review has no separate store/controller, production dependency, artificial
internal transport, Markdown executor or native backend mutation. Implementation
rationale lives in code comments and documentation.

## Request and receipt contracts

`POST /api/v1/context/writing` retains namespace, principal, process model ID,
target URI, exact expected revision, mode, additional resources/layers, user
instructions, context byte budget and requested output tokens. Its completed
response contains raw `text`, `receipt_id`, `prompt_tokens`, `generated_tokens`
and `context_bytes`. The frontend captures the target's exact title/text and
revision separately from mutable editor state. Revisions stay decimal strings,
including values beyond JavaScript's exact numeric range.

The request's native receipt is fetched through
`POST /api/v1/context/receipt` with the original principal/namespace/receipt ID.
The existing service returns `prompt`, `output`, `sources_json` and
`metadata_json`, enforcing current source permissions. The frontend requires:

- receipt output exactly equals returned preview text;
- metadata `writing_mode` equals the requested operation;
- first source URI/revision equals the target snapshot and uses the explicit layer;
- every counter reported by generation agrees with receipt metadata;
- the shared receipt parser validates schema, context boundary, all ordered source
  byte ranges/layers/revisions and strict UTF-8 slicing.

Malformed/mismatched/unreadable receipts are not adopted and disable acceptance.
A valid bounded generation output may remain visible/downloadable if receipt
fetch fails after generation. There is no administrative resource-read fallback,
auto regrant, guessed prompt text or substitution of a different receipt.

The Context disclosure shows actual sources, exact revisions, layers, included
versus original bytes and inclusion status. Per-source text inspection shows at
most 2,000 code points; the exact executed prompt retains all included text.
Raw source JSON, prompt and model/sampling metadata remain available in the
separate disclosure. Native model path/file size are observations recorded in
that request, not immutable model lineage. Timing, streaming speed and model
performance are not invented from counters.

## Permissions, edits and persistence

Loaded receipt text is a client observation. It cannot detect an external
revocation while sitting in memory. **Read saved receipt again** clears the
loaded receipt first, then authorizes through the native receipt API anew. A
denial removes its prompt/source display and disables acceptance, retaining raw
model output and editor text. Explicit regrant followed by reread can recover.
The receipt's original principal remains pinned even if another view selects a
different Context identity. No preview action grants additional access.

Acceptance also requires a clean editor, matching target/revision and unchanged
baseline content. Native typing updates the disabled action/status without
replacing the textarea or losing its selection. A rerender performs the same
checks. Combined continuation must fit the native 1 MiB content bound.
An external writer can still advance the head after generation: copying to the
local editor is not a native authorization/transaction. Final accepted Save uses
the existing native receipt validation and exact revision CAS; conflicts or
revoked permissions preserve unsaved prose. No automatic stale retry/merge occurs.

Preview/base/receipt state is window memory, surviving workspace rerenders but
not reload/process exit. Receipts themselves already persist in the native
store and remain inspectable from Context subject to current permissions.
Discarding a preview never deletes a persisted receipt. Closing with an unaccepted
preview does not create a new unload warning; existing dirty-editor warnings
remain. Native Writing has no streaming/cancel contract, so this completed-response
review does not claim either capability.

## Bounds and rendering

Canonical non-nil UUIDs, supported Writing identity families, positive exact
signed-64-bit revisions, operation and original principal are validated. Title
is at most 1,024 UTF-8 bytes. Baseline/output are each at most 1 MiB, with no NUL;
continued proposed content must also fit 1 MiB before acceptance. Counters are
nonnegative safe integers. Missing usage remains `null` and displays Unknown,
rather than zero or an estimate.

Comparisons are lazy and share the history limits: 1 MiB per text, 20,000 lines,
250,000 changed-region LCS cells and 1,000 displayed rows. Unsupported comparisons
report the limit while full prose remains available to read/download. Added and
Removed labels work independently of color. All model/source/title/error text
is escaped; output is plain text, never executable HTML/SVG/URLs. Viewport scroll
limits do not truncate stored output.

Listeners belong to the mounted review/editor nodes. Active-view/mount guards
prevent an old disclosure painting another workspace. Input guards update only
the preview action/status. There is no per-keystroke comparison, network read,
editor repaint or duplicate model execution.

## Verification and remaining scope

`tests/writing-generation.test.mjs` covers exact baseline/append/replacement and
empty output, large revisions, receipt mismatch/counter/source binding,
dirty/stale/duplicate/oversized acceptance, invalid native response bounds,
unknown usage, inert markup and bounded comparison. Existing history comparison
and source/receipt tests remain required.

`tests/writing-generation-browser.mjs` composes the full owned native/browser
matrix. It exercises real completed native generation, exact preview downloads,
comparison without editor mutation, actual permission revocation/denial/regrant,
native keyboard typing, disabled acceptance across rerender, dirty draft
preservation on discard/reload and duplicate-apply prevention. Discard after
copying is followed by native Save; a separate pure-C CLI export verifies accepted
receipt provenance still persists. Light/dark/390px review images are retained
only after the full matrix passes. See `../qa/WRITING_REVIEW.md` for evidence.

Full Snep parity, inherited controls/trackers/batch workflows, rich document
interchange, native jobs/stream/cancel, Explore character references, final
platform/installer/accessibility and realistic authoring-scale audits remain in
the active full product goal. This review does not redefine that scope.
