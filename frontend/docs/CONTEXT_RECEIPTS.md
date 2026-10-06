# Context receipt inspection and dataset sources

Status: 2026-10-06. This frontend milestone consumes existing native receipt
read/list services. It adds no backend route or schema. Authoritative contracts
are [Context HTTP](../../backend/CONTEXT_HTTP.md),
[receipts](../../backend/REQUEST_RECEIPTS.md) and `backend/core/fyodor_store.c`
(`fyodor_receipt_read`, `RECEIPT_VISIBLE`).

## Readable inspection

Context retains its existing grants, layers, budgets, generation and history.
Reading a receipt now shows Context used with source count, included count and
executed context byte total. Each source shows its URI, exact revision, layer,
included/original byte count and actual inclusion:

| Status | Stored evidence | Display |
| --- | --- | --- |
| Included | Positive included length equals original length | Included |
| Truncated | Positive included length is less than original | Prefix included |
| Omitted | Included length is zero, original length positive | Not included |
| Empty | Both lengths are zero | Empty source |

Text used from this source reveals the escaped executed slice. The preview shows
at most 2,000 Unicode code points; the exact prompt contains the complete included
text. Counts are stored execution values, not new estimates. The native receipt
does not record whether byte budgeting or token reservation caused truncation,
so the UI does not invent a cause.

Exact prompt, source JSON and model/sampling metadata remain inspectable. Original
JSON strings remain unchanged rather than being reserialized from parsed objects.
The source disclosure preserves `#context-sources`. Denied reads clear the executed
receipt in Context. No administrative title/content lookup enriches denied sources.

## Dataset workflow

In Datasets → Create from Fyodor sources, choose Context receipts. Its identity
defaults to the existing Context identity on first use and is explicitly editable.
Dataset namespace stays its selected namespace; the picker does not silently
change Context's namespace/identity. Browse sources lists currently accessible
receipts using native UUID pagination. Read one from history or enter its ID, then
select individual included source resources.

Saved requests collapses after reading to leave room for source choices. Omitted
or empty sources remain visible with byte counts but cannot be selected. Selected
sources show receipt ID/revision. Editing the ID field does not relabel loaded
rows: a separate loaded receipt ID pins them until another read succeeds.

Selections can mix with Writing/Explore and chats under the shared 32-source cap.
Identity changes clear Context selections and receipt rows/history; namespace
changes clear all selections. State is window-only. Corpus/SFT mappings contribute
the **included prefix**, not the entire original resource. SFT uses the explicit
shared resource prompt and chosen delimiter policy. Original user instructions
and generated output are not copied into source content.

Create dataset draft freshly reads each distinct namespace/identity/receipt tuple
through native authorization, validates the receipt and extracts selected ranges.
Multiple selections from one receipt share one read during that preparation.
Different preparations never reuse authorization. Failed reads, mappings or
bounds preserve the editor, source references and saved split copies. Success
creates a fresh unsaved dataset; Save is separate. Export/Training retain the
explicit file workflow, not a fabricated URI path or automatic job.

## Permission boundary

Native receipt reads require the original principal/namespace and current READ
on **every** recorded source, including omitted/unselected sources. Grant checks
and receipt read share one SQL statement snapshot. Revoked/deleted sources hide
later reads. Denial propagates with no administrative-resource fallback. Listing
uses the same visibility predicate; reading checks again after listing.

The local bearer authenticates the administrator; the principal UUID selects
stored local grants and is not remote identity authentication. This feature
grants nothing and changes no source. Revocation cannot recall disclosed text.
Creation-time reads and later Save are separate: this frontend does not claim
revocation-atomic dataset persistence or Training launch. Native jobs would need
their own commit-time contract.

Writing/Explore still use explicit human administrative reads. Only the Context
receipt branch uses this principal-enforced snapshot contract; provenance records
them distinctly. A grant query followed by an administrative get is never used.

## UTF-8 and integer integrity

`context-receipt.js` validates schema-1 metadata, a context boundary at most 1 MiB,
prompt bytes at most 2 MiB + 2, source/metadata JSON at most 64 KiB each, up to 64
unique sources, known layers and ordered nonoverlapping ranges. Included lengths
cannot exceed original lengths or the context prefix. UTF-8 splits and NULs fail.
Snapshot decoding preserves leading BOMs. Offsets are UTF-8 bytes, never JS UTF-16
indices, so Unicode sources cannot spill into separators or user instructions.

Native source JSON contains numeric revision literals. Current JSON.parse engines
give the reviver the primitive's original `context.source`, retained as a positive
decimal string within signed-64-bit native storage. Older engines fall back only
for safe integers and reject large revisions instead of rounding. String revisions
are accepted for compatible future adapters. Raw source JSON remains unchanged;
other consumed numeric fields are bounded safe integers. Arbitrary model metadata
is not reserialized or used to infer immutable model identity from a path/size.

## Provenance

Shared creation-time mapping/output hashes and dataset ranges are described in
[Dataset sources](DATASET_SOURCES.md). Context appends `;receipt-source-prefix`
to the mapping label. Each selected source records an origin such as:

```json
{
  "kind": "context-receipt",
  "namespace": "workspace",
  "principal": "<original principal UUID>",
  "receipt_id": "<immutable native receipt UUID>",
  "uri": "fyodor://writing/documents/<uuid>",
  "revision": "3",
  "content_sha256": "<SHA-256 of included prefix UTF-8 bytes>",
  "receipt_source": {
    "index": 0,
    "offset": 0,
    "length": 3,
    "original_length": 40,
    "layer": "session"
  },
  "first_unit": 0,
  "unit_count": 1,
  "byte_start": 0,
  "byte_length": 3
}
```

`receipt_source.offset/length` refer to the immutable executed prompt. Dataset
`byte_start/byte_length` refer to original mapped output. `content_sha256` hashes
the included prefix, not the whole original resource. Original length/revision
make truncation explicit. Later edits preserve attribution but may invalidate
creation-time ranges/hash equality. Saving creates no model receipt or permission
attestation for subsequent dataset use.

## Remaining work

This implements selected **executed Context resource snapshots**. It does not add
principal-scoped HTTP reads of arbitrary current full resources, new retrieval/
ranking/assembly-preview endpoints, universal workspace identity coordination,
durable native chats, direct dataset Training or remote authentication. Never
present a truncated prefix as a full resource or substitute an administrative get
when a full current-resource Context read is required.

Materialization, run/evaluation lineage, atomic jobs, native tokenizer validation
and large/streaming datasets remain open. Actual desktop windows, other platforms,
realistic large receipts and full reader/keyboard/zoom audits remain unverified.

## Verification and maintenance

`context-receipt.js` is a pure parser plus an injected fresh-read adapter.
`context-receipt-view.js` renders attribution. `context-workspace.js` retains
requests/state/grants. The picker owns receipt references; `dataset-sources.js`
maps authorized slices through shared bounds/hashes/draft logic.

Five new checks cover Unicode/BOM, exact large revisions, invalid/overlapping
ranges, omitted/empty sources, fresh denial, one read per receipt per preparation,
SFT without instructions/output, no administrative fallback and inert display.
The selected suite has 40 checks; run commands are in [Architecture](ARCHITECTURE.md).

The Windows browser flow generates a three-byte context prefix, checks readable
attribution, selects it in Datasets, revokes READ and observes denied preparation
with an intact prior draft. Explicit regrant allows export of exactly that prefix.
CLI export verifies receipt/principal/source revision/range provenance. The actual
download completes one native CPT update against the tiny fixture, establishing
execution interoperability rather than convergence. Light/dark/390px inspection
and picker images are in `frontend/qa/themes`.
