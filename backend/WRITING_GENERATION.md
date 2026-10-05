# Native Writing generation previews

Writing exposes Generate, Rewrite and Continue for saved documents, notes,
characters and project text. Select a loaded model in the existing top bar,
enter instructions and budgets, optionally select additional saved resources,
then request a preview. Unsaved editor changes must be saved first.

## Shared C service

`fyodor_writing_generate` inserts the target as the first explicit source.
Additional entries retain the common layer ordering and permission policy.
The target's observed revision is compared with its authorized assembly
snapshot before inference; stale targets return conflict without generating or
creating a receipt. All sources require READ, and retrieved entries additionally
require SEARCH. At most 63 extra resources are accepted. Duplicating the target
in extra context is rejected. Deleted/inaccessible targets return denial.

Writing now automatically includes readable parent-project text as workspace
context, nearest parent first. Target metadata, parent traversal, source grants,
revision checks and content reads share one snapshot. A denied parent stops
traversal; neither its text nor its ancestors are exposed through discovery.
Parents grant no access to children, and sibling resources are not enumerated.
Manually selected sources reserve slots and retain their chosen layer; discovered
duplicates are omitted. Discovery stops at 64 total sources or 64 ancestors.
The existing shared byte/token budget still governs the resulting prompt.

This provider runs in the shared C service for Desktop, HTTP and CLI/TUI Writing
generation. No new request field is required. To allow a project, explicitly
grant READ through Context, CLI, or the Writing context selector. Unselecting
that project afterward does not revoke its grant; it remains eligible for
automatic discovery. Receipts record included projects, original revisions,
workspace layers and byte ranges, including zero-byte budget omissions.

Action prefixes are `Write:`, `Rewrite:` or `Continue:`, followed by a newline
and the user's instructions. Native context generation constructs the exact
context-plus-instructions prompt, applies UTF-8 byte limits and tokenizer
reservation, performs inference and saves its immutable receipt. The receipt
contains the executed action/instructions in the prompt, source identities,
revisions, included byte ranges, sampling and output. Generation never modifies
the target, its history or its grants. Greedy sampling is used by both clients.
Model instruction-following and writing quality depend on the loaded model.

## Local HTTP and CLI/TUI

`POST /api/v1/context/writing` shares the local administrator bearer/origin and
loopback-only restrictions of the other context routes. It accepts the normal
generation fields plus `target` URI, `expected_revision` decimal string and
`mode` (`generate`, `rewrite`, `continue`). `resources` and optional parallel
`layers` describe extra context, not the target. Success returns preview text,
token counts, context bytes and receipt ID. Permission/revision validation occurs
in C, not only in the desktop UI. This is not a remote identity mechanism.

```text
fyodor --store PATH --namespace NAME context write PRINCIPAL MODEL TARGET REVISION MODE BYTE_BUDGET MAX_TOKENS INSTRUCTIONS [URI...]
```

Extra CLI sources may use `LAYER=URI`, as in ordinary context generation. The
same command works through the TUI dispatcher. It emits a preview and receipt
ID; use `context receipt PRINCIPAL RECEIPT_ID` to inspect the saved request.

Accept edited or unchanged output through the shared native save service:

```text
fyodor --store PATH --namespace NAME context accept PRINCIPAL TARGET REVISION RECEIPT_ID TITLE CONTENT
```

HTTP uses `/api/v1/resources/update` with its usual fields plus `receipt_id`
and `principal`. This remains a trusted local administrator edit, not a
principal-scoped WRITE operation. It grants no permissions.

## Desktop behavior

Writing uses the same remembered context identity as the Context workspace.
The Set read-only grants button explicitly replaces grants for the target and
selected additional sources; selecting them alone grants nothing. Additional
Writing sources use the workspace layer, in selection order. The generic CLI
and HTTP service can also reference other permitted kinds such as world lore.
The desktop [lore-reference picker](WRITING_LORE.md) now persists shared lore
links. Permitted links on the target and ancestor projects enter workspace
context without manual reselection. Linking never grants access.

Generated output is escaped plain text. Generate and Rewrite offer replacement
of editor text; Continue offers exact appending. Neither applies automatically.
Applying a preview creates a dirty draft. A draft changed after generation or
loaded from a different target/revision cannot accept that preview. Saving is
revision-checked and links the accepted receipt to the saved resource, so
concurrent edits cannot be silently overwritten. Further draft edits retain
the pending receipt link. A rejected save retains the draft and link.

## Accepted-edit provenance

`fyodor_writing_save` validates the stored receipt's owner and namespace,
current READ access to every live source, Writing mode, and first source's
target identity and revision. The target head must still match that revision.
These checks, the content update, search index and immutable history revision
commit in one writer transaction. Any failure rolls back the entire save.

The service preserves metadata and existing provenance, appending an object to
`provenance.writing_generations` with `receipt_id`, decimal-string
`source_revision` and `saved_revision`, `mode`, and boolean `text_edited`.
Generate/Rewrite compare saved text with receipt output; Continue compares it
with the prior full document followed by that output. Title changes do not
affect this text flag. Later manual saves preserve these references.

A pre-existing non-array `writing_generations` field causes a conflict instead
of being overwritten; the normal 64 KiB provenance limit still applies.
Generic receipts and older receipts without structured `writing_mode` cannot
be accepted through this service, though they remain readable under the usual
policy. Imported provenance is untrusted descriptive data, not an execution
attestation or permission grant; validation uses the actual stored receipt.
There is no database schema change.

Richer character/lore tools, sibling retrieval and streaming or
cancellation remain open. Persisted links can be inspected in exported resource
packages; a desktop viewer for those links remains unfinished. No completed
Writing acceptance claim is made.

## Evidence

The native test exercises all three action prompts with the trained tiny CPU
model, verifies nonzero generation and durable receipts, proves the target is
unchanged, and rejects missing grants, stale revisions, duplicate sources,
invalid modes, too many sources and deleted targets. HTTP tests repeat all
three actions and compare a native CLI-produced receipt. Browser QA exercises
permission denial, explicit grants, additional context, all three previews,
unchanged editor text, accepting Continue, saving a second revision and reload.
These are correctness fixtures, not writing-quality measurements.

Accepted-save tests also cover wrong owners/namespaces, revocation, stale
revisions, ineligible receipts, injected history-write rollback, exact large
metadata integers, edited output and preservation of earlier receipt links.
HTTP and CLI acceptance interoperate. After browser acceptance and reload, an
independent native CLI export verifies the saved Continue receipt reference.

Project-provider checks cover nearest-parent ordering, a denied intermediate
parent, namespace isolation, revocation, stale target moves, manual duplicate
precedence, retrieved-source permissions and capacity/zero-byte budgets. Native
generation and HTTP/CLI interchange verify exact project text in the executed
prompt. Browser QA grants a project, unselects it, generates, and checks that
the receipt still contains it as workspace context. Revocation hides prior
receipts and excludes that project from subsequent automatic context.
Five affected CTest entries pass under C17, C23 and ASan/UBSan for this provider;
the real browser workflow and frontend production build pass.

Seven selected C17 application/CLI/TUI tests pass. Five affected tests each pass
under C23 and ASan/UBSan. Real browser QA, all 16 frontend unit tests and the
production build pass.
