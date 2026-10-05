# Writing UX: Snep reference and Fyodor adaptation

Updated 2026-10-06 following the user's request to inspect
`C:/Users/Colin/snep`. Reference files consulted were its `README.md`,
`index.html`, `styles.css`, relevant portions of `main.js` and license. The
reference calls itself Cherenkov: a framework-free story workspace. It was
inspected read-only. Its user configuration/data file was not needed or read.
No bundled font, image, script or stylesheet was copied into Fyodor.

## Reference UX

Snep treats prose as the primary surface: a large scene editor sits on the left,
with story paths and creative controls on the right. Its header shows the current
path, word/character count and save state. A terminal performs navigation and
story commands. Creative controls supply generation instructions, core hook,
style, world rules and character arcs; parameter multipliers and context scope
have documented effects rather than purely decorative sliders.

Its story hierarchy is `/folder/book/chapter/scene`. Creative values inherit from
global defaults through the hierarchy, with local overrides. Containers have
read-only descendant batch views. It has trackers, recursive/missing/stale
generation, cached request configuration, streamed text/reasoning and token-rate
indicators. These rely on its own data/runtime model and are not automatically
available from Fyodor's current native Writing API.

## Implemented adaptation

Fyodor Writing now puts the text editor in the wide left column and library,
Creative controls, linked lore and folder placement in the right sidebar.
The same native resource controller continues to own requests, grant checks,
revision guards and accepted generation receipts. Composition lives separately
in `src/writing-studio.js`; it moves existing nodes rather than duplicating that
controller into a second editor implementation.

The text surface is taller with readable line height and padding, using Fyodor's
chosen UI font and semantic palette. It remains plain text/Markdown source
editing. Save/discard/export actions stay above the text surface. Library creation,
import and namespace controls sit in a compact Create & import disclosure;
folder browsing and New document remain visible. The catalog has a bounded scroll
area so it does not consume the entire tools column.
A Focus on text button hides tools/history and centers the editor;
Show writing tools restores them. Draft text and generation state survive this
layout change. On narrow screens the editor precedes the stacked tool panels.

Live counts update without repainting the textarea or losing the caret.
Words are whitespace-separated groups; characters are Unicode code points.
This is a simple prose count, not language-specific segmentation or a model-token
estimate. Save state updates immediately when the title or content changes.
Saving is still explicit; there is no autosave claim.

## Markdown and plain-text interchange

Writing's Create & import disclosure accepts `.md`, `.markdown` and `.txt`
files. Each file is decoded as strict UTF-8, allowing a UTF-8 BOM. Files over
1 MiB, invalid byte sequences, null characters and other file extensions are
rejected before the editor is replaced. The imported document title comes from
the basename without its extension and is bounded to the native 1024-byte title
limit without splitting a Unicode code point.

Import creates a new document draft with a fresh typed UUID. It does not update
the open resource, import permissions/provenance or write the database immediately.
In a selected project/folder, the draft receives that existing native parent
reference; All Writing or Unfiled creates an unfiled document. Save creates the
first persistent revision. A dirty editor blocks another import. Invalid files
leave the current draft/resource intact.

Export draft offers Markdown and plain text downloads of the **current editor
state**, including unsaved content. File names retain readable Unicode, replace
filesystem separators/control characters and avoid Windows device basenames.
These files contain text only: resource identity, folders, lore, grants and
provenance are not included. Export package remains the distinct saved-resource
JSON format and requires a clean editor.

Import preserves source line endings in draft state. A browser textarea displays
normalized newlines; once a user edits that textarea, its value uses the browser's
normalization. The initial untouched draft export/save preserves the imported
CRLF bytes after stripping a BOM. Export emits UTF-8 without a BOM. There is no
conversion from rich text, PDF, Word or a foreign project format.

Ctrl/Cmd+S saves while focus is inside the Writing editor and suppresses the
browser's Save page action there. The visible Save button documents the shortcut
in its tooltip. Reload saved version explicitly replaces the editor with the
latest live title/text and revision. A stale write keeps local text; it does not
automatically retry or merge. If the native read fails, the local draft remains.
If the read succeeds but the following history/list refresh fails, the loaded
saved version is clean and can be edited again.

Resources and Writing request the standard browser unload warning when either
has a dirty draft. This is a last-chance warning where the host supports it,
not durable draft persistence or confirmed native window-close behavior. Import
controls support keyboard Enter/Space as well as pointer activation.

## Creative guidance and context

Creative controls retain native generation instructions, maximum new tokens and
context byte budget. Optional **Story guidance (this session)** adds Core hook,
Style guide, World rules and Characters and arcs. Each field allows up to 512
characters. They are plain user-authored text appended to generation instructions
in that order, separated by blank lines; empty fields add nothing. They are not
resource references, permission grants or instructions that bypass the native
context budget/permissions. The resulting prompt is recorded in the executed
receipt, subject to the normal native bounds.

Guidance is keyed by namespace and resource URI in the current window's memory.
It survives rerenders and resource switching, but not a window reload. It is
neither saved metadata nor inherited configuration. The label communicates that
lifetime. Other existing instruction/token/budget controls remain controller-wide
session settings. Future persistent controls need an agreed native schema.

The existing additional-source selection and dedicated lore links remain separate
from prose guidance. Model access is explicit. Project context and linked lore
are assembled in the native snapshot provider, not by dumping the workspace into
a browser-created prompt. Generated text remains a preview until applied; Save
records a new revision and validated receipt provenance.

Clay material is shared with the rest of Fyodor, rather than adopting Snep's
black/glowing palette. Named themes, customized radius/fonts/spacing and shadow
preferences still apply. The reference's editor-first behavior guides the UX;
Fyodor's design system remains authoritative.

## Reference features awaiting contracts

| Snep pattern | Fyodor now | Required before full adaptation |
| --- | --- | --- |
| Folder/book/chapter/scene path tree | Typed projects/folders and Writing documents, flat folder selection | Agreed container types/order/path resolution and hierarchy listing. Do not fake paths as resource identity. |
| Inherited creative overrides | Session guidance on the opened resource; native ancestor text context | Reserved versioned metadata, per-key inheritance/reset, CAS update and permission-aware resolution. |
| Length/chaos/tension multipliers | Explicit bounded output tokens; native Writing sampling currently fixed | Native configurable sampling/options contract and receipt fields. No slider claims to change fixed sampling. |
| Pacing/dialogue prompt controls | Authors can express these in instructions/style | Persisted guidance schema if dedicated inherited controls are added. |
| Workspace/book/previous-scene context | Explicit additional sources plus authorized project/lore provider | Permission-aware ordered scope retrieval, provenance and budget semantics. |
| Batch descendant view | Current project edits its own text | Ordered descendant reads and read-only composition; never save combined text into one child. |
| Named scoped trackers | Not implemented | Typed tracker definitions/values/domain and revision semantics. |
| Recursive/missing/stale generation | Generate/rewrite/continue one resource | Job orchestration, stale criteria, cancellation, atomic acceptance and per-scene receipts. |
| Shell story commands | Native CLI/TUI exists; no Writing terminal embedded | Shared command semantics/path mapping and error/output protocol before adding a terminal. |
| Streamed prose/reasoning and token rate | Completed native previews; remote Chat streaming separate | Native Writing stream/cancel contract, measured timing/usage and reasoning policy. |

Backend ownership stays separate. Until these contracts exist, do not store
unvalidated inherited settings in arbitrary metadata or build synthetic state
that looks persistent. Existing Writing operations must remain usable while
new contracts are integrated incrementally.

## Verification

Unit checks cover Unicode counts, whitespace handling, namespace-separated
guidance keys, explicit prompt composition and escaped guidance markup.
File checks cover exact UTF-8/BOM/newline handling, empty files, invalid encoding,
NULs, size/title bounds and portable export names.
The headless browser workflow checks editor/sidebar ordering, live counts,
focus-mode preservation and guidance across rerenders. The existing native
generation/accepted-receipt, folders, lore and reload checks run through the new
composition. Light/dark/390px screenshots are in `../qa/themes/clay-writing*.png`.
The file workflow imports Markdown into a selected folder, blocks replacement of
a dirty draft, reads actual `.md`/`.txt` downloads to verify their exact contents,
saves using the keyboard shortcut, injects a second writer and checks conflict
preservation/explicit reload. Separate native CLI export verifies the final text
and project reference.

Full Snep feature parity, native stream/cancel, terminal commands, inherited
settings, desktop window/installer behavior and a complete accessibility audit
remain open. See `ROADMAP.md` for the wider frontend scope.
