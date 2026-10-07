# Revision comparison

Updated 2026-10-07. Writing, Resources and Explore share a lightweight frontend
comparison of a selected historical resource revision against the current editor.
It uses historical reads that those controllers already perform; no backend
service, saved annotation or new resource format is introduced.

## Reviewing changes

Select a revision in Revision history to load the existing read-only preview.
Open Compare saved revision with editor beneath that preview. The comparison
labels its direction: saved revision first, current editor second. If the editor
is dirty, it says unsaved editor; otherwise it names the editor's revision. An old
saved preview can therefore be compared with unsaved text without pretending that
either side is the latest resource head.

Changed titles appear as Saved title and Editor title. Text is compared line by
line. Removed and Added labels, plus old/new line numbers, identify the direction
without relying on color. An en dash means that side has no corresponding line.
Three unchanged lines surround each change. Larger unchanged spans have an
explicit omitted-line count. Identical text says Text unchanged and has no
redundant line listing. Empty lines are labelled; `[CR]` exposes an actual carriage
return at the end of a line, including CRLF versus LF differences.

Comparison is read-only. It does not select text for replacement, autosave, merge,
advance revisions, retrieve extra Context or grant access. Existing Copy title
and text to editor actions remain explicit and keep their dirty guards. Existing
native compare-and-swap rules still govern Save. Reload saved version is the
existing deliberate way to replace a local editor with the current native head.

With the disclosure open, title/text input schedules a refresh after 180 ms of
quiet. Counts/save state still update immediately through the original controller.
Only the comparison body changes; the textarea, focus and selection remain intact.
A short Updating comparison status prevents an old display being mistaken for
the edited text. A closed disclosure does no comparison work. Controller repaint
recreates it closed, including after operations that refresh history. This is a
deliberate window-local review display, not a saved preference.

## Algorithm and bounds

`src/revision-comparison.js` owns domain comparison, escaped markup and disclosure
installation. Controllers pass a snapshot getter and their existing active-mount
guard. Snapshots contain saved title/text/revision and current title/text/revision/
dirty state. The shared component makes no HTTP requests and never mutates either
snapshot. Revisions remain strings, including values above Number's exact range.

1. Validate string inputs and enforce 1 MiB of UTF-8 per side and 20,000 lines per
   side. Empty text has zero lines; otherwise splitting at LF retains final empty
   lines and carriage returns. This is exact text comparison, not normalization.
2. Trim equal prefixes/suffixes before computing a longest common subsequence in
   the changed region. Bound its old-line count multiplied by its new-line count
   to 250,000 cells. The row/column margins add bounded table overhead. The table
   uses a typed integer array; no editor dependency is needed.
3. Reconstruct Unchanged/Removed/Added operations. Ties remove first so repeated
   lines have a deterministic result. Operations reconstruct both source texts
   exactly and have monotonic one-based old/new line numbers. Matching chooses
   one valid minimal line-edit sequence; repeated prose can have other equally
   valid alignments. It is not a semantic or word-level comparison.
4. Compute exact added/removed counts from all operations, then select display
   context. If the display would exceed 1,000 rows including omission separators,
   retain the counts but omit the detailed listing with an explicit explanation.

If input or changed-region limits are exceeded, the UI reports detailed
comparison unavailable and directs the reader to the existing full saved preview
and editor. It does not fabricate edit counts, silently truncate a comparison,
or substitute a cheaper inaccurate diff. Title comparison can still be shown.
Long unchanged edges allow a small edit in a long document to remain inexpensive.
Large unrelated documents can exceed the changed-region budget even when their
byte/line limits fit. Those limits are part of this frontend feature, not claims
about native resource storage limits.

Text, titles and labels are escaped before markup insertion; raw markup never
becomes an element. Line listings use shared semantic clay colors, bounded scroll
height and wrapping for long text. On narrow screens each line's text moves below
its line numbers and change label. The list is read-only native content and the
disclosure remains keyboard-operable without custom focus trapping.

## Mount and asynchronous lifetime

`installRevisionComparison` installs only when both a historical preview and an
editor exist. Resources/Writing call it after their normal input listeners;
Explore calls it after its editor wiring. State therefore updates before the
comparison's bubbled input event schedules a refresh. It attaches listeners to
the disclosure and editor nodes belonging to that paint, not the persistent
workspace root, so repaint does not accumulate listeners.

A pending debounce checks both DOM connection and the controller's mount/view
guard before rendering. Switching workspaces, replacing the preview or repainting
the controller cannot cause that callback to write into a new screen. Toggling
the disclosure clears its pending timer. Old detached nodes are not retained by
global registries; a pending timer releases its references when it finishes.

## Verification

Run `node --test frontend/tests/revision-comparison.test.mjs` for exact operation
reconstruction, repeated-line determinism, blank/final-newline/CRLF/Unicode cases,
long unchanged edges, source line numbers, all computation/display limits and
escaped title/text/revision presentation. Run the full selected test command in
`ARCHITECTURE.md` and `npm.cmd run build --prefix frontend` for integration.

`node frontend/tests/resource-browser.mjs` runs the existing owned browser/native
workflow. Resources and Explore open real revision 1 previews and verify that
comparison leaves current text intact. The Writing helper selects an already
saved synthetic character, compares revision 1 with unsaved multiline text and a
changed title, verifies counts and inert script-shaped prose, checks caret/focus
after native text insertion and debounce, verifies dirty-copy rejection and
navigates away while a refresh is queued, then explicitly reloads the exact
original title/text/revision. It does not save those comparison edits. Separate
native CLI exports in the same harness still verify stored resources and previous
dataset/evaluation/training workflows.

Light/dark/390px evidence is in `../qa/themes/revision-comparison*.png`; the harness
checks document/workspace overflow. Full screen-reader, zoom, forced-colors,
cross-platform Tauri window and installer verification remain open. This feature
does not implement conflict merging, cross-resource comparison, word diff or
generation-result acceptance; each has separate semantics and tests.
