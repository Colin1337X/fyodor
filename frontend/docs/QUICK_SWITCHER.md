# Quick switcher

Updated 2026-10-07. This desktop frontend feature uses existing navigation,
window-local conversations and native resource catalog/read APIs. Backend
implementation remains separately owned.

## User workflow

The header search button or Ctrl/Cmd+K opens a native modal. Type words to find
workspaces, conversations, New conversation, Settings or Toggle sidebar. Workspace
entries come from the same `navigationGroups` export as the sidebar. Matching is
case-insensitive, Unicode NFKC-normalized and literal: every whitespace-separated
word must occur in the label, group or navigation keywords. Text never becomes a
command or regular expression. Conversation labels remain data, even if they
match action names.

Enter in the search field opens its first command result. Down focuses the first
result; Up/Down move between native buttons, Home/End select the first/last, and
Up from the first result returns to search. Enter and Space on buttons use native
activation. Escape or Close dismisses the modal. Tab uses the browser's modal
focus containment. The previous connected, enabled editor receives focus and
its selection range on dismissal; otherwise the header button receives focus.
Navigation can replace the previous editor, so a disconnected node is not focused.

Opening the switcher closes narrow navigation. It cannot open over another
modal, including Settings. IME composition and Alt/Shift-modified Ctrl/Cmd+K are
ignored. Existing shell shortcuts do not intercept keys while a modal is open.
The switcher is a navigation aid, not a text editor or generation action.

## Saved title search

Choose Saved resource titles, enter a namespace (default `workspace`) and a
nonblank query, then press Enter in search or click Search saved titles. This
explicit action searches title and URI fields through
`api.resources(namespace, cursor, "all")`. Merely typing does not query storage.
Changing query or namespace clears old matches and their continuation cursor.

This is the existing local administrative resource catalog. It does not create
Context grants, claim Context read permission, perform Context retrieval, inspect
resource bodies while searching, or search conversation messages. A namespace is
a storage selection, not an authentication boundary. No indexing service is
introduced. The catalog supplies current headers page by page; concurrent changes
mean it is not a transaction-wide snapshot.

Each search batch scans at most 25 catalog pages. Responses must contain at most
100 records and a string continuation cursor; the current native page size is
32. Scanning stops after a complete page brings matches to at least 30, or the
catalog ends. Keeping the whole last page prevents dropped matches between
batches: at most 129 results are possible with the supported response bound
(61 with current native pages). Continue searching replaces the displayed batch
with the next one; it does not accumulate an unbounded DOM. The status reports
matches, records scanned and whether continuation exists. A repeated cursor
within a batch fails visibly. No total match count is inferred from a partial
scan. Repeating a search starts from the catalog beginning.

Search text is bounded to 256 UTF-8 bytes. The input's 256-character cap is an
additional UI limit; multibyte input may reach the byte limit sooner. Namespace
names accept 1–64 ASCII letters, digits, underscores or hyphens. Resource
references require a supported typed `fyodor://` URI, bounded to 512 characters,
and a positive decimal revision string. Revisions are never converted to Number,
including values above JavaScript's exact integer range. Malformed catalog
records fail the batch rather than being presented as usable links.

## Opening and edit protection

A result queues a read of its exact namespace, URI and displayed revision in
Resources. The generic inspector rejects a queued selection while busy, while
another selection is pending or while it has unsaved edits. That rejection keeps
the modal open with an actionable message. Navigation does not autosave. Writing
and other workspace drafts retain their own existing state and dirty guards.

Resources calls the historical `api.resource(namespace, uri, revision)` reader.
It verifies returned identity/revision and rejects deleted records before
adopting the new draft. A read failure retains the prior inspector draft. Once
loaded, history and the namespace catalog refresh through the existing paths.
The historical content stays selected even if its head advanced after search.
Saving still uses that exact revision as the compare-and-swap base; a stale save
fails and retains the edited text. A search result is not authorization to
overwrite the latest resource head.

## Code organization and lifecycle

| Module | Responsibility |
| --- | --- |
| `src/icons.js` | Shared navigation groups and search icon. |
| `src/switcher-data.js` | Pure matching, bounded catalog paging and validated references. |
| `src/switcher.js` | Modal rendering, focus, keyboard behavior and asynchronous search lifetime. |
| `src/main.js` | Shell adapter for navigation, conversations and existing settings actions. |
| `src/resources.js` | Queued exact-revision inspection and existing dirty/CAS protection. |
| `src/clay.css` | Responsive modal layout using shared semantic clay tokens. |

Result rendering replaces only the result list; typing does not replace the input.
Titles, URIs and labels are escaped, indices are generated numeric attributes,
and status/error text uses `textContent`. At most 300 existing conversations are
eligible; command results display at most 60 matches in their existing order.
There is no fuzzy ranking or persisted search history.

Search disables its fields, mode controls and results while running; Close stays
available. A generation counter prevents a late response from repopulating a
closed or newly reopened switcher. Dismissal does not cancel the native request.
The controller is installed once for the window lifetime. New Settings opens
after the switcher closes so two modal dialogs are not deliberately stacked.
The resource queue is consumed by the Resources mount path, independently of
the modal's lifetime.

## Verification and limits

`node --test frontend/tests/switcher.test.mjs` covers matching, bounds, inert
titles, large revision strings, malformed references/responses, sparse paging,
cursor repetition and complete-page continuation. The owned browser/native
harness `node frontend/tests/resource-browser.mjs` covers Ctrl+K, Escape and caret
restoration, ArrowDown/native Enter, retained Writing drafts, conversation and
Settings navigation, title search, historical reads after independent head
advancement, stale save retention and dirty inspector rejection. Its Enter event
includes the character event needed to exercise Chromium native button activation.

Synthetic light, dark and 390px screenshots are in `../qa/themes/quick-switcher*.png`.
The harness checks document and modal horizontal overflow. Existing integration
checks and separate native CLI exports run in the same owned fixture workflow.
See `ARCHITECTURE.md` for setup and cleanup; user databases/profiles are not used.

This does not establish screen-reader behavior on every platform, full zoom or
forced-colors coverage, macOS Cmd behavior, Linux webview behavior, Tauri installer
validation or the brief's native C TUI command palette. Those remain separate
audits/work. Full-text resource indexing and shared persistent conversations
need separately agreed backend contracts.
