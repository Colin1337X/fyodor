# Frontend architecture and maintenance guide

Status: 2026-10-06. This describes the implemented vanilla JavaScript frontend,
not the entire product requested in `prompt.md`. Backend work is owned separately;
frontend changes should consume documented contracts rather than add native APIs.

## Entry points and ownership

`index.html` loads local theme definitions and `public/appearance.js` before the
main stylesheet paints. Vite bundles `src/main.js`; there is no frontend framework,
client router, remote font dependency or server-rendered page.

| Module | Responsibility | Maintenance boundary |
| --- | --- | --- |
| `src/main.js` | Application shell, navigation, selected model, chat, playground, training, logs, settings | Existing large module. Extract one view at a time; avoid adding new feature controllers here. |
| `src/icons.js` | Static SVG icons and navigation groups | No resource or model text may enter SVG markup. |
| `src/workspace.js` | Overview, Models and API view markup | Uses actual runtime/model capabilities. |
| `src/resources.js` | Resources and Writing editors, folders, revisions, generation previews | Shared editor controller; metadata mutations go through native endpoints. |
| `src/writing-lore.js` | Rendering linked lore and link controls | Rendering only; resource controller owns requests and state. |
| `src/writing-studio.js` | Editor-first Writing composition, counts and explicit session guidance | Moves existing nodes; resource controller retains write/context logic. |
| `src/writing-files.js` | Strict UTF-8 text import and current-draft text downloads | Separate from resource package JSON; never round-trips loaded metadata. |
| `src/explore-data.js` | Typed world/lore identity, draft creation, bounded catalog filtering | Pure helpers except injected API access; no DOM or gameplay state. |
| `src/explore-workspace.js` | World/lore authoring controller and view | Existing resource APIs only; see `EXPLORE.md`. |
| `src/dataset-data.js` | Dataset format inspection, deterministic transforms, annotation and paging | Pure helpers; structural checks do not replace native tokenizer validation. |
| `src/dataset-workspace.js` | Dataset editing, imports, derived drafts, split persistence and export | Existing resource APIs; file handoff to Training, documented in `DATASET_STUDIO.md`. |
| `src/dataset-sources.js`, `src/dataset-source-picker.js` | Source snapshots, explicit mappings/provenance and source selection controls | Parent owns busy state/editor replacement; see `DATASET_SOURCES.md` for administrative/Context boundary. |
| `src/resource-text.js` | Complete-code-point UTF-8 title bounds | Shared by Writing imports and new/derived dataset titles. |
| `src/context-workspace.js` | Source selection, explicit grants, search, generation receipts | Selected principal is an access label, not remote authentication. |
| `src/context-receipt.js`, `src/context-receipt-view.js` | Exact receipt ranges/revisions and readable attribution | Native receipt authorization; no administrative fallback. See `CONTEXT_RECEIPTS.md`. |
| `src/evaluation-data.js`, `src/evaluation-workspace.js` | Saved quality definitions/results, bounded checks, sequential local generation and human review | Existing generic store and generation; observed request timing is not native benchmark/loss. See `EVALUATIONS.md`. |
| `src/evaluation-datasets.js`, `src/evaluation-dataset-picker.js`, `src/evaluation-editor.js` | Reusable test collections, strict file interchange, exact source revisions and explicit mappings; shared paged case markup | Parent owns writes/editor replacement; no implicit correctness or trainer-format reinterpretation. See `EVALUATION_DATASETS.md`. |
| `src/api.js` | Authenticated local HTTP and compatible external endpoint requests | Keep protocol details here, never in rendering templates. |
| `src/platform.js` | Tauri connection, native pickers and desktop commands | Browser mode must report unavailable desktop-only operations. |
| `src/stream.js` | Bounded remote SSE decoding | Preserve cancellation and split UTF-8 behavior. |
| `src/message.js`, `src/syntax.js` | Escaping, text/code rendering, lexical highlighting | Untrusted text must remain inert. |
| `src/revision-comparison.js` | Bounded exact line comparison and lazy disclosure shared by authoring views | Read-only snapshots; no writes, grants or editor replacement. See `REVISION_COMPARISON.md`. |
| `src/model-inspection-data.js`, `src/model-inspector.js` | Loaded-file/native shape observation and explicit raw tokenization/download | Runtime IDs are process observations; no persistent identity or annotations. See `MODEL_INSPECTION.md`. |
| `src/session.js` | Conversation validation and generation controls | Imports and persisted storage are untrusted. |
| `src/benchmark.js`, `src/training-metrics.js` | Parse measured telemetry and render results | Missing measurements must not become invented zeros/results. |
| `public/themes.js`, `src/themes.css` | Named palette data and semantic CSS roles | Generated by `scripts/generate-themes.py`; sources and attribution in `themes/`. |
| `public/appearance.js`, `src/appearance-controls.js` | Bounded preferences and settings UI | Appearance storage is separate from sessions and credentials. |
| `src/styles.css`, `src/clay.css` | Layout/component base and shared clay material | New screens reuse tokens and components, not hardcoded palette colors. |

## Navigation and rendering

The active view is `s.view` in `main.js`, restored only if it is a known view.
`render()` sets `body.dataset.view`, updates navigation and the header/footer,
then mounts the view inside `#workspace`. New views need an allowed persisted
key, `meta` entry, render dispatch and navigation entry.

Use/Create/Run/Serve navigation has Overview, Chat, Models, Writing, Explore,
Resources, Datasets, Context, Playground, Training, Evaluations, Benchmarks, Logs and API access.
The top model selection is shared. Explore authoring does not use that selection.
Writing/Context use local native generation, even when Chat's provider is remote.

Resources, Writing, Context, Explore, Datasets and Evaluations own module-level state. Their mount
generation and active-view checks prevent an old asynchronous completion from
painting into a different screen. Busy operations disable their controls. They
can finish while another view is open; their state remains available on return.
Do not assume a mounted DOM node still exists after an awaited request.

Views currently repaint their templates, then attach listeners. Keep editable
values in state on every input event before any repaint. Preserve native
selection/focus when future changes introduce continuous updates. Chat streaming,
training telemetry and logs already update selected nodes rather than rebuild
their entire form on each token/log line.

## State and persistence

| State | Lifetime/storage | Notes |
| --- | --- | --- |
| View, model selection, chats, generation/provider/training settings | `fyodor-state` localStorage | Known primitive settings are validated on restoration. |
| Theme, density, fonts, radius, colors, spacing, motion, sidebar | `fyodor-appearance` localStorage | Invalid/unavailable storage falls back safely. |
| Context principal UUID | `fyodor-context-identity` localStorage | Explicit grant identity; not a login or network credential. |
| Resource namespace, drafts, current pages, previews | Current JS window memory | Unsaved resource drafts are not a durable autosave. |
| Saved resources, revisions, grants, receipts | Native store | Frontend never becomes the canonical database. |
| Local bearer token | Connection returned by Tauri, cached in memory | Browser development uses Vite environment configuration. Do not print it. |
| Remote API key | Memory unless user selects Remember | Remember writes the key into session localStorage. |

Explore, Resources, Writing, Datasets and Evaluations warn on window unload with a dirty draft where
the host supports the standard browser warning. Explore also blocks library,
namespace and resource changes that would discard that draft. Switching application
views retains its draft in memory. Other editors have their own dirty guards;
there is not yet a common application-wide durable draft/autosave service.

## Resource write rules

1. New resources have a typed UUID URI, package schema 1, empty metadata/provenance,
   and expected revision `"0"`.
2. Existing title/text edits use `/resources/update` with the loaded revision.
   Do not stringify the entire loaded package for an ordinary edit: JS can lose
   precision in arbitrary numeric metadata.
3. Revisions remain decimal strings. Never increment them with JS arithmetic.
4. Writing moves and lore links use their dedicated CAS endpoints. These preserve
   other metadata/provenance server-side and do not create grants.
5. On conflict retain the local draft. Reload saved version explicitly replaces
   it; never automatically retry a stale write against a newer revision.
6. History previews copy title/text into a draft only. Saving makes a new revision;
   it does not overwrite the old revision or import historical provenance.
7. Export retrieves raw package JSON. Import sends the original validated JSON
   text rather than a JS object round trip.
8. Library browsing is local administrative access. Model reads still need
   explicit native permissions. Visible content is not proof of model access.

Resource packages have 1 MiB content, 64 KiB metadata and provenance limits and
an 8 MiB package limit. The server validates UTF-8, JSON and parent/reference
relationships. Character-count UI constraints do not replace byte validation.

## Claymorphic design system

The layer order is base layout in `styles.css`, named palette semantic roles,
then `clay.css` imported by `main.js`. Clay uses theme-derived edge/light/shade
colors and shared `--clay-lift`, `--clay-control`, `--clay-inset` shadows. Panels
are raised, fields recessed and active controls pressed. Default radius is 18px;
user radius/spacing settings override the default.

Use existing `.panel`, `.resource-toolbar`, `.resource-columns`, `.resource-list`,
`.resource-editor` and `.resource-history` primitives when building authoring
views. Feature-specific styles should only cover the missing layout/interaction.
Keep labels, text, border and accent roles semantic. Do not add a second shadow
palette for each screen or force white highlights into dark themes.

Shadows can be disabled; borders can be hidden; OS/user reduced-motion choices
are respected. Forced-colors removes decorative shadows and retains panel
borders. Primary palette text contrast has unit coverage. This does not establish
contrast for every combination of user-provided custom colors.

## Accessibility and untrusted content

The [quick switcher](QUICK_SWITCHER.md) shares sidebar navigation definitions,
uses a native modal and separates pure matching/catalog paging from shell actions.
Explicit saved-title search queues guarded exact-revision Resources inspection;
it does not create Context permissions. Its documentation records focus lifetime,
search bounds, continuation, concurrency and remaining platform audits.

- Native buttons, inputs, selects, labels, details and dialogs carry interactions.
- Active navigation exposes `aria-current`; Explore selected library cards expose
  `aria-pressed`. Status/errors use a status region; Explore adds live announcement.
- Narrow navigation uses `inert` to remove the closed overlay/background from
  interaction. Escape and sidebar keyboard shortcuts live in the shell.
- Resource titles, content, URIs and errors are escaped before template insertion.
  Textarea content is escaped too, preventing a closing tag from becoming markup.
- Model code is escaped before syntax highlighting. Unknown/large code blocks use
  safe plain text. Imported JSON does not become arbitrary HTML attributes.
- Status is not communicated exclusively through color; errors contain text.

Browser checks cover 390px layouts and horizontal overflow, but comprehensive
screen-reader, keyboard order, zoom and high-contrast manual audits remain open.
Future repaint-heavy changes should explicitly test focus retention.

## Validation and working practices

Run commands from project root:

```powershell
npm.cmd run build --prefix frontend
node --test frontend/tests/appearance.test.mjs frontend/tests/syntax.test.mjs frontend/tests/ui.test.mjs frontend/tests/sidecar.test.mjs frontend/tests/explore.test.mjs frontend/tests/writing-studio.test.mjs frontend/tests/writing-files.test.mjs frontend/tests/dataset.test.mjs frontend/tests/dataset-sources.test.mjs frontend/tests/context-receipt.test.mjs frontend/tests/evaluation.test.mjs frontend/tests/evaluation-datasets.test.mjs frontend/tests/switcher.test.mjs frontend/tests/revision-comparison.test.mjs frontend/tests/model-inspection.test.mjs
node frontend/tests/resource-browser.mjs
```

The browser test requires installed frontend dependencies, Node with global
WebSocket, Windows Edge, `build-cpu/fyodor-backend.exe`, `build-cpu/fyodor.exe`
plus `build-cpu/fyodor-train.exe` and the trained tiny fixture at
`build-cpu/backend/test_models/pretraining.trained.gguf`. It owns a temporary
database, two model copies, private headless browser profile and Vite instance at
127.0.0.1:5179. It cleans those owned processes/files; it does not use the user's
browser profile or production database. The browser connects as localhost for
the allowed origin. A port collision is a test setup failure, not permission to
terminate an unrelated service.

Screenshots and theme evidence go to `frontend/qa/themes`. Separate CLI export
checks prove saved UI data is visible outside the browser. Keep fixture token
budgets small: its context length is 32, not a normal production model capacity.

Keep rationale in source comments and docs. Product screens should explain an
action or a meaningful limitation, not show implementation notes or test claims.
Add tests for behavior boundaries such as stale writes, paging and inert content;
avoid snapshot tests that merely duplicate every template string.

## Known maintenance debt

`main.js` and `resources.js` still contain dense legacy controllers/templates.
Extract controllers and rendering progressively while preserving existing tests.
Explore separates domain/paging helpers from its controller as a starting point.
There is no shared typed error object: `api.js` currently converts response errors
to `Error(message)`, losing status/code for view-specific recovery. A future
error abstraction must preserve current human-readable messages.

Namespace/principal coordination is view-local rather than one global workspace
session. Do not silently change another view's namespace or grant identity. New
shared behavior needs an explicit state migration and clear user interaction.
