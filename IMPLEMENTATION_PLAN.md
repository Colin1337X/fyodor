# Unified Fyodor implementation

## Audit checkpoint — 2026-09-29

Starting revision: `34921f4`. The only pre-existing working-tree change was the
untracked user brief `prompt.md`; preserve it. This document tracks implementation,
not a declaration that the requested product exists.

### Existing and reusable

- Root and standalone backend CMake builds use C17 by default, with C23, strict
  warnings and optional ASan/UBSan. `nya-engine` is separate from `nya-server`.
- Native model inspection, tokenization, generation, CPU autograd, pretraining,
  CPT/SFT/DPO, accumulation, checkpoint resume and held-out evaluation exist.
- UTF-8 file helpers preserve exclusive output creation on Windows and POSIX.
- CPU, CUDA, Vulkan and detachable ROCm/MLX adapters exist. Hardware detection,
  unavailable-hardware skips and removal checks must remain intact.
- Tauri supervises the server and trainer. Vanilla JS provides chat, playground,
  model loading, training, imported benchmarks, logs and API configuration.
  Appearance is applied before first paint and stored separately from sessions.
- Native/OpenAI/Anthropic HTTP APIs are authenticated. Local generation returns
  complete responses. Remote OpenAI SSE support is not native node streaming.

### Gaps and constraints

- No shared durable application-resource registry or typed resource identity.
  Model runtime IDs and browser conversation IDs are not a shared resource store.
- No unified context permission/retrieval/assembly service, persisted Explore,
  Writing, Dataset Studio, evaluation registry, agent runs or workflow service.
- No dedicated pure-C `fyodor` client or `fyodor-tui`; existing server, trainer
  and benchmark executables remain useful and must retain compatibility.
- No authenticated WSS node pairing/revocation/multiplexing implementation.
- Public training remains CPU-only. Resident CUDA matrices, scratch lifetimes,
  finite checks, activations, RMSNorm and embeddings are component foundations;
  they do not establish a full GPU training graph, optimizer or convergence.
- UI palettes already exist but do not yet meet the common Desktop/TUI semantic
  theme and customization requirements. Mobile builds remain unverified.

### Evidence discipline

The benchmark index and current training reports establish important boundaries:
historical timing is not a new baseline; GPU component tests are not complete
training; generated bundles and extracted installers are not interactive QA.
The embedding composition exposed a real accumulation-order defect, repaired
without relaxing tolerance. Preserve that regression and all excluded evidence.
Continue reviewing the older detailed reports before touching their subsystems.
No performance claim or unrelated process termination is needed for foundations.

### Dependency order and executable checkpoints

1. **Identity:** independent C application library; canonical typed resource URI
   construction/parsing with bounded input, explicit errors and round-trip tests.
2. **Persistence:** versioned resource records, durable atomic changes, migration,
   concurrent-client semantics, metadata/provenance, safe import/export tests.
3. **Context:** explicit namespace and principal authorization, retrieval, ranking,
   bounded assembly and saved provenance; then actual generation integration.
4. **Native clients:** shared C command/application service, CLI stdout/exit/JSON
   contracts; TUI renderer and terminal lifecycle over that same service.
5. **Nodes:** truthful capabilities, established TLS, explicit pairing and revocation,
   bounded WSS framing, streaming/cancellation/reconnect; shared remote client.
6. **Workspaces:** Writing and Explore resource lifecycles, Dataset Studio using the
   native trainer's accepted formats, persisted evaluation and model ancestry.
7. **Integration:** Desktop services and themes, agent permissions, workflows,
   dataset/training/evaluation loops, packaging and end-to-end hardening.

Keep engine and presentation independent from the application library. URI
identity is not a filesystem path or authorization grant. Namespaces and access
decisions belong to the application service, never inferred from possession of
a URI. Reuse existing engine calls and trainer semantics instead of duplicating
inference/training or wrapping subprocesses as the new native client core.

### Regression gates

For each checkpoint: strict C17 compilation, focused behavioral tests and the
applicable CPU suite. Before completion: CPU and available accelerated suites,
C23, supported sanitizers, optional-provider removal, frontend tests/build,
Tauri package verification and real cross-interface workflows. Report exact
results and platform/hardware skips; do not borrow previous reports' counts.

## Progress

- Repository and current evidence audit started; implementation is incomplete.
- Implemented `fyodor-app`, a pure-C static application library with no engine,
  HTTP, UI or optional-provider dependency. Its first public interface is
  `backend/include/fyodor_resource.h`.
- Implemented OS-random UUIDv4 generation, canonical URI parsing/formatting and
  typed equality for 14 resource kinds. Nested lore/save/run identities require
  a parent; other identities forbid one. Parsing is length-bounded and rejects
  nil IDs, noncanonical encodings, trailing data and unsupported routes. Failure
  preserves caller outputs. Existing engine IDs and browser data are unchanged.
- Windows GCC C17 strict build passed. Complete CPU CTest: **52 passed, 0 failed,
  4 hardware skips** (ROCm/MLX). LLVM-MinGW ASan/UBSan focused identity test:
  **1 passed, 0 failed**. Both root and standalone backend builds were exercised.
  GCC C23 also built and passed the focused test (**1 passed, 0 failed**) in the
  existing CUDA-enabled configuration; this identity test performs no GPU work.
  CTest logs remain in the respective build directories' `Testing/Temporary`.
- Initial command setup needed two corrections: use the actual bundled CMake
  executable rather than its Python launcher; configure the existing sanitizer
  build from `backend`, matching its cache. Neither attempt reached compilation.
  The C23 build needed explicit regeneration to discover the new target.
- This checkpoint does not implement persistence, authorization, a CLI/TUI or
  any complete product workspace. No throughput measurement, accelerated test,
  platform other than Windows, or process termination is claimed.
- Implemented schema-1 durable storage in `fyodor-app`: revision-checked writes,
  complete atomic history, soft deletion, same-namespace live parent checks,
  bounded UTF-8/JSON records and unknown/foreign/modified-schema rejection.
- Vendored hash-verified SQLite 3.53.4, pure C/public domain, statically linked
  only for application storage. No runtime language or server dependency added.
  Rationale, source hashes and licensing are in `backend/vendor/sqlite/README.md`.
- Storage checkpoint: full CPU suite **54 passed, 4 hardware skips**; final
  focused application tests **3 passed**, also **3 passed under ASan/UBSan**.
  Real independent process tests prove reopen, hot-journal recovery, concurrent
  revision conflict and UTF-8 filenames. See `backend/RESOURCE_STORE.md` for
  the exact coverage and limitations, including the upstream compiler warning.
- Implemented bounded keyset enumeration and schema-1 JSON package import/export,
  rejecting unknown fields, duplicate decoded keys, NULs and invalid text before
  mutation. Explicit caller namespace/revision policy cannot be supplied by a
  package. Tests cover maximum JSON expansion and unchanged data after failures.
- Added the real pure-C `fyodor` executable and shared native command dispatcher
  for ID creation, JSONL listing, piped import, export and revision-checked delete.
  Unicode Windows arguments use a wide entry point. No subprocess wrapper or
  new runtime dependency. This is the beginning of the CLI, not its completion.
- Latest CPU suite: **56 passed, 4 hardware skips**. Five application tests pass
  under both ASan/UBSan and C23. See `backend/RESOURCE_CLIENT.md` for usage, schema,
  actual process tests and limits. No unrelated process was terminated.
- Added schema-2 exact principal/resource permissions with atomic v1 migration
  and no automatic grants. Added authorized reads, literal ranked retrieval and
  UTF-8-safe byte-bounded assembly with revision/offset attribution in one read
  snapshot. Native CLI exposes permissions, inspection, search and assembly
  preview. WRITE/APPEND/DELETE/EXECUTE bits are reserved, not implemented scoped
  operations. Trusted admin storage APIs remain separate from context access.
- CPU suite: **57 passed, 4 hardware skips**; six application tests pass under
  ASan/UBSan, followed by the extended native CLI test after its final changes.
  See `backend/CONTEXT.md` for concrete semantics and coverage limits.
- Next: connect bounded context to actual native generation, model-token budget
  checks and durable request provenance. Continue scoped mutation enforcement,
  context layers/indexing, Desktop/TUI and node authentication afterward.
- Added `fyodor-inference`, reusing the native generation API and a new public
  native-tokenizer count query. Authorized context is shortened deterministically
  to reserve requested output tokens; the user prompt is not shortened. Returned
  prompt/output/source attribution describe actual inference. CPU suite **58
  passed, 4 hardware skips**; integration plus trained fixture passed under
  ASan/UBSan and C23. See the native-generation section of `backend/CONTEXT.md`.
- Next remaining integration: durable executed-request provenance and client
  generation commands. Do not confuse current assembly preview with persisted
  proof of which context entered a completed request.
- Implemented schema-3 immutable receipts and migrations, storing exact prompt,
  output, source revisions/ranges and native token/sampling metadata. Reading
  receipts requires both original ownership and current source READ grants.
  Native `context generate` and `context receipt` now demonstrate the full
  local authorized context → inference → saved provenance workflow in separate
  CLI processes. The shared `fyodor-client` library owns command dispatch;
  `fyodor-app` remains engine independent.
- Latest rebuilt CPU suite: **58 passed, 4 hardware skips**. ASan/UBSan application
  selection: **8 passed including the trained-model fixture**. Final added
  migration/atomic-failure assertions also pass. See `backend/REQUEST_RECEIPTS.md`.
- Next: expand authenticated application operations, layered/indexed context,
  native TUI and Desktop integration. Remaining major product workspaces and
  remote-node protocol are still open; this checkpoint is not product completion.
- Added scoped WRITE, APPEND and DELETE with one writer transaction for
  grant/revision checks, mutation and history. APPEND preserves non-content
  fields and does not imply READ. Native CLI append/delete use these APIs.
  All 64 permission masks and history-failure rollback are exercised. CPU:
  **58 passed, 4 hardware skips**; sanitizer and C23 application selections:
  **8 passed including the trained-model fixture**. EXECUTE remains reserved.
- Next concrete interface work: native TUI over these shared services, followed
  by Desktop integration and remaining context/node/workspace capabilities.
- Added the initial pure-C `fyodor-tui`: a paged local resource workspace,
  bounded content/command previews, shared native command dispatch and safe
  plain-terminal fallback. No CLI subprocesses. ASCII-safe rendering escapes
  untrusted controls and non-ASCII bytes; full Unicode/themes remain open.
- Three TUI tests pass under C17, C23 and ASan/UBSan. A private hidden Windows
  console test exercises the real alternate screen, help, command dispatch,
  quit and Ctrl+C mode restoration. The hidden host did not retain a simulated
  resize, so resize and POSIX behavior remain unverified. See `backend/TUI.md`.
- Continue shared semantic themes/Unicode terminal rendering, application
  workspace integration and the still-open product capabilities in this plan.
- Rebuilt full C17 CPU regression after TUI integration: **61 passed, 4 hardware
  skips** (65 tests total). Logs: `build-cpu/tui-build.log` and
  `build-cpu/tui-tests.log`. No benchmark/performance claim is added.
- Added one shared semantic theme source and generated C/CSS/JS adapters for
  all ten required presets. Desktop defaults to Fyodor and migrates old names;
  Settings persists choices. TUI supports explicit/automatic color capability,
  startup selection and live cycling. Catppuccin data/license are vendored and
  licenses ship in both interfaces. See `themes/README.md` for exact scope.
- Four native theme/TUI tests pass under C17, C23 and ASan/UBSan; 14 frontend
  tests and production build pass. Real headless browser QA verifies all ten
  palettes and Settings selection. Advanced appearance controls, terminal
  Unicode, POSIX testing and wider product work remain open.
- Added bounded persisted Desktop appearance controls: accent, typography,
  corner radius, sidebar/workspace spacing, chat width, borders/shadows,
  supported translucency and motion. Code-fence text and lexical syntax colors
  are customizable; escaping/text preservation remain tested. All 16 frontend
  tests and production build pass. Browser QA exercises live controls, reload,
  reset, computed styles and inert code rendering; inspected customized UI.
- Remaining: TUI Unicode/layout preferences and full workspace/native service
  integration, layered context, nodes, training/dataset workflows and the other
  product requirements above. No full-product completion claim is made.
- Connected the desktop Resources library to authenticated local C HTTP and
  the same persisted store used by CLI/TUI. Tauri supplies its app-data path;
  C handles list/get/put/title-content update/export/delete. Revision strings,
  metadata-preserving updates and raw JSON package transfer avoid browser
  integer rounding. Remote-enabled servers cannot expose these admin routes.
- Real browser create/save/reload -> separate native CLI export passes. Native
  HTTP tests cover auth, namespace isolation, competing updates, restart and
  large-number metadata preservation. CPU integration suite: **63 passed,
  4 hardware skips**; affected HTTP tests pass under C23 and sanitizers. Frontend
  build/tests and Tauri cargo check pass. See `backend/RESOURCE_HTTP.md`.
- Next: desktop authorized context selection/generation/provenance and richer
  resource workspaces, while keeping the remaining original requirements open.
- Added the Desktop Context workspace and local HTTP permissions/generate/
  receipt routes over the existing shared C service. Selection does not grant
  access. Explicit grants, real native tokenizer/output reservation and durable
  receipts are enforced; later revocation hides generation context and receipt
  reads. Principal labels remain distinct from local bearer authentication.
- Real browser and HTTP tests exercise denied → grant → native inference →
  receipt → reload → revoke → denial, with the same executed receipt read by a
  separate CLI process. Full CPU suite: **63 passed, 4 hardware skips**;
  affected C23/sanitizer checks and frontend build/tests pass. Details and
  remaining context gaps: `backend/CONTEXT_HTTP.md`.
- Added permission-filtered receipt history in shared C, native CLI/TUI,
  local HTTP and Desktop. UUID keyset pages expose only authorized IDs/times;
  reads recheck permissions. Reopening after reload no longer requires keeping
  each receipt ID manually. Existing schema remains unchanged.
- History pagination/owner/revocation and real browser reopening checks pass;
  15 selected C17 application/API tests, affected C23/sanitizer tests and
  frontend production build pass. Continue layered/searchable context and
  richer workspaces; the original full product objective remains open.

- Connected permission-filtered context search to local HTTP and Desktop using
  the shared C service. READ and SEARCH are both required; grants are explicit,
  matches stay selectable, and identity/namespace changes clear stale results.
  Literal first-20 title/content search is not indexed or semantic retrieval.
- Search permission/revocation checks pass through HTTP and the real browser;
  affected C23/sanitizer suites (four each), 16 frontend tests and production
  build pass. Continue indexed retrieval/context layers and richer workspaces.

- Added five-layer shared C assembly/generation: explicit, workspace, session,
  retrieved and global, with stable priority ordering and persisted source-layer
  attribution. All sources require READ even beyond budgets; retrieved sources
  also require SEARCH. CLI/TUI accept LAYER=URI; HTTP accepts parallel layers;
  Desktop provides per-resource selectors and receipts show the executed layer.
- Verified layer order, same-layer stability, authorization, shape validation,
  real tokenizer truncation and receipt attribution. Selected C17 tests (15),
  affected C23/sanitizer tests (six each), browser workflow, 16 frontend tests and
  production build pass. Automatic workspace/session/global providers, persisted
  memberships and indexed retrieval remain open alongside the larger brief.

- Added persistent FTS5 trigram context indexing in schema 4. Resource writes,
  imports, scoped edits/appends and deletes update it in the same transaction
  as heads/history. Validated migration indexes live content and preserves grants
  and receipts. Search keeps literal semantics, permission filtering and stable
  title-first ranking; queries shorter than three Unicode scalars still scan.
- Differential index/scan, Unicode/literal boundary, rollback, revocation,
  namespace, VACUUM and populated-v3 migration checks pass. Full CPU suite:
  64 passed, four hardware skips; eight affected C23/sanitizer tests and real
  browser/CLI interchange pass. Browser QA now copies its model fixture to
  avoid locking the training suite's output. See backend/CONTEXT_INDEX.md.
- Next: automatic feature context providers and persisted memberships, richer
  retrieval/ranking and the remaining specialized workspace/product requirements.

- Added the Writing page over shared resources: native filtered pagination,
  document/note/character creation, lightweight text editing and package transfer.
  Character IDs remain reusable shared identities. Added C revision-history
  pagination and historical export, exposed through local HTTP and CLI/TUI.
- Writing/Resources can preview old text and copy it into a draft; saving appends
  a CAS-checked revision and preserves current metadata/provenance. Browser QA
  verified original -> edit -> historical copy -> third revision, subsequent
  Context generation, character creation and reload persistence. Native tests
  cover history pagination/tombstones, metadata precision and namespace filtering.
- Seven selected C17 tests, five affected C23/sanitizer tests each, browser QA,
  16 frontend tests and production build pass. Project/folder organization, lore
  links and integrated Writing generate/rewrite/continue remain open.

- Connected Writing Generate/Rewrite/Continue to a shared native C preview
  service with target revision validation on the authorized assembly snapshot.
  Added local HTTP and CLI/TUI access, model/context selection, explicit grants,
  exact-prompt receipts and desktop review/apply controls. Inference does not
  mutate documents; accepting a preview creates a draft saved with normal CAS.
- Native/HTTP/CLI/browser tests cover all modes, denial, stale targets, unchanged
  documents and applying/saving Continue. Seven selected C17 tests, five affected
  C23/sanitizer tests each, 16 frontend tests and production build pass.
- Next: persist the accepted-edit-to-receipt provenance link; project/folder
  organization, lore tools and the broader product requirements remain open.
  Details: backend/WRITING_GENERATION.md.

- Added transactional accepted-edit receipt links through shared C, local HTTP,
  CLI/TUI and Desktop Writing. Saves verify receipt ownership, current source
  access and target revision, preserve metadata/provenance and append mode,
  source/saved revisions and whether the accepted text was edited. Failed
  validation or history insertion leaves the resource unchanged.
- Native tests cover revocation, stale/ineligible receipts, rollback, large
  metadata integers and repeated links; HTTP/CLI saves interoperate. Browser
  acceptance/reload is independently verified by native package export.
  Seven selected C17 tests, five affected C23/sanitizer tests each, browser QA,
  all 16 frontend tests and production build pass.
- Next: project/folder organization and automatic feature context providers,
  lore-reference tools and a persisted-link viewer. Explore, Dataset Studio,
  remote nodes and the remaining original product requirements remain open.

- Added nested Writing projects/folders with stable resource identities,
  revisioned parent metadata, native cycle/live-parent validation, protected
  nonempty deletion, folder pagination and transactional moves. Desktop, HTTP
  and CLI/TUI share these services. Package transfer preserves membership;
  import ancestors first. See backend/WRITING_FOLDERS.md.
- Twelve selected C17 application tests and seven affected C23/sanitizer tests
  each pass. Browser create/move/filter/reload and independent CLI export pass,
  along with metadata-limit and injected rollback checks.
- Applied the user's claymorphic UI direction through shared material tokens
  and component styling in frontend/src/clay.css, preserving theme palettes,
  appearance overrides, reduced motion and forced-color behavior. Removed
  implementation commentary from the visible Writing, Context, training and
  appearance flows; technical rationale belongs in source comments.
- Frontend build and all 16 unit tests pass. Browser QA checks ten palettes,
  preference persistence and folder workflow, with light/dark and 390px-wide
  screenshots. Automatic project context, lore tools, subtree transfer and the
  remaining full product scope are still open.

- Connected the Writing project hierarchy to native context assembly. Readable
  parent-project text enters the workspace layer automatically, nearest first;
  denied parents stop traversal. Discovery, target CAS validation, permissions
  and source content use one snapshot. Manual selections reserve source slots
  and retain their layers; byte/token budgets and receipt attribution still
  apply. No implicit grants or sibling/database-wide enumeration were added.
- Native tests verify ordering, intermediate denial, namespace isolation,
  manual duplicate precedence, source limits, zero budgets and stale moves.
  Actual inference and HTTP/CLI tests verify exact prompts, immutable project
  revisions in receipts and revocation behavior. Browser QA grants then
  unselects a project and verifies automatic inclusion in the saved receipt.
- Five affected CTest entries pass under C17, C23 and ASan/UBSan; real browser
  QA and the frontend production build pass. The browser fixture explicitly
  requests two output tokens after reload to fit its tiny 32-token model.
- Remaining: lore-reference editing, richer project context retrieval and
  inspection, subtree transfer, Explore/Dataset Studio and the other original
  product requirements. The claymorphic UI and source-only implementation
  commentary requirements continue to apply to subsequent work.

- Added persistent Writing lore references to existing shared world-lore IDs.
  Native validation, revision-checked replacement, history/package preservation
  and deletion guards apply to HTTP, CLI/TUI and imports. The same snapshot
  provider includes readable direct/project lore, deduplicates selections and
  retains executed source revisions in receipts. Links never grant permissions.
- Added a separate render-only lore component for Desktop: paginated browsing,
  link/unlink, explicit model access, reload persistence and titles. Shared
  metadata mutation and HTTP URI-array parsing avoid duplicate implementation.
- Seven affected CTest entries pass under C17, C23 and ASan/UBSan; browser
  linking/grant/generation/reload plus independent CLI export, 16 frontend
  unit tests and production build pass. See backend/WRITING_LORE.md.
- Next: the Explore workspace and shared character/world/lore workflows,
  while retaining the remaining Dataset Studio, evaluation, agent, remote-node,
  packaging and other original product requirements.

## Frontend priority and Snep writing reference (2026-10-05)

- User assigned backend work to another agent. Subsequent implementation in this
  milestone is frontend-only; existing native APIs/binaries are integration
  dependencies, not newly implemented gameplay services.
- Added Explore world/lore authoring via existing typed resource endpoints:
  create/edit, scoped lore library, revision preview/copy, raw package export,
  deletion confirmation, dirty guards and explicit stale-write recovery.
  Domain/paging helpers and the view controller are separate modules. The catalog
  compatibility adapter bounds scans and retains server cursors until native
  world/parent-filtered listing exists.
- Inspected the user's C:/Users/Colin/snep reference read-only. Writing now uses
  its editor-first composition: larger left text surface, right library/creative
  controls, live counts, visible unsaved status and focus mode. Optional story
  guidance contributes explicit prompt text and remains session-only. Native
  permissions, folders/lore, previews and accepted receipts retain their existing
  semantics. No speculative inherited metadata or sampling sliders were added.
- Thorough frontend documentation now covers architecture/module ownership,
  lifecycle/state/persistence, clay styling, write boundaries, API contracts,
  Explore failure/paging behavior, Snep adaptation, backend-owned requirements,
  validation setup and the remaining delivery map. Start at frontend/README.md.
- Production build and 21 frontend unit tests pass. Browser QA also exercises
  the existing Writing/resource/context flow through the new layouts, actual
  world/lore persistence and independent CLI exports. Screenshots/evidence live
  under frontend/qa/themes. Desktop installers and unimplemented gameplay are
  not represented as verified.
- Remaining frontend work includes inherited creative controls/path hierarchy,
  trackers/recursive Writing, Explore gameplay once contracts exist, Dataset
  Studio, evaluations/model lineage, agents/workflows, remote pairing and full
  accessibility/platform validation. See frontend/docs/ROADMAP.md and
  frontend/docs/WRITING_UX.md for contract boundaries.

## Writing text interchange and recovery (2026-10-06)

- Added ordinary Markdown/plain-text import/export alongside existing resource
  package JSON. Strict UTF-8/BOM, format, NUL and 1 MiB checks run before replacing
  the editor. Imports become fresh unsaved documents, retaining a selected native
  project parent, and never overwrite an open resource or grant permissions.
- Draft downloads preserve current text, including unsaved changes. Portable
  filenames preserve Unicode and avoid invalid Windows/device names. Package
  export retains the saved-resource boundary. Encoding, newline behavior and
  format/lifetime limits are documented in frontend/docs/WRITING_UX.md.
- Added Ctrl/Cmd+S in the Writing editor, explicit latest saved-version reload,
  dirty unload warnings and keyboard activation for both file-import controls.
  Existing native CAS failures retain local drafts. Successful reloads adopt the
  current native revision and clear generated/accepted previews.
- Production frontend build and all 24 selected unit tests pass. The Windows
  browser workflow checks exact downloaded Markdown/text bytes, import dirty
  guards, keyboard activation/save, project membership and a concurrent writer's
  conflict/reload path. Separate CLI export confirms stored text and parent.
  Independent read-only review found the import keyboard gap, which is fixed
  and covered by the browser check.
- No backend implementation was changed or rebuilt for this frontend milestone.
  Remaining work follows frontend/docs/ROADMAP.md; the full original goal is
  still open and includes backend-owned contracts and broad product validation.

## Dataset Studio frontend checkpoint (2026-10-06)

- Added a dedicated claymorphic dataset view and separate domain/controller
  modules. Existing native resource APIs persist dataset UUIDs and revisions;
  no new backend implementation is part of this checkpoint.
- Versioned frontend format annotations distinguish pretraining/CPT corpus,
  two-field SFT and three-field DPO. Strict UTF-8 import and structural validation
  enforce documented native grammar bounds without claiming tokenizer/context
  suitability. Inspection shows record fields, issues and exact duplicates.
- Seeded sample/shuffle/split, filtering and exact dedup create fresh derived
  resources with source URI/revision and parameter provenance. Split copies save
  independently, retaining successful revisions and guarding unsaved parts.
- Text exports preserve current draft bytes; saved raw packages preserve native
  metadata. Export/open Training sets the format and clears stale data paths;
  selecting the downloaded file is still required. Direct native resource
  training and source ingestion need the separately owned backend contracts.
- Thorough maintenance/API/format/lifetime/failure documentation is in
  frontend/docs/DATASET_STUDIO.md, linked from the architecture and delivery map.
- Production build, 29 frontend checks and real browser/native integration pass.
  The exported SFT fixture completes one native trainer update. The existing full
  CPU C17 suite passes 64 entries with four hardware-dependent skips; this is
  checkpoint verification of earlier native work, not new backend implementation.
- The full goal remains open. Upcoming work includes context inspection and the
  backend-dependent workflows recorded in frontend/docs/ROADMAP.md.

## Frontend dataset source ingestion (2026-10-06)

- Added a source picker and separate mapping/provenance domain module. Human
  selections mix exact saved Writing/Explore revisions with frozen local chat
  text. Sources remain unchanged and new dataset drafts require explicit Save.
- Corpus mapping preserves resource text and labels chat roles. SFT explicitly
  maps a shared resource prompt/completion or adjacent user/assistant pairs;
  tool-interrupted pairs are excluded. Tabs/newlines reject by default; explicit
  space replacement is versioned in provenance. No DPO preference labels are
  invented. Source/output byte bounds, exact hashes, source revisions, mapping
  ranges and chat message indices are recorded and documented thoroughly.
- Existing local administrative reads do not enforce Context principal grants.
  Native permission-scoped Context ingestion, durable chat snapshots, managed
  materialization/direct Training and broad product requirements remain open.
  No backend implementation changed in this frontend milestone.
- Production build, 35 selected frontend checks and real browser integration
  pass. Browser/CLI evidence verifies a selected historical revision after a
  concurrent source edit, mixed sources, guard/recovery behavior and downloaded
  bytes. Real Writing/Explore CPT and local-chat SFT exports each complete one
  native training update. Source-picker light/dark/mobile images are retained.
- Shared UTF-8 title bounds prevent derived Unicode names exceeding the native
  title limit. Maintenance contracts and limitations live in
  frontend/docs/DATASET_SOURCES.md and the architecture/delivery map.

## Frontend Context receipt inspection and ingestion (2026-10-06)

- Verified existing native receipt access: original identity/namespace plus
  current READ on every source in one SQL snapshot. Frontend selects included
  sources through fresh native reads, with no administrative fallback/new API.
- Added pure exact-revision/UTF-8 range validation and render-only attribution.
  Context shows included/truncated/omitted/empty usage and actual source text;
  original prompt/metadata/source JSON remain inspectable. Large revisions retain
  exact source lexemes or fail on unsupported engines rather than rounding.
- Dataset mapping copies selected prefixes, excluding instructions/output. It
  records receipt/principal/source revision, original/included ranges, layers
  and hashes, rechecks access and preserves drafts on denial. Reads are separate
  from later Save/Training; already disclosed text cannot be recalled.
- Build, 40 selected frontend checks and real browser/native integration pass.
  Revocation after selection blocks creation and preserves the prior draft;
  explicit regrant recovers. CLI provenance/export bytes agree, and the Context
  corpus completes one native CPT update. Light/dark/mobile images are retained.
- Thorough docs are in frontend/docs/CONTEXT_RECEIPTS.md. Arbitrary current full
  Context reads, direct Training/materialization, evaluations, Explore play,
  agents/workflows, remote nodes and platform/accessibility validation remain
  part of the active full goal. No backend implementation changed here.
