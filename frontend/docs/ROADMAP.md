# Frontend delivery map

Updated 2026-10-06. Frontend work is prioritized; another agent owns backend
implementation. The original product brief remains the wider target. This map
separates usable frontend behavior from planned screens and required contracts.

## Current coverage

| Area | Usable now | Still open |
| --- | --- | --- |
| Shell/appearance | Grouped navigation, shared model selector, responsive overlay, named themes, clay material, preferences | Full keyboard/screen-reader/zoom audit and final cross-platform packaging QA. |
| Chat | Stored conversations, JSON import/export, safe code, regenerate/edit, remote streaming/Stop | Native local streaming/cancel contract and richer agent execution UI. |
| Writing | Editor-first studio, focus mode, live counts, session story guidance, document/note/character/project editing, folders, history, lore links, model preview, accepted receipt provenance, Markdown/text files, keyboard save, conflict reload | Inherited creative controls, trackers, recursive generation, complete project workflows and rich document formats; see `WRITING_UX.md`. |
| Resources | Paged local library, package import/export, edit/delete/history | Typed catalog filters and richer navigation across every resource family. |
| Context | Explicit grants/revocation, layers, search, generation, readable executed-source usage and receipt-backed datasets | Current full-resource inspection/retrieval APIs, shared workspace/session coordination and automatic retrieval UX; see `CONTEXT_RECEIPTS.md`. |
| Explore | World/lore creation/editing/history/export, scoped lore library | Play/saves/state/timeline/conditions/hooks/assets; see `EXPLORE.md`. |
| Models | Registry and runtime capability display, load/unload, compute selection | Complete model card/evaluation/lineage UI over agreed native registry metadata. |
| Evaluations | Saved test definitions/results, 1–4 local model comparisons, literal/JSON checks, human verdicts/notes, exact revision references and package export | Saved evaluation datasets/mappings, immutable model/training lineage, native jobs/cancellation and genuine loss/perplexity; see `EVALUATIONS.md`. |
| Datasets | Native persistence, corpus/SFT/DPO inspection, import/edit, seeded transforms, exact dedup, Writing/Explore/chat and permission-checked Context receipt ingestion, provenance and export | Arbitrary current full-resource Context ingestion, durable chats, native tokenizer validation, direct resource training and large/streaming datasets; see dataset/Context docs. |
| Training | Native file paths, modes/parameters, logs/loss/validation telemetry, stop/save; explicit Dataset Studio export/mode handoff | Managed dataset materialization, curated run/evaluation history and richer reproducible launch flow. |
| Benchmarks | Import measured reports and view metrics | Native job orchestration if/when an agreed service exposes it. |
| Logs/API | Logs with filters and API route presentation | Authenticated remote pairing/capability/session UI, WSS lifecycle. |
| Agents/workflows | Existing Chat endpoint tools/settings | Full persisted agent/workflow editor, permissions, runs and pause/resume/status. |

## Recommended frontend sequence

1. Continue authoring workflows and maintainability. Extract view/controller pieces
   from `main.js` and `resources.js`, retain behavior coverage and shared tokens.
   Improve source inspection and context receipts without exposing engineering notes
   as primary product text.
2. Extend Dataset Studio with current full-resource Context ingestion and direct training once native
   revision materialization, permission and tokenizer contracts are agreed.
   Basic editing/inspection/transforms use existing resource storage; file-path
   Training remains the usable launch path.
3. Integrate Explore gameplay once save/state/action contracts exist. Build the
   transcript and state inspector first, then timeline/branches and conditional
   lore. Preserve the authoring service independently of runtime play.
4. Build model/evaluation/run and agent/workflow views using real persisted records.
   Avoid sample cards presented as a live registry or fabricated successful jobs.
5. Add remote connection/pairing and recovery after the backend specifies auth,
   capability grants/revocation, WSS events, reconnect and cancel semantics.
6. Complete focus/keyboard/reader/zoom/high-contrast audits, realistic dataset/model
   scale, actual Tauri windows and platform installer verification.

This is sequencing guidance, not approval to alter backend contracts. Frontend
can implement isolated presentation/controllers, but unavailable mutations must
remain unavailable and their pending requirements should be documented here.

## Handoff checklist for new contracts

Before wiring a feature, document its exact route/event schema, typed identity,
revision semantics, permissions, bounds and error behavior. Include:

- pagination ordering/cursor termination and filtering;
- creation/update/delete idempotency and conflict recovery;
- cancellation, reconnect, terminal job states and partial-output persistence;
- provenance, actual source visibility and permission revocation;
- package/import compatibility and safe treatment of unknown metadata;
- capability discovery so the UI can show only supported actions;
- test fixture setup and independent persistence verification.

Prefer a contract recorded in repository docs to assumptions embedded in markup.
The frontend API boundary is `src/api.js`; domain helpers should consume it through
small adapters where unit tests benefit from injecting a fake transport.

## Validation record

2026-10-05 Explore authoring and Writing studio frontend: production Vite build and 21 selected
frontend unit tests pass. Browser results are recorded after the corresponding
end-to-end run; see `EXPLORE.md` and `../qa/themes`. Existing native binaries are
used as integration fixtures. No backend rebuild/change is part of this frontend
milestone. Actual desktop/installer and unimplemented feature claims are excluded.

2026-10-06 Writing file interchange adds three unit checks (24 selected frontend
checks total) and a browser workflow covering real text downloads, import into
project/folder, dirty guards, keyboard save and concurrent-edit recovery. See
`WRITING_UX.md` for format/lifetime limitations and the reproducible test commands.

2026-10-06 Dataset Studio adds five boundary checks (29 selected frontend checks
total). Browser QA verifies real SFT persistence, source-preserving derivation,
independent split saves, exported bytes, metadata reload, invalid export recovery
and clearing old Training paths during the mode handoff. A separate native CLI
export confirms provenance; the exported TSV completes one actual SFT update.
See `DATASET_STUDIO.md` for supported formats and remaining native contracts.

2026-10-06 Writing/Explore/local-chat source ingestion adds six domain checks
(35 selected frontend checks total). Browser QA verifies exact historical reads
after an independent writer advances a source head, mixed-library selection,
dirty guards, delimiter policy, actual corpus/SFT downloads and unchanged chats.
Separate CLI exports confirm provenance and the exports complete native CPT/SFT
updates. See `DATASET_SOURCES.md` for snapshot/hash semantics and native boundaries.

2026-10-06 Context receipt inspection/ingestion adds five checks (40 total).
Browser QA verifies included-prefix display, selection, denial after revocation
without draft loss, explicit regrant/recovery, downloads and CLI provenance. The
Context export completes one native CPT update. Current full-resource Context
reads and direct dataset Training remain separate contracts.

2026-10-06 Evaluation frontend adds seven domain checks (47 selected checks
total). Real browser/native QA runs exact/manual/overflow cases on two loaded
fixture copies, saves/reloads observations, edits verdicts with focus/count
updates, recovers from an independent CAS conflict and downloads a package.
Independent CLI exports verify definitions, exact source revision, native token
counts, actual error rows and saved review notes. Light/dark/390px views have
no page overflow. Native benchmark timing, loss, jobs and immutable model identity
are not inferred from these frontend records.
