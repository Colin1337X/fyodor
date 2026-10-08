# Agents frontend and backend handoff

Implemented 2026-10-08. This workspace extends agents beyond the existing Chat
toggle with saved profiles, explicit tool choices, execution limits, Stop and
durable run history. It consumes the existing native generic resource store and
OpenAI-compatible endpoint client. It is not a canonical native Agent service,
background job scheduler, workflow engine or Node Protocol implementation.
Backend implementation remains owned separately.

## User workflow

1. Configure an OpenAI-compatible connection in Settings. Choose an endpoint that
   supports Chat Completions function calls. Existing native compatibility routes
   do not implement tool calling; selecting a local model does not enable it.
2. Open Agents from Run navigation or the quick switcher. Create a profile, give
   it a title and instructions, and specify its endpoint/model name. “Use
   configured connection” copies only those two fields from Settings. It does not
   copy or save the API key, change Chat settings, or load/unload a local model.
3. Enable individual read-only tools if needed. New profiles enable none. Set the
   model-turn, tool-call and per-turn sampling limits, then save the profile.
4. Enter a goal and run the saved profile. The current configured endpoint must
   match the profile's normalized endpoint before its in-memory key can be used.
   The model name comes from the saved profile, independently of Chat's model.
5. Inspect assistant output and the ordered recorded steps. Stop aborts a pending
   endpoint fetch and prevents further requests/tool reads. Save the finished run
   explicitly, or discard it. Refresh/open history to inspect earlier saved runs.
6. Export a saved profile/run as a native resource package. Export retrieves the
   current native head, not a pinned historical export. Resources provides generic
   package interchange/inspection; there is no separate Agent file importer.

Runs can finish while another workspace is open. Their unsaved results remain in
the current window; returning to Agents shows them. They do not continue after
window/process exit. An unload warning protects dirty profiles, running work and
unsaved runs where supported by the host. There is no automatic save/recovery of
drafts or in-flight requests. Opening another profile, changing namespace, paging
or starting another run requires the current profile/result to be saved/discarded.
Reload saved profile deliberately discards its editor changes and is blocked
while a run is unsaved. Discard editor and run is an explicit destructive-to-draft
action; it never deletes the saved native resources.

## Modules and maintenance boundaries

| File | Responsibility |
| --- | --- |
| `src/agent-data.js` | Typed identities, closed versioned schemas, exact revision references, endpoint binding, resource packages and bounded catalogs. No DOM or credentials in stored data. |
| `src/agent-runner.js` | Bounded complete-response endpoint/tool loop, narrow native observations, metrics and cancellation. Injected API, independent of presentation. |
| `src/agent-workspace.js` | Editor, start-head check, CAS writes, dirty/pending guards, progress, history and package downloads. |
| `src/api.js` | Existing native authenticated HTTP and external endpoint fetch, including AbortSignal. No new transport implementation. |
| `src/icons.js`, `src/main.js` | Shared navigation, known persisted view, header metadata and mount dispatch only. |
| `src/clay.css` | Semantic clay surfaces, responsive profile/run/history layouts and bounded preformatted output. No palette-specific colors. |

State is module-local. Mount generation plus active-view checks prevent an old
async completion painting into another workspace. The latest mounted renderer
receives progress on return. Inputs update state without repainting on each
keystroke. Work disables mutable controls; Stop stays available during execution.
Profile controls stay locked while a run awaits save/discard. Run output, titles,
errors and tool observations are escaped text, never executable HTML/SVG.
Implementation rationale belongs in comments and this guide, not product prose.

## Native storage and concurrency

Profiles use `fyodor://agents/<uuid>`; runs use the native child identity
`fyodor://agents/<uuid>/runs/<uuid>`. UUIDs are lowercase canonical, non-nil values.
The native store derives type/parent from these identities and requires a live
agent parent on run creation. Frontend code does not fabricate parent columns.

Packages retain native schema 1 and `metadata.agent_studio = {version:1,
kind:"definition"|"run"}`. Provenance identifies `agent-definition` or
`client-endpoint-run`. Existing profile title/content changes use
`/api/v1/resources/update`, preserving native metadata/provenance. New profiles
and runs use `/resources/put` with expected revision `"0"`. UUIDs are retained
across retries of ambiguous writes; retries do not silently create duplicate
records. A conflicting or ambiguous save preserves editor/result state.
There is no automatic reconciliation of a write that succeeded but lost its
response; inspect the native resource before retrying or discarding in that case.

`/resources/get` supplies the current head and exact decimal-string revision.
Immediately before execution the controller re-reads that head. A revision
change blocks execution until Reload, preventing a stale permission profile from
starting after another client edited it. The exact started definition, namespace,
URI and revision are embedded in the run. Namespace uses 1–64 ASCII letters,
digits, underscores/dashes; revision is a positive decimal string up to native
signed-64-bit range. It is never rounded through a JavaScript Number.

The start check cannot prevent an edit/revocation after the check. This is a
frontend snapshot guard, not transactional native execution authorization.
Native resource administrative access is not a login or Context grant. Agent
records use that existing namespace store; endpoint credentials are not native
resource permissions. Native Agent execution must eventually own those checks.

Catalogs use existing `/resources/list`, filtering exact types and exact run
parent. A scan consumes at most 25 native pages, with at most 100 rows per page,
stopping at 20 matches or exhaustion. Every match on the final consumed page is
retained; Next uses the server's actual continuation cursor. Sparse pages are
consumed and repeated cursors reject. Definitions with future/unknown annotations
can appear in the library but fail closed on opening, leaving generic inspection
and export available through Resources. No lossy implicit schema migration occurs.

## Definition format and bounds

Closed JSON schema 1 has `kind:"agent-definition"` and exactly these fields:

| Field | Supported values |
| --- | --- |
| `instructions` | Raw UTF-8 string, up to 16 KiB, no NUL. |
| `target` | `{kind:"configured-endpoint",endpoint,model}`. Endpoint HTTP(S), up to 2048 UTF-8 bytes, no embedded credentials/query/fragment; normalized URL with trailing slash removed. Nonblank model name up to 256 bytes. |
| `tools` | Unique subset of `runtime_status`, `list_loaded_models`, `local_time`; defaults empty. |
| `permissions` | Exactly `{context:false,resources:false}`. Unsupported grants reject rather than execute. |
| `limits` | Integer `turns` 1–12, integer `tool_calls` 0–48. |
| `sampling` | Integer `max_tokens` 1–4096, finite `temperature` 0–5, finite `top_p` greater than 0 and at most 1. |

Unknown keys, versions, malformed/oversized values and unsupported permissions
reject. Drafts may be incomplete until Save; execution uses only a saved valid
definition. No engine seed/context size is inferred for the remote endpoint.
Sampling is requested behavior; the endpoint may implement or reject it.

## Execution and permission boundary

The runner sends an exact system instruction and goal (`input` up to 8 KiB),
then non-streaming Chat Completions turns through the existing external client.
Only enabled function schemas are advertised. No tools means no `tools` key.
No script interpretation, dynamic API property lookup, shell, filesystem,
resource body, Context retrieval, local generation, node or arbitrary HTTP tool
exists in this runner. Local read results may be sent to the configured endpoint
only when the corresponding tool was explicitly enabled.

Tool adapters return narrow observations:

- `runtime_status`: actual version-1 native `/runtime` compiler booleans and loaded
  model runtime IDs/active compute. Compiler inclusion is not detected GPU support.
- `list_loaded_models`: actual version-1 native `/models` IDs, filename basenames
  and reported architecture. Absolute paths, other metadata and bearer tokens are
  excluded. IDs are process observations, not immutable model lineage.
- `local_time`: the client's actual current ISO timestamp.

Native observations validate scalar IDs/types and known compute providers; missing
compiler fields fail rather than become invented false values. Tool output is
bounded to 32 KiB. There are at most eight calls in one response. The entire
batch's names, unique nonempty IDs (up to 128 UTF-8 bytes), empty-object JSON
arguments (up to 1024 bytes) and remaining call budget are checked before any
member executes. One denied call rejects the entire batch without local reads.
Permitted calls execute sequentially; successful observations enter the next
model request with the corresponding tool-call ID. No automatic tool retry occurs.

Stop calls AbortController on the endpoint fetch and checks between steps. An
already-started native read has no cancellation API; it may finish and its actual
observation is recorded, but no later model request is sent after Stop. Browser
abort is not evidence that a remote server stopped its computation, refunded
tokens, acknowledged cancellation, or deleted submitted text. Endpoint model
requests already issued may have cost. There is no native job ID, pause/resume,
reconnection, background scheduling or remote cancellation acknowledgement.

## Run records and measurements

Closed schema 1 uses `kind:"agent-run"`, exact `agent` reference, full saved
`definition`, original `input`, status `completed|stopped|failed|limit`, `steps`,
latest actual assistant `output`, `error`, client `started_at`/`finished_at`.
Saved child identity must match the embedded agent URI; history additionally
checks namespace/parent before opening. A saved run is read-only in this workspace.
It does not rerun tools or silently adopt a changed current definition.

Steps contain `kind`, `status`, `text`, nullable `tool`, `error`, nullable
`request_ms` and nullable `tokens`. Model request duration measures client time
through fetch/JSON completion (or failure), not engine-only latency, throughput,
TTFT or benchmark performance. Tokens are the endpoint's actual safe nonnegative
`usage.completion_tokens`; absent/invalid usage remains null and is labelled
“Tokens not reported”. Tool steps have no token/duration measurements. Failed
requests and tools preserve explicit errors; malformed responses cannot produce
successful output. Final status records limit/error/Stop separately from completion.

Assistant/step text is up to 64 KiB each; errors up to 4096 bytes. At most 64
steps and 1 MiB serialized record are accepted. The runner bounds cumulative
recorded steps to 512 KiB and serialized outgoing history to 768 KiB, stopping
with an explicit limit rather than truncating a purported complete result.
Client timestamps are epoch milliseconds and must be ordered safe integers.
The UI clips only the scroll viewport, retaining complete bounded text.

API keys/connection bearer tokens are not intentionally serialized in profiles,
runs or tool observations. They remain in the existing connection memory unless
the user enables Settings Remember (existing localStorage behavior). Arbitrary
endpoint output/errors are untrusted text and may contain what that endpoint
returns; the runner is not a general secret redaction service. Profile endpoint
and model name, instructions, goal and actual responses are intentionally saved
when the user saves a run. There is no prompt-level Context receipt for this loop.

## Verification and remaining work

`tests/agent.test.mjs` checks identities/closed schemas/UTF-8 bounds, endpoint key
binding, default-denied permissions, narrow native projections, actual loop
messages, nullable usage, whole-batch denial, tool/turn limits, malformed output,
tool errors, abort and completed-read retention, preflight rejection and sparse
catalog/cursor behavior. No new production dependency or backend source edit.

`tests/agent-endpoint-fixture.mjs` is an authored OpenAI-compatible HTTP protocol
fixture, not an actual remote model or reasoning benchmark. The owned browser
test configures it through Settings, creates a native profile, uses an independent
native writer for stale-run/CAS recovery, executes the actual native runtime read,
checks inert endpoint output, saves/reloads history, aborts a deliberately pending
HTTP request and saves the stopped run. Independent native CLI exports verify
parent identity, exact revision, tool observation, counters and stopped status.
The fixture checks the real forwarded runtime JSON and records all protocol
errors. The full existing browser/resource/training/evaluation regression remains
required. New light/dark/390px profile/result captures are retained after success;
other matrix captures are private temporary artifacts to avoid unrelated churn.

Before treating the full Agent/workflow brief as complete, the backend needs:

1. Canonical versioned Agent/run/workflow records shared by Desktop, pure-C CLI
   and TUI, including explicit migration from these frontend annotations.
2. Native start/status/event APIs, transactionally enforced execute/resource/tool
   permissions, Context budgets/receipts and revocation semantics.
3. Local tool-calling execution or an agreed native planning interface; choosing a
   local model must never fabricate tool support from text generation alone.
4. Durable jobs, bounded logs/events, cancellation acknowledgement, pause/resume,
   reconnect/recovery and clear distinction between client and engine metrics.
5. Node authentication, capability grants, expiry/revocation and resource isolation;
   remote nodes must never inherit this client's native administrative access.
6. Workflow graph/control/state/execution contracts and history, plus real platform,
   installer, keyboard/reader/zoom/high-contrast and realistic endpoint-scale QA.

These are explicit remaining requirements within the active full product goal.
