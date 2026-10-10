# Workflow foundation and native service handoff

Implemented 2026-10-09. Workflows composes actual Fyodor operations with an
ordered vanilla JavaScript editor: text composition, local native generation,
literal/JSON output checks and saved endpoint agents. Definitions and explicitly
saved observations use existing native workflow resources. Execution currently
runs in the Desktop/browser window; this is not a native workflow scheduler,
durable job service, Node Protocol or claim of complete workflow delivery.
Backend sources remain under separate ownership.

## User workflow

1. Open Workflows under Run or through the quick switcher. Create a workflow,
   name each step and choose its operation.
2. Choose Run input or any earlier step's output as that step's input. Optional
   before/after text surrounds it literally. There is no script evaluation or
   template language. A failed check stops the remaining steps.
3. Generation steps share the local model selected for that run, with separate
   per-step sampling settings. Prompts are raw text, without inferred chat
   templates. This selection does not change Chat's global provider/model.
4. Agent steps select a saved profile in the current namespace. Selection reads
   and pins its exact current revision, showing endpoint, model and enabled
   read-only tools. Inspect pinned profile reads that historical revision without
   editing the workflow. Use current saved profile deliberately repins the head
   and makes the workflow dirty, requiring Save before another run.
5. Save the definition, enter the run input, then Run saved steps. The workflow
   head and every referenced agent must still match the saved revisions. Changed
   profiles require explicit review/reselection; a workflow does not silently
   renew obsolete permissions. Agent connections must match current Settings.
6. Inspect the actual outputs/steps and Stop if needed. Save a finished run
   explicitly to keep history; otherwise it remains window-local. Saved history
   is read-only here. Export workflow/run retrieves the current native resource
   head as raw package JSON. Resources supplies generic package interchange.

Move up/down and Remove validate all input links before replacing the draft.
Removing a referenced step, moving a dependency after its consumer, self-links,
forward references and duplicate IDs fail without losing the editor. Rebind
dependent inputs first. Step IDs survive edits/reordering and are independent of
display names; this can support future graph edges without treating positions or
titles as identity. The current editor has no parallel execution, loops, arbitrary
branches, visual canvas, plugins or executable code operation.

## Modules and state

| Module | Responsibility |
| --- | --- |
| `src/workflow-data.js` | Closed schemas, stable identities/input links, deterministic composition, trace consistency, native resource packages and bounded catalog classification. |
| `src/workflow-runner.js` | Whole-run preflight, actual sequential operations, explicit checks, observed metrics, bounded traces and Stop. Injected API; no DOM. |
| `src/workflow-workspace.js` | Editor, profile review, local model choice, native CAS/head guards, progress, unsaved-result protection, history and package export. |
| `src/evaluation-data.js` | Reused native sampling validation, literal/JSON checks and runtime model observations. No second interpretation of check formats. |
| `src/agent-data.js`, `src/agent-runner.js` | Reused saved profile validation, endpoint credential binding and bounded explicit read-only tool execution. |
| `src/api.js` | Existing native authenticated HTTP and external complete-response client. No workflow endpoint or new transport was invented. |
| `src/main.js`, `src/icons.js`, `src/clay.css` | Mount/navigation and semantic responsive clay surfaces only; no execution logic in the shell. |

State is module-local. Inputs update draft state before any repaint; typing does
not rebuild the editor or lose its caret. Busy work disables mutable controls,
keeping Stop active during execution. An unsaved run locks the definition until
save/discard. Mount-generation/active-view checks keep async progress out of other
workspaces. Runs may finish while another view is open and appear on return.
They stop existing when the window/process exits; there is no daemon, recovery
queue or durable in-flight request. Unload warnings protect dirty/running/pending
state where the host supports them. This is not durable draft autosave.

Changing namespace, opening another definition, paging or starting another run
requires clean state. Reload saved workflow deliberately replaces dirty editor
content but refuses to discard an unsaved run. Opening/reloading a definition
clears its launch input. Discard editor and run drops only current window state,
not persisted resources. Conflicts preserve edits/results and never auto-retry
against a newer head. Failed or ambiguous run saves keep the generated UUID for
retry; a write that succeeded but lost its response still needs native inspection
before retry/discard. There is no automatic reconciliation or duplicate creation.

## Native identity, storage and catalog boundary

Native C recognizes `fyodor://workflows/<uuid>` but has no typed workflow-run
child identity. Both definitions and saved run observations therefore use
distinct native workflow UUIDs, with `metadata.workflow_studio` version 1 and
`kind:"definition"|"run"`. A run annotation additionally holds its definition URI
as `workflow`. Its JSON content records the full namespace/URI/revision reference.
This is a frontend logical association, not a native parent/foreign key or job.
Never invent `fyodor://workflows/<uuid>/runs/<uuid>`: the current native parser
does not accept it. Canonical service migration must address these annotations.

New resources use `/resources/put`, native package schema 1 and expected revision
`"0"`. Definition edits use `/resources/update`, retaining opaque native metadata
and provenance. Raw exports use `/resources/export`; exports read the current
head rather than a pinned historical version. Revisions remain exact positive
decimal strings up to signed-64-bit range, never JavaScript Number arithmetic.
Namespace is 1–64 ASCII letters/digits/underscores/dashes. UUIDs are canonical,
lowercase and non-nil. A run cannot reference itself as its own definition.

The current native `/resources/list` title catalog omits metadata. To classify
definitions/runs without guessing from titles, the frontend reads only workflow
identities with `/resources/get` at the catalog's exact revision. It inspects the
annotation and releases the body; only title rows remain in library state. Reads
are administrative native resource reads, not permission-checked Context search.
They never enter a model prompt or create grants.

A scan consumes at most 25 catalog pages (at most 100 rows each), stops after 20
matches, exhaustion or at least 100 workflow lookups, and always finishes the
whole final page. Consequently at most 199 lookups occur; this prevents a mid-page
budget from silently skipping matching records. Next retains the actual native
continuation cursor. Repeated cursors, malformed responses and mismatched exact
revisions reject. Each body is bounded to 1 MiB. Unsupported/unannotated workflow
identities can appear in the definition library but fail closed on opening;
generic Resources inspection/export remains available. Run history filters the
exact logical definition URI and validates namespace/ref/content on opening.

This classification is deliberately bounded but can still read substantial data
at scale. Backend catalog metadata/type/parent filters are the preferred handoff,
not a claim that client scans are a completed scalable native workflow registry.
Deleting a definition through Resources does not cascade these logical run
records. Historical records remain generic resources; this workspace does not
provide delete/retention management or automatically recreate missing definitions.

## Definition and input formats

Closed schema 1 has `kind:"workflow-definition"` and `steps` (1–12). Each step
has exactly `id`, `label`, `operation`, `input` and `config`:

| Field | Values/bounds |
| --- | --- |
| `id` | Unique canonical non-nil UUID, stable across editing/reordering. |
| `label` | Nonblank text up to 256 UTF-8 bytes. |
| `operation` | `text`, `generate`, `check`, `agent`; unknown operations reject. |
| `input` | `{source,prefix,suffix}`; source is `run-input` or an earlier step UUID. Prefix/suffix each up to 2 KiB, no NUL. |
| `text` config | Empty object only; produces literal prefix + source + suffix. |
| `generate` config | Exact native `temperature`, `top_p`, `top_k`, `max_tokens`, `seed` fields, validated by the existing Evaluation helper. Temperature 0–5, Top P >0–1, integer Top K 0–1,000,000, tokens 1–4096, exact nonnegative safe integer seed. |
| `check` config | `{type,expected,pointer}`; `exact`, `contains`, `json` or `pointer`. Expected text up to 8 KiB and pointer up to 512 bytes. Contains requires nonempty expected text. JSON checks reuse the 32-level/10,000-value and own-property rules. Human review is not silently treated as a passed automated check. |
| `agent` config | Exact `{namespace,uri,revision}` saved agent reference; current execution requires the workflow namespace. No instructions, keys or tool grants are copied into the definition. |

Run input is raw UTF-8 up to 8 KiB, no NUL. Composed/native inputs and outputs are
up to 64 KiB. An Agent step still obeys its narrower 8 KiB input bound; oversized
input fails before issuing that endpoint call. There is no implicit truncation or
automatic reduction. Missing referenced output fails instead of substituting an
empty string. Literal `${...}`/markup remains text; it cannot become JavaScript,
shell, HTML, regex, a resource lookup or an arbitrary API function.

## Execution, observations and Stop

Before execution, the controller re-reads the workflow head and rejects a changed
revision. The runner validates the definition/ref/input and preflights every
referenced Agent head/endpoint before any text/generation operation. An Agent
step also rechecks its head immediately before execution. These checks protect
the chosen start/step snapshots; they are not transactional native execute grants
or protection against a revocation occurring after a check.

Generation uses actual `/generate` with exact raw composed prompt/settings and
the run-selected loaded generation model. The runner validates returned model ID,
seed and text. The model snapshot comes from current `/models` and `/runtime`:
process ID, file path/size, format, architecture and compute at start. This is not
immutable model/training lineage or proof of complete GPU execution. Compiler
availability and active compute must not be promoted to universal capabilities.

Checks evaluate the actual chosen input; successful checks pass that same text
onwards. Failure records its reason and schedules no downstream operations. Text
steps compose deterministically. Agent steps reuse the bounded endpoint loop and
explicit saved profile tool choices described in [AGENTS.md](AGENTS.md). Their
inputs may cross to that selected endpoint because the workflow explicitly binds
them; no resource bodies, global Context, filesystem, shell or arbitrary network
tool is silently available. Existing read-only Agent observations are the only
local tool adapters. No grant is created by selecting a profile or running a flow.

Stop aborts an Agent endpoint fetch through the shared AbortSignal and prevents
further scheduling. Native generation has no in-flight cancellation API: a request
already issued may finish, its actual returned output/counters are retained, then
the workflow stops before the next step. Native profile/model reads may also
finish. Stop is not a remote cancellation acknowledgement, engine interrupt,
refund guarantee, pause/resume, reconnect or durable job cancellation.

The configured provider is copied into run-local memory at start so later Settings
edits cannot redirect later stages or alter that run's credentials. Keys are not
intentionally stored in workflow definitions/results. Agent endpoints must match
the copied current connection before that key can be used. Normal Settings
Remember behavior remains separate. Untrusted returned content/errors are not a
general secret-redaction service; input/output and native file observations are
intentionally persisted only when the user explicitly saves the run.

## Run schema and metrics

Closed schema 1 uses `kind:"workflow-run"`, exact `workflow` reference, full saved
`definition`, original `input`, nullable observed `model`, final `status`
(`completed|failed|stopped|limit`), prefix `steps`, `output`, `error`, client epoch-ms
`started_at`/`finished_at`. `output` is the last completed step's output; a failed
or stopped Agent's partial text remains in its step/nested trace. No remaining
step receives a fabricated result.

Each recorded step has `id`, `status`, exact composed `input`, `output`, `error`,
nullable `request_ms`, nullable `prompt_tokens`/`generated_tokens`, and nullable
`agent_run`. Trace order/IDs must match the definition prefix; a noncompleted
step can only be last. Completed text/check outputs must equal their input, and
completed checks must actually pass. A completed workflow requires all steps to
be completed. Agent records must match their pinned ref/input/output/status and
pass the shared Agent-run validator. Their observations are embedded here, not
silently saved as a second standalone Agent-history resource.

Local generation token counters are actual safe nonnegative native values;
missing/invalid counters stay null. Client request duration includes HTTP and
response completion/failure; it is not engine-only time, TTFT or benchmark
performance. Text/check operations have no token/request metrics. Agent request
usage/duration stays in its nested trace with missing usage labelled unknown.
There is no fabricated aggregate throughput or fabricated Context receipt.

Run content is at most 1 MiB; recorded steps are bounded to 512 KiB including
nested Agent traces. Errors are at most 4096 bytes. Timestamps are ordered safe
integers. When a completed operation's trace cannot fit, status becomes explicit
`limit`; that oversized observation is not falsely claimed to have been recorded.
Last previously recorded completed output is retained, and no further work is
scheduled. The UI scrolls complete bounded text rather than truncating it silently.
These are unsigned client observations, not native authenticated execution receipts.

## Verification and remaining native work

`tests/workflow.test.mjs` covers closed identities/schemas, stable input links,
cycles/dangling/dependency edits, executable-extension denial, exact composition,
real API payloads, native counters versus missing metrics, saved Agent binding,
stale/wrong/cross-namespace preflight, failed checks, mismatched native responses,
in-flight native Stop, step-time Agent recheck, nested aborted traces, literal
input, byte/trace caps, exact-revision body classification and complete-page/cursor
bounds. The selected frontend suite now includes 83 checks.

The owned browser/native test exercises public authoring controls, native keyboard
caret retention, link-safe removal/reordering, profile review, independent native
CAS/head changes, real tiny-GGUF generation `a` → `bc`, an exact check, the real
native runtime tool observation forwarded through an authored endpoint protocol
fixture, explicit save/reload, inert text and pending HTTP abort. Separate native
CLI exports and an independent native generation response verify exact references,
input bindings, observations, outputs and counters. Fixture endpoint responses
are authored protocol evidence, not a real remote model or reasoning benchmark.

Tests use an installed Edge/Chrome binary with an owned private profile, loopback
services, temporary database/model copies and synthetic credential. The old Edge
path was unavailable during the first attempt; the harness now selects installed
Chrome as fallback and reports bounded startup diagnostics. No user browser
profile/process is used. Only new Workflow light/dark/390px captures are copied
after success; the earlier Agent evidence remains its historical checkpoint.

Native canonical workflow/run schemas and migration, typed run identities/catalog
filters, transactional execute/resource/Context permissions and receipts, local
tool planning, durable jobs/events/pause/resume/recovery, authenticated node
selection, branches/parallelism and platform/accessibility/realistic-scale QA
remain explicit requirements within the active full product goal. The current
foundation makes composition usable while keeping those contracts unimplemented.
