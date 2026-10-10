// Editor/controller only. Stable input bindings and bounded execution live in
// independent domain/runner modules. Native generic resources supply CAS; this
// view never claims native workflow jobs, node auth or automatic Context grants.
import { backend } from "./api.js";
import { escapeHtml as esc } from "./message.js";
import { writingDownloadName } from "./writing-files.js";
import { checkTypes } from "./evaluation-data.js";
import { agentPage, parseAgent, agentTools } from "./agent-data.js";
import {
  newWorkflow,
  newWorkflowStep,
  validateWorkflow,
  validateWorkflowWiring,
  workflowOperations,
  workflowChecks,
  workflowResource,
  parseWorkflow,
  workflowPage,
} from "./workflow-data.js";
import { runWorkflow } from "./workflow-runner.js";
const state = {
  namespace: "workspace",
  items: [],
  next: "",
  runs: [],
  runNext: "",
  agents: [],
  agentNext: "",
  previews: new Map(),
  models: [],
  modelId: 0,
  resource: null,
  value: null,
  revision: "0",
  dirty: false,
  busy: false,
  running: false,
  controller: null,
  active: null,
  input: "",
  pending: null,
  pendingResource: null,
  savedRun: null,
  error: "",
  notice: "",
};
let mount = 0,
  repaint = () => {};
window.addEventListener("beforeunload", (event) => {
  if (state.dirty || state.pending || state.running) {
    event.preventDefault();
    event.returnValue = "";
  }
});
const name = (model) => model.path.split(/[\\/]/).at(-1);
const numericFields = [
  ["max_tokens", "New tokens", 1, 4096, 1],
  ["temperature", "Temperature", 0, 5, 0.1],
  ["top_p", "Top P", 0.01, 1, 0.01],
  ["top_k", "Top K", 0, 1000000, 1],
  ["seed", "Seed", 0, Number.MAX_SAFE_INTEGER, 1],
];
function stepEditor(step, index, steps) {
  const preview = state.previews.get(step.id);
  return `<article class="workflow-step" data-workflow-step="${step.id}"><div class="workflow-step-heading"><b>${index + 1}</b><label>Step name<input data-workflow-field="label" value="${esc(step.label)}" maxlength="256"></label><div class="resource-toolbar"><button data-workflow="up" ${index === 0 ? "disabled" : ""}>Move up</button><button data-workflow="down" ${index === steps.length - 1 ? "disabled" : ""}>Move down</button><button data-workflow="remove">Remove</button></div></div>
    <div class="config-grid"><label>Operation<select data-workflow-field="operation">${Object.entries(
      workflowOperations,
    )
      .map(
        ([key, label]) =>
          `<option value="${key}" ${step.operation === key ? "selected" : ""}>${label}</option>`,
      )
      .join(
        "",
      )}</select></label><label>Input from<select data-workflow-field="source"><option value="run-input" ${step.input.source === "run-input" ? "selected" : ""}>Run input</option>${steps
      .slice(0, index)
      .map(
        (source, i) =>
          `<option value="${source.id}" ${step.input.source === source.id ? "selected" : ""}>${i + 1}. ${esc(source.label)}</option>`,
      )
      .join("")}</select></label></div>
    <div class="config-grid"><label>Text before input<textarea data-workflow-field="prefix" rows="2">${esc(step.input.prefix)}</textarea></label><label>Text after input<textarea data-workflow-field="suffix" rows="2">${esc(step.input.suffix)}</textarea></label></div>
    ${step.operation === "generate" ? `<div class="config-grid workflow-sampling">${numericFields.map(([key, label, min, max, increment]) => `<label>${label}<input type="number" data-workflow-setting="${key}" value="${step.config[key]}" min="${min}" max="${max}" step="${increment}"></label>`).join("")}</div>` : ""}
    ${step.operation === "check" ? `<div class="config-grid"><label>Check<select data-workflow-setting="type">${workflowChecks.map((key) => `<option value="${key}" ${step.config.type === key ? "selected" : ""}>${checkTypes[key]}</option>`).join("")}</select></label>${step.config.type === "pointer" ? `<label>JSON pointer<input data-workflow-setting="pointer" value="${esc(step.config.pointer)}"></label>` : ""}</div>${step.config.type !== "json" ? `<label>Expected ${step.config.type === "pointer" ? "JSON value" : "text"}<textarea data-workflow-setting="expected" rows="2">${esc(step.config.expected)}</textarea></label>` : ""}<p class="fine-print">A failed check stops the remaining steps.</p>` : ""}
    ${step.operation === "agent" ? `<label>Saved agent<select data-workflow-field="agent"><option value="">Select a profile</option>${state.agents.map((agent) => `<option value="${esc(agent.uri)}" ${step.config.uri === agent.uri ? "selected" : ""}>${esc(agent.title || "Untitled agent")} · Revision ${esc(agent.revision)}</option>`).join("")}</select></label><div class="resource-toolbar"><button data-workflow="inspect-agent" ${step.config.uri ? "" : "disabled"}>Inspect pinned profile</button><button data-workflow="pin-agent" ${step.config.uri ? "" : "disabled"}>Use current saved profile</button></div><p class="fine-print">${step.config.uri ? `Pinned revision ${esc(step.config.revision)} · ${esc(step.config.uri)}` : "Choose the profile whose tools this step may use."}</p>${preview ? `<details open><summary>Selected profile</summary><p>${esc(preview.title)} · ${esc(preview.value.target.model)}</p><p class="fine-print">${esc(preview.value.target.endpoint)}</p><p>${preview.value.tools.length ? preview.value.tools.map((tool) => esc(agentTools[tool])).join(" · ") : "No tools enabled"}</p></details>` : ""}` : ""}</article>`;
}
function resultsMarkup(run) {
  return `<section class="workflow-results" aria-label="Workflow result"><div class="workflow-step-heading"><h3>${state.running ? "Current run" : "Run result"}</h3><span class="provider-tag">${esc(state.running ? "running" : run.status)}</span></div><p class="fine-print">Definition revision ${esc(run.workflow.revision)}${run.model ? " · " + esc(name({ path: run.model.path })) + " · " + esc(run.model.compute_at_start || "Compute not reported") : ""}</p><h4>Last completed output</h4><pre id="workflow-output">${esc(run.output || "No completed output.")}</pre>${run.error ? `<p class="danger-action">${esc(run.error)}</p>` : ""}<details class="workflow-trace"><summary>${run.steps.length} recorded steps</summary><ol>${run.steps.map((record, index) => `<li><b>${esc(run.definition.steps[index].label)} · ${esc(record.status)}</b>${record.request_ms !== null ? `<small>${Math.round(record.request_ms)} ms request · ${record.generated_tokens === null ? "Generated tokens not reported" : record.generated_tokens + " generated tokens"} · ${record.prompt_tokens === null ? "Prompt tokens not reported" : record.prompt_tokens + " prompt tokens"}</small>` : ""}<details><summary>Input and output</summary><h4>Input</h4><pre>${esc(record.input)}</pre><h4>Output</h4><pre>${esc(record.output)}</pre></details>${record.error ? `<p class="danger-action">${esc(record.error)}</p>` : ""}${record.agent_run ? `<details><summary>Agent observations · ${record.agent_run.steps.length} steps</summary><ul>${record.agent_run.steps.map((item) => `<li><b>${esc(item.tool ? agentTools[item.tool] : "Model request")} · ${esc(item.status)}</b>${item.kind === "model" ? `<small>${item.tokens === null ? "Tokens not reported" : item.tokens + " reported tokens"}</small>` : ""}<pre>${esc(item.text || item.error)}</pre></li>`).join("")}</ul></details>` : ""}</li>`).join("")}</ol></details></section>`;
}
export function workflowWorkspace(root, getProvider, preferredModel) {
  const generation = ++mount,
    current = () =>
      generation === mount && document.body.dataset.view === "workflows";
  const requireClean = () => {
    if (state.dirty || state.pending || state.running)
      throw Error("Save or discard the current changes and run first.");
  };
  async function load(after = "") {
    const page = await workflowPage(
      backend,
      state.namespace,
      "definition",
      null,
      after,
    );
    state.items = page.resources;
    state.next = page.next;
  }
  async function loadRuns(after = "") {
    if (state.revision === "0" || !state.resource?.uri) {
      state.runs = [];
      state.runNext = "";
      return;
    }
    const page = await workflowPage(
      backend,
      state.namespace,
      "run",
      state.resource.uri,
      after,
    );
    state.runs = page.resources;
    state.runNext = page.next;
  }
  async function choices(after = "") {
    const page = await agentPage(
      backend,
      state.namespace,
      "definition",
      null,
      after,
    );
    state.agents = page.resources;
    state.agentNext = page.next;
    state.models = (await backend.models()).models.filter(
      (model) => model.generation_supported,
    );
    if (
      !state.modelId &&
      state.models.some((model) => model.id === preferredModel)
    )
      state.modelId = preferredModel;
    if (
      state.modelId &&
      !state.models.some((model) => model.id === state.modelId)
    )
      state.modelId = 0;
  }
  async function work(task) {
    if (state.busy) return;
    state.busy = true;
    state.error = "";
    state.notice = "";
    paint();
    try {
      await task();
    } catch (error) {
      state.error = error.message;
    } finally {
      state.busy = false;
      repaint();
    }
  }
  function dirty() {
    state.dirty = true;
    root.querySelector("#workflow-save-state").textContent = "Unsaved changes";
    root.querySelector("[data-workflow=run]").disabled = true;
  }
  function paint() {
    if (!current()) return;
    const d = state.value,
      run = state.pending || state.savedRun?.value;
    root.innerHTML = `<section class="resource-workspace workflow-workspace"><header class="view-heading"><div><p class="eyebrow">OPERATIONS IN ORDER</p><h1>Workflows</h1><p class="muted">Connect steps, check their outputs and keep the results.</p></div></header><div class="resource-toolbar"><label>Namespace<input id="workflow-namespace" maxlength="64" value="${esc(state.namespace)}"></label><button data-workflow="refresh">Refresh library</button><button data-workflow="new">New workflow</button><button data-workflow="discard">Discard editor and run</button></div><p id="workflow-status" role="status" aria-live="polite" class="${state.error ? "danger-action" : "muted"}">${esc(state.error || state.notice || (state.running ? "Running " + (d.steps.find((step) => step.id === state.active)?.label || "workflow") + "…" : state.busy ? "Working…" : ""))}</p>
      <div class="resource-columns"><section aria-label="Workflow library"><ul class="resource-list">${state.items.map((item) => `<li><button data-workflow-uri="${esc(item.uri)}" aria-pressed="${state.resource?.uri === item.uri}"><b>${esc(item.title || "Untitled workflow")}</b><small>Revision ${esc(item.revision)}</small></button></li>`).join("") || '<li class="muted">No saved workflows on this page.</li>'}</ul><button data-workflow="next" ${state.next ? "" : "disabled"}>Next page</button></section><section class="panel workflow-editor" aria-label="Workflow definition">${
        d
          ? `<div class="workflow-step-heading"><h2>${esc(state.resource.title)}</h2><span id="workflow-save-state" class="fine-print">${state.dirty ? "Unsaved changes" : "Revision " + esc(state.revision)}</span></div><label>Title<input id="workflow-title" maxlength="1024" value="${esc(state.resource.title)}"></label><div class="resource-toolbar"><button class="primary" data-workflow="save">Save workflow</button>${state.revision !== "0" ? '<button data-workflow="reload">Reload saved workflow</button><button data-workflow="export">Export workflow</button>' : ""}<button data-workflow="choices">Refresh agents and models</button><button data-workflow="agents-next" ${state.agentNext ? "" : "disabled"}>Next agent page</button></div><div class="workflow-step-list">${d.steps.map((step, index) => stepEditor(step, index, d.steps)).join("")}</div><div class="resource-toolbar"><label>Add operation<select id="workflow-new-operation">${Object.entries(
              workflowOperations,
            )
              .map(([key, label]) => `<option value="${key}">${label}</option>`)
              .join(
                "",
              )}</select></label><button data-workflow="add" ${d.steps.length >= 12 ? "disabled" : ""}>Add step</button></div>`
          : '<h2>Create a workflow</h2><p class="muted">Choose an operation for each step and where its input comes from.</p>'
      }</section></div>
      ${d ? `<div class="workflow-run-grid"><section class="panel workflow-launch"><h2>Run saved workflow</h2><label>Run input<textarea id="workflow-input" rows="3">${esc(state.input)}</textarea></label>${d.steps.some((step) => step.operation === "generate") ? `<label>Local model for generation<select id="workflow-model"><option value="0" ${state.modelId === 0 ? "selected" : ""}>Choose a model</option>${state.models.map((model) => `<option value="${model.id}" ${model.id === state.modelId ? "selected" : ""}>${esc(name(model))}</option>`).join("")}</select></label>` : ""}<div class="resource-toolbar"><button class="primary" data-workflow="run" ${state.dirty || state.revision === "0" || state.pending ? "disabled" : ""}>Run saved steps</button><button data-workflow="stop" ${state.running ? "" : "disabled"}>${state.controller?.signal.aborted ? "Stop requested" : "Stop"}</button></div>${state.pending ? '<div class="resource-toolbar"><button data-workflow="save-run">Save run</button><button data-workflow="discard-run">Discard run</button></div>' : ""}${run ? resultsMarkup(run) : ""}${state.savedRun && !state.pending ? '<button data-workflow="export-run">Export saved run</button>' : ""}</section><section class="panel workflow-history"><h2>Run history</h2><div class="resource-toolbar"><button data-workflow="history">Refresh history</button><button data-workflow="history-next" ${state.runNext ? "" : "disabled"}>Next page</button></div><ul class="resource-list">${state.runs.map((item) => `<li><button data-workflow-run-uri="${esc(item.uri)}" aria-pressed="${state.savedRun?.resource.uri === item.uri}"><b>${esc(item.title || "Workflow run")}</b><small>Revision ${esc(item.revision)}</small></button></li>`).join("") || '<li class="muted">No saved runs on this page.</li>'}</ul></section></div>` : ""}</section>`;
    if (state.busy)
      root
        .querySelectorAll("input,textarea,select,button")
        .forEach((control) => {
          control.disabled =
            control.dataset.workflow !== "stop" ||
            !state.running ||
            state.controller?.signal.aborted;
        });
    if (state.pending)
      root
        .querySelectorAll(
          ".workflow-editor input,.workflow-editor textarea,.workflow-editor select,.workflow-editor button",
        )
        .forEach((control) => {
          control.disabled = true;
        });
    root
      .querySelectorAll("[data-workflow]")
      .forEach((button) =>
        button.addEventListener("click", () =>
          action(
            button.dataset.workflow,
            button.closest("[data-workflow-step]")?.dataset.workflowStep,
          ),
        ),
      );
    root.querySelectorAll("[data-workflow-uri]").forEach((button) =>
      button.addEventListener(
        "click",
        () =>
          void work(async () => {
            requireClean();
            await open(button.dataset.workflowUri);
            await loadRuns();
          }),
      ),
    );
    root.querySelectorAll("[data-workflow-run-uri]").forEach((button) =>
      button.addEventListener(
        "click",
        () =>
          void work(async () => {
            requireClean();
            const loaded = await backend.resource(
                state.namespace,
                button.dataset.workflowRunUri,
              ),
              value = parseWorkflow(loaded.resource);
            if (
              value.kind !== "workflow-run" ||
              value.workflow.namespace !== state.namespace ||
              value.workflow.uri !== state.resource.uri
            )
              throw Error("Run belongs to another workflow or namespace.");
            state.savedRun = { resource: loaded.resource, value };
          }),
      ),
    );
    root
      .querySelector("#workflow-namespace")
      .addEventListener("change", (event) => {
        const namespace = event.target.value;
        void work(async () => {
          requireClean();
          if (!/^[A-Za-z0-9_-]{1,64}$/.test(namespace))
            throw Error("Use 1–64 letters, digits, underscores or dashes.");
          clear();
          state.namespace = namespace;
          await load();
          await choices();
        });
      });
    if (!d) return;
    root.querySelector("#workflow-title").addEventListener("input", (event) => {
      state.resource.title = event.target.value;
      dirty();
    });
    root.querySelector("#workflow-input").addEventListener("input", (event) => {
      state.input = event.target.value;
    });
    root
      .querySelector("#workflow-model")
      ?.addEventListener("change", (event) => {
        state.modelId = Number(event.target.value);
      });
    root.querySelectorAll("[data-workflow-step]").forEach((article) => {
      const step = d.steps.find(
        (item) => item.id === article.dataset.workflowStep,
      );
      article.querySelectorAll("[data-workflow-field]").forEach((control) => {
        const field = control.dataset.workflowField;
        control.addEventListener(
          control.tagName === "SELECT" ? "change" : "input",
          () => {
            const value = control.value;
            if (field === "operation")
              void work(async () => {
                const fresh = newWorkflowStep(value);
                if (step.label === workflowOperations[step.operation])
                  step.label = fresh.label;
                step.operation = value;
                step.config = fresh.config;
                if (value === "agent") step.config.namespace = state.namespace;
                state.previews.delete(step.id);
                state.dirty = true;
              });
            else if (field === "agent")
              void work(async () => {
                if (value) await pinAgent(step, value);
                else {
                  step.config = {
                    namespace: state.namespace,
                    uri: "",
                    revision: "0",
                  };
                  state.previews.delete(step.id);
                  state.dirty = true;
                }
              });
            else {
              if (field === "label") step.label = value;
              else step.input[field] = value;
              dirty();
            }
          },
        );
      });
      article.querySelectorAll("[data-workflow-setting]").forEach((control) =>
        control.addEventListener(
          control.tagName === "SELECT" ? "change" : "input",
          () => {
            const key = control.dataset.workflowSetting;
            step.config[key] =
              step.operation === "generate"
                ? Number(control.value)
                : control.value;
            dirty();
            if (key === "type") paint();
          },
        ),
      );
    });
  }
  function clear() {
    Object.assign(state, {
      resource: null,
      value: null,
      revision: "0",
      dirty: false,
      input: "",
      pending: null,
      pendingResource: null,
      savedRun: null,
      runs: [],
      runNext: "",
      active: null,
    });
    state.previews.clear();
  }
  async function open(uri) {
    const loaded = await backend.resource(state.namespace, uri),
      value = parseWorkflow(loaded.resource);
    if (value.kind !== "workflow-definition")
      throw Error("Select a workflow definition.");
    state.resource = loaded.resource;
    state.value = value;
    state.revision = loaded.revision;
    state.dirty = false;
    state.savedRun = null;
    state.input = "";
    state.previews.clear();
  }
  async function pinAgent(step, uri) {
    const head = await backend.resource(state.namespace, uri),
      value = parseAgent(head.resource);
    if (head.resource.uri !== uri || value.kind !== "agent-definition")
      throw Error("Choose a saved agent profile.");
    step.config = { namespace: state.namespace, uri, revision: head.revision };
    state.previews.set(step.id, { title: head.resource.title, value });
    state.dirty = true;
  }
  async function download(resource) {
    const text = await backend.exportResource(state.namespace, resource.uri),
      url = URL.createObjectURL(new Blob([text], { type: "application/json" })),
      link = document.createElement("a");
    link.href = url;
    link.download = writingDownloadName(resource.title, "txt").replace(
      /\.txt$/,
      ".json",
    );
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    state.notice = "Saved package exported.";
  }
  async function execute() {
    requireClean();
    if (state.revision === "0")
      throw Error("Save the workflow before running it.");
    const head = await backend.resource(state.namespace, state.resource.uri);
    if (
      head.resource?.uri !== state.resource.uri ||
      head.revision !== state.revision
    )
      throw Error(
        "Workflow changed in another client. Reload it before running.",
      );
    const definition = parseWorkflow(head.resource);
    state.controller = new AbortController();
    state.running = true;
    state.savedRun = null;
    repaint();
    try {
      state.pending = await runWorkflow({
        definition,
        reference: {
          namespace: state.namespace,
          uri: state.resource.uri,
          revision: state.revision,
        },
        input: state.input,
        provider: getProvider(),
        modelId: state.modelId,
        api: backend,
        signal: state.controller.signal,
        onUpdate: (value, active) => {
          state.pending = value;
          state.active = active;
          repaint();
        },
      });
      state.notice = `Workflow ${state.pending.status}. Save its run to keep the results.`;
    } finally {
      state.running = false;
      state.controller = null;
      state.active = null;
    }
  }
  function action(name, id) {
    if (name === "stop") {
      state.controller?.abort();
      repaint();
      return;
    }
    const addedOperation =
      name === "add"
        ? root.querySelector("#workflow-new-operation").value
        : null;
    void work(async () => {
      if (name === "discard") clear();
      else if (name === "discard-run") {
        state.pending = null;
        state.pendingResource = null;
        state.savedRun = null;
      } else if (name === "refresh" || name === "next") {
        requireClean();
        await load(name === "next" ? state.next : "");
      } else if (name === "new") {
        requireClean();
        clear();
        state.value = newWorkflow();
        state.resource = { uri: null, title: "Untitled workflow" };
        state.dirty = true;
        await choices();
      } else if (name === "choices" || name === "agents-next") {
        if (state.pending) throw Error("Save or discard the run first.");
        await choices(name === "agents-next" ? state.agentNext : "");
      } else if (name === "add") {
        if (state.value.steps.length >= 12)
          throw Error("Use at most 12 steps.");
        const item = newWorkflowStep(
          addedOperation,
          state.value.steps.at(-1)?.id || "run-input",
        );
        if (addedOperation === "agent") item.config.namespace = state.namespace;
        state.value.steps.push(item);
        state.dirty = true;
      } else if (["up", "down", "remove"].includes(name)) {
        const steps = structuredClone(state.value.steps),
          index = steps.findIndex((step) => step.id === id);
        if (name === "remove") steps.splice(index, 1);
        else {
          const next = index + (name === "up" ? -1 : 1);
          if (next < 0 || next >= steps.length) return;
          [steps[index], steps[next]] = [steps[next], steps[index]];
        }
        validateWorkflowWiring(steps);
        state.value.steps = steps;
        state.previews.delete(id);
        state.dirty = true;
      } else if (name === "inspect-agent") {
        const step = state.value.steps.find((item) => item.id === id),
          loaded = await backend.resource(
            step.config.namespace,
            step.config.uri,
            step.config.revision,
          ),
          value = parseAgent(loaded.resource);
        if (
          loaded.resource.uri !== step.config.uri ||
          loaded.revision !== step.config.revision ||
          value.kind !== "agent-definition"
        )
          throw Error("Pinned agent revision is unavailable.");
        state.previews.set(step.id, { title: loaded.resource.title, value });
      } else if (name === "pin-agent") {
        const step = state.value.steps.find((item) => item.id === id);
        await pinAgent(step, step.config.uri);
      } else if (name === "save") {
        if (state.pending) throw Error("Save or discard the run first.");
        const value = validateWorkflow(state.value),
          resource = state.resource.uri
            ? { ...state.resource, content: JSON.stringify(value) }
            : workflowResource("definition", state.resource.title, value);
        state.resource = resource;
        const saved = await (
          state.revision === "0" ? backend.saveResource : backend.updateResource
        )(state.namespace, resource, state.revision);
        state.value = value;
        state.revision = saved.revision;
        state.dirty = false;
        await load();
        await loadRuns();
        state.notice = "Workflow saved.";
      } else if (name === "reload") {
        if (state.pending) throw Error("Save or discard the run first.");
        await open(state.resource.uri);
        await load();
        await loadRuns();
        state.notice = "Saved workflow loaded.";
      } else if (name === "run") await execute();
      else if (name === "save-run") {
        if (!state.pending || state.running)
          throw Error("Finish the workflow before saving.");
        state.pendingResource ??= workflowResource(
          "run",
          `${state.resource.title} · ${state.pending.status} · ${new Date(state.pending.started_at).toLocaleString()}`,
          state.pending,
        );
        await backend.saveResource(state.namespace, state.pendingResource, "0");
        state.savedRun = {
          resource: state.pendingResource,
          value: state.pending,
        };
        state.pending = null;
        state.pendingResource = null;
        await loadRuns();
        state.notice = "Workflow run saved.";
      } else if (name === "history" || name === "history-next") {
        requireClean();
        await loadRuns(name === "history-next" ? state.runNext : "");
      } else if (name === "export" || name === "export-run") {
        requireClean();
        await download(
          name === "export" ? state.resource : state.savedRun.resource,
        );
      }
    });
  }
  repaint = paint;
  paint();
  if (!state.busy)
    void work(async () => {
      await load();
      await choices();
      await loadRuns();
    });
}
