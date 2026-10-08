// Presentation owns drafts and unsaved results. Native typed resources own CAS
// and history; the bounded endpoint runner owns execution. These client tool
// choices are not native ACL grants, jobs, node permissions or Context access.
import { backend } from "./api.js";
import { escapeHtml as esc } from "./message.js";
import { writingDownloadName } from "./writing-files.js";
import {
  agentTools,
  newAgent,
  validateAgent,
  agentResource,
  parseAgent,
  agentPage,
} from "./agent-data.js";
import { runAgent } from "./agent-runner.js";

const state = {
  namespace: "workspace",
  items: [],
  next: "",
  runs: [],
  runNext: "",
  resource: null,
  value: null,
  revision: "0",
  dirty: false,
  busy: false,
  running: false,
  controller: null,
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
const fields = [
  ["limits", "turns", "Model turns", 1, 12, 1],
  ["limits", "tool_calls", "Tool calls", 0, 48, 1],
  ["sampling", "max_tokens", "Tokens per turn", 1, 4096, 1],
  ["sampling", "temperature", "Temperature", 0, 5, 0.1],
  ["sampling", "top_p", "Top P", 0.01, 1, 0.01],
];
const resultMarkup = (
  run,
  live,
) => `<section class="agent-result" aria-label="Run result"><div class="agent-result-heading"><h3>${live ? "Current run" : "Saved run"}</h3><span class="provider-tag">${esc(state.running && live ? "running" : run.status)}</span></div>
  <p class="fine-print">${esc(run.definition.target.model)} · Definition revision ${esc(run.agent.revision)}</p>
  <h4>Input</h4><pre>${esc(run.input)}</pre><h4>Assistant output</h4><pre id="agent-output">${esc(run.output || (state.running ? "Waiting for assistant…" : "No assistant text."))}</pre>
  ${run.error ? `<p class="danger-action">${esc(run.error)}</p>` : ""}
  <details class="agent-steps" ${state.running ? "open" : ""}><summary>${run.steps.length} recorded steps</summary><ol>${run.steps.map((item) => `<li><b>${esc(item.kind === "tool" ? agentTools[item.tool] : "Model request")} · ${esc(item.status)}</b>${item.kind === "model" ? `<small>${item.request_ms === null ? "Duration not reported" : Math.round(item.request_ms) + " ms request"} · ${item.tokens === null ? "Tokens not reported" : item.tokens + " reported tokens"}</small>` : ""}${item.text ? `<pre>${esc(item.text)}</pre>` : ""}${item.error ? `<p class="danger-action">${esc(item.error)}</p>` : ""}</li>`).join("")}</ol></details></section>`;

export function agentWorkspace(root, getProvider) {
  const generation = ++mount;
  const current = () =>
    generation === mount && document.body.dataset.view === "agents";
  const requireClean = () => {
    if (state.dirty || state.pending || state.running)
      throw Error("Save or discard the current changes and run first.");
  };
  async function load(after = "") {
    const page = await agentPage(
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
    if (!state.resource?.uri || state.revision === "0") {
      state.runs = [];
      state.runNext = "";
      return;
    }
    const page = await agentPage(
      backend,
      state.namespace,
      "run",
      state.resource.uri,
      after,
    );
    state.runs = page.resources;
    state.runNext = page.next;
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
    root.querySelector("#agent-save-state").textContent = "Unsaved changes";
    root.querySelector("[data-agent=run]").disabled = true;
    state.notice = "";
  }
  function paint() {
    if (!current()) return;
    const d = state.value;
    root.innerHTML = `<section class="resource-workspace agent-workspace"><header class="view-heading"><div><p class="eyebrow">PURPOSE & PERMISSION</p><h1>Agents</h1><p class="muted">Give each agent a purpose, choose its tools and keep its run history.</p></div></header>
      <div class="resource-toolbar"><label>Namespace<input id="agent-namespace" maxlength="64" value="${esc(state.namespace)}"></label><button data-agent="refresh">Refresh library</button><button data-agent="new">New agent</button><button data-agent="discard">Discard editor and run</button></div>
      <p id="agent-status" role="status" aria-live="polite" class="${state.error ? "danger-action" : "muted"}">${esc(state.error || state.notice || (state.running ? "Running…" : state.busy ? "Working…" : ""))}</p>
      <div class="resource-columns"><section aria-label="Agent library"><ul class="resource-list">${state.items.map((item) => `<li><button data-agent-uri="${esc(item.uri)}" aria-pressed="${state.resource?.uri === item.uri}"><b>${esc(item.title || "Untitled agent")}</b><small>Revision ${esc(item.revision)}</small></button></li>`).join("") || '<li class="muted">No saved agents on this page.</li>'}</ul><button data-agent="next" ${state.next ? "" : "disabled"}>Next page</button></section>
      <section class="panel agent-editor" aria-label="Agent profile">${
        d
          ? `<div class="agent-result-heading"><h2>${esc(state.resource.title)}</h2><span id="agent-save-state" class="fine-print">${state.dirty ? "Unsaved changes" : "Revision " + esc(state.revision)}</span></div>
        <label>Title<input id="agent-title" maxlength="1024" value="${esc(state.resource.title)}"></label><label>Instructions<textarea id="agent-instructions" rows="6" placeholder="Describe the agent’s purpose and how it should respond.">${esc(d.instructions)}</textarea></label>
        <div class="config-grid"><label>Endpoint<input id="agent-endpoint" type="url" value="${esc(d.target.endpoint)}" placeholder="https://your-endpoint/v1"></label><label>Model name<input id="agent-model" maxlength="256" value="${esc(d.target.model)}"></label></div>
        <h3>Allowed tools</h3><div class="agent-tool-list">${Object.entries(
          agentTools,
        )
          .map(
            ([name, label]) =>
              `<label class="check"><input type="checkbox" data-agent-tool="${name}" ${d.tools.includes(name) ? "checked" : ""}><span>${label}</span></label>`,
          )
          .join("")}</div>
        <h3>Execution limits</h3><div class="config-grid agent-limits">${fields.map(([group, key, label, min, max, step]) => `<label>${label}<input type="number" data-agent-group="${group}" data-agent-field="${key}" value="${d[group][key]}" min="${min}" max="${max}" step="${step}"></label>`).join("")}</div>
        <div class="resource-toolbar"><button class="primary" data-agent="save">Save profile</button><button data-agent="connection">Use configured connection</button>${state.revision !== "0" ? '<button data-agent="reload">Reload saved profile</button><button data-agent="export">Export profile</button>' : ""}</div>
        <p class="fine-print">This agent uses the matching connection in Settings. Its tools can read runtime status, model names and local time.</p>`
          : '<h2>Create an agent</h2><p class="muted">Save a profile to start a run.</p>'
      }</section></div>
      ${
        d
          ? `<div class="agent-run-grid"><section class="panel agent-launch"><h2>Start a run</h2><label>Goal<textarea id="agent-input" rows="4" placeholder="What would you like this agent to do?">${esc(state.input)}</textarea></label><div class="resource-toolbar"><button class="primary" data-agent="run" ${state.dirty || state.revision === "0" || state.pending ? "disabled" : ""}>Run saved profile</button><button data-agent="stop" ${state.running ? "" : "disabled"}>Stop</button></div>
        ${state.pending ? `<div class="resource-toolbar"><button data-agent="save-run">Save run</button><button data-agent="discard-run">Discard run</button></div>${resultMarkup(state.pending, true)}` : state.savedRun ? `${resultMarkup(state.savedRun.value, false)}<button data-agent="export-run">Export saved run</button>` : ""}</section>
      <section class="panel agent-history"><h2>Run history</h2><div class="resource-toolbar"><button data-agent="history">Refresh history</button><button data-agent="history-next" ${state.runNext ? "" : "disabled"}>Next page</button></div><ul class="resource-list">${state.runs.map((item) => `<li><button data-agent-run-uri="${esc(item.uri)}" aria-pressed="${state.savedRun?.resource.uri === item.uri}"><b>${esc(item.title || "Agent run")}</b><small>Revision ${esc(item.revision)}</small></button></li>`).join("") || '<li class="muted">No saved runs on this page.</li>'}</ul></section></div>`
          : ""
      }</section>`;
    if (state.busy)
      root.querySelectorAll("input,textarea,button").forEach((control) => {
        control.disabled = control.dataset.agent !== "stop" || !state.running;
      });
    if (state.pending)
      root
        .querySelectorAll(
          ".agent-editor input,.agent-editor textarea,.agent-editor button",
        )
        .forEach((control) => {
          control.disabled = true;
        });
    root
      .querySelector("#agent-namespace")
      .addEventListener("change", (event) => {
        const value = event.target.value;
        void work(async () => {
          requireClean();
          if (!/^[A-Za-z0-9_-]{1,64}$/.test(value))
            throw Error("Use 1–64 letters, digits, underscores or dashes.");
          clear();
          state.namespace = value;
          await load();
        });
      });
    root
      .querySelectorAll("[data-agent]")
      .forEach((button) =>
        button.addEventListener("click", () => action(button.dataset.agent)),
      );
    root.querySelectorAll("[data-agent-uri]").forEach((button) =>
      button.addEventListener(
        "click",
        () =>
          void work(async () => {
            requireClean();
            await open(button.dataset.agentUri);
            await loadRuns();
          }),
      ),
    );
    root.querySelectorAll("[data-agent-run-uri]").forEach((button) =>
      button.addEventListener(
        "click",
        () =>
          void work(async () => {
            requireClean();
            const loaded = await backend.resource(
              state.namespace,
              button.dataset.agentRunUri,
            );
            const value = parseAgent(loaded.resource);
            if (
              value.agent.namespace !== state.namespace ||
              value.agent.uri !== state.resource.uri
            )
              throw Error("Run belongs to another agent or namespace.");
            state.savedRun = { resource: loaded.resource, value };
          }),
      ),
    );
    if (!d) return;
    for (const [selector, update] of [
      [
        "#agent-title",
        (value) => {
          state.resource.title = value;
        },
      ],
      [
        "#agent-instructions",
        (value) => {
          d.instructions = value;
        },
      ],
      [
        "#agent-endpoint",
        (value) => {
          d.target.endpoint = value;
        },
      ],
      [
        "#agent-model",
        (value) => {
          d.target.model = value;
        },
      ],
    ])
      root.querySelector(selector).addEventListener("input", (event) => {
        update(event.target.value);
        dirty();
      });
    root.querySelectorAll("[data-agent-tool]").forEach((control) =>
      control.addEventListener("change", () => {
        d.tools = Array.from(
          root.querySelectorAll("[data-agent-tool]:checked"),
          (item) => item.dataset.agentTool,
        );
        dirty();
      }),
    );
    root.querySelectorAll("[data-agent-field]").forEach((control) =>
      control.addEventListener("input", () => {
        d[control.dataset.agentGroup][control.dataset.agentField] = Number(
          control.value,
        );
        dirty();
      }),
    );
    root.querySelector("#agent-input").addEventListener("input", (event) => {
      state.input = event.target.value;
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
    });
  }
  async function open(uri) {
    const loaded = await backend.resource(state.namespace, uri),
      value = parseAgent(loaded.resource);
    if (value.kind !== "agent-definition")
      throw Error("Select an agent profile.");
    Object.assign(state, {
      resource: loaded.resource,
      value,
      revision: loaded.revision,
      dirty: false,
      savedRun: null,
      input: "",
    });
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
      throw Error("Save the profile before running it.");
    // Re-read head before a run; another client may have revoked a tool choice.
    // This protects the start snapshot, not subsequent native ACL revocations.
    const head = await backend.resource(state.namespace, state.resource.uri);
    if (head.revision !== state.revision)
      throw Error(
        "Profile changed in another client. Reload the saved profile before running.",
      );
    const definition = parseAgent(head.resource);
    state.savedRun = null;
    state.controller = new AbortController();
    state.running = true;
    repaint();
    try {
      state.pending = await runAgent({
        definition,
        reference: {
          namespace: state.namespace,
          uri: state.resource.uri,
          revision: state.revision,
        },
        input: state.input,
        provider: getProvider(),
        api: backend,
        signal: state.controller.signal,
        onUpdate: (value) => {
          state.pending = value;
          repaint();
        },
      });
      state.notice = `Run ${state.pending.status}. Save it to keep its history.`;
    } finally {
      state.running = false;
      state.controller = null;
    }
  }
  function action(name) {
    if (name === "stop") {
      state.controller?.abort();
      return;
    }
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
        state.resource = { uri: null, title: "Untitled agent" };
        state.value = newAgent(getProvider());
        state.dirty = true;
      } else if (name === "connection") {
        if (state.pending) throw Error("Save or discard the run first.");
        const provider = getProvider();
        if (provider.type !== "openai")
          throw Error(
            "Configure an OpenAI-compatible connection in Settings first.",
          );
        state.value.target.endpoint = provider.endpoint;
        state.value.target.model = provider.model;
        state.dirty = true;
      } else if (name === "save") {
        if (state.pending)
          throw Error("Save or discard the run before editing its profile.");
        const value = validateAgent(state.value);
        const resource = state.resource.uri
          ? { ...state.resource, content: JSON.stringify(value) }
          : agentResource("definition", state.resource.title, value);
        state.resource = resource;
        const saved = await (
          state.revision === "0" ? backend.saveResource : backend.updateResource
        )(state.namespace, resource, state.revision);
        state.value = value;
        state.revision = saved.revision;
        state.dirty = false;
        state.notice = "Agent profile saved.";
        await load();
        await loadRuns();
      } else if (name === "reload") {
        if (state.pending)
          throw Error("Save or discard the run before reloading.");
        await open(state.resource.uri);
        await load();
        await loadRuns();
        state.notice = "Saved profile loaded.";
      } else if (name === "export") {
        requireClean();
        await download(state.resource);
      } else if (name === "export-run") {
        requireClean();
        await download(state.savedRun.resource);
      } else if (name === "history" || name === "history-next") {
        requireClean();
        await loadRuns(name === "history-next" ? state.runNext : "");
      } else if (name === "run") await execute();
      else if (name === "save-run") {
        if (!state.pending || state.running)
          throw Error("Finish the run before saving.");
        // Retain the UUID across retries of an ambiguous network write.
        state.pendingResource ??= agentResource(
          "run",
          `${state.resource.title} · ${state.pending.status} · ${new Date(state.pending.started_at).toLocaleString()}`,
          state.pending,
          state.resource.uri,
        );
        await backend.saveResource(state.namespace, state.pendingResource, "0");
        state.savedRun = {
          resource: state.pendingResource,
          value: state.pending,
        };
        state.pending = null;
        state.pendingResource = null;
        await loadRuns();
        state.notice = "Agent run saved.";
      }
    });
  }
  repaint = paint;
  paint();
  if (!state.busy)
    void work(async () => {
      await load();
      await loadRuns();
    });
}
