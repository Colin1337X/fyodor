// Native storage owns persistence/CAS and native generation owns inference.
// This controller schedules bounded local calls; it does not claim native jobs,
// in-flight cancellation, model hashes or engine-only performance measurements.
import { backend } from "./api.js";
import { escapeHtml as esc } from "./message.js";
import { writingDownloadName } from "./writing-files.js";
import { isDataset, datasetPage } from "./dataset-data.js";
import {
  evaluationDatasetCaseLimit,
  validateEvaluationDataset,
  parseEvaluationDataset,
  newEvaluationDataset,
  decodeEvaluationDataset,
} from "./evaluation-datasets.js";
import {
  evaluationEditor,
  evaluationCasePageSize,
} from "./evaluation-editor.js";
import {
  evaluationDatasetPicker,
  selectEvaluationDataset,
} from "./evaluation-dataset-picker.js";
import {
  checkTypes,
  defaultEvaluationSettings,
  newEvaluationCase,
  validateDefinition,
  validateRun,
  evaluationResource,
  parseEvaluation,
  evaluationPage,
  modelSnapshot,
  runEvaluation,
} from "./evaluation-data.js";

const state = {
  namespace: "workspace",
  items: [],
  next: "",
  libraryKind: "evaluations",
  casePage: 0,
  models: [],
  selected: new Set(),
  resource: null,
  value: null,
  revision: "0",
  dirty: false,
  busy: false,
  running: false,
  stop: false,
  error: "",
  notice: "",
  pending: null,
  pendingResource: null,
};
let mount = 0,
  repaint = () => {};
window.addEventListener("beforeunload", (event) => {
  if (state.dirty || state.pending || state.running) {
    event.preventDefault();
    event.returnValue = "";
  }
});
const modelName = (model) => model.path.split(/[\\/]/).at(-1);
export function evaluationWorkspace(root, preferredModel) {
  const generation = ++mount;
  const current = () =>
    generation === mount && document.body.dataset.view === "evaluations";
  const requireClean = () => {
    if (state.dirty || state.pending || state.running)
      throw Error("Save or discard the current changes and results first.");
  };
  async function load(after = "") {
    const page = await (
      state.libraryKind === "datasets" ? datasetPage : evaluationPage
    )(backend, state.namespace, after);
    state.items = page.resources;
    state.next = page.next;
  }
  async function models() {
    state.models = (await backend.models()).models.filter(
      (model) => model.generation_supported,
    );
    if (
      !state.selected.size &&
      state.models.some((model) => model.id === preferredModel)
    )
      state.selected.add(preferredModel);
  }
  async function run(work) {
    if (state.busy) return;
    state.busy = true;
    state.error = "";
    state.notice = "";
    paint();
    try {
      await work();
    } catch (error) {
      state.error = error.message;
    } finally {
      state.busy = false;
      repaint();
    }
  }
  function paint() {
    if (!current()) return;
    const d = state.value,
      definition = d?.kind === "definition",
      caseEditor = definition || d?.kind === "evaluation-dataset";
    root.innerHTML = `<section class="resource-workspace evaluation-workspace"><header class="view-heading"><div><p class="eyebrow">MODEL QUALITY</p><h1>Evaluations</h1><p class="muted">Save test cases, compare local models and review their actual outputs.</p></div></header>
      <div class="resource-toolbar"><label>Namespace<input id="evaluation-namespace" maxlength="64" value="${esc(state.namespace)}"></label><label>Library<select id="evaluation-library"><option value="evaluations" ${state.libraryKind === "evaluations" ? "selected" : ""}>Evaluations and results</option><option value="datasets" ${state.libraryKind === "datasets" ? "selected" : ""}>Saved datasets</option></select></label><button data-evaluation="refresh">Refresh library</button><button data-evaluation="new">New evaluation</button><button data-evaluation="new-dataset">New test dataset</button><button data-evaluation="import-cases">Import test cases</button><input id="evaluation-cases-file" type="file" accept=".json,application/json" hidden><button data-evaluation="discard">Discard editor / results</button></div>
      <p role="status" aria-live="polite" class="${state.error ? "danger-action" : "muted"}">${esc(state.error || state.notice || (state.busy ? "Working…" : ""))}</p>
      <section id="evaluation-dataset-picker"></section><div class="resource-columns"><section aria-label="Evaluation library"><ul class="resource-list">${state.items.map((item) => `<li><button data-evaluation-uri="${esc(item.uri)}" aria-pressed="${state.resource?.uri === item.uri}"><b>${esc(item.title || "Untitled evaluation")}</b><small>Revision ${esc(item.revision)}</small></button></li>`).join("") || `<li class="muted">No ${state.libraryKind === "datasets" ? "saved datasets" : "evaluations"} on this page.</li>`}</ul><button data-evaluation="next" ${state.next ? "" : "disabled"}>Next page</button></section>
      <section class="panel resource-editor">${
        d
          ? `<p id="evaluation-save-state" class="fine-print">${state.revision === "0" ? "New draft" : "Revision " + esc(state.revision)}${state.dirty ? " · Unsaved changes" : ""}</p><label>Title<input id="evaluation-title" maxlength="1024" value="${esc(state.resource.title)}"></label><div class="resource-toolbar"><button class="primary" data-evaluation="save">Save ${definition ? "definition" : caseEditor ? "test dataset" : "review"}</button>${state.revision !== "0" ? '<button data-evaluation="reload">Reload saved version</button><button data-evaluation="export">Export package</button>' : ""}${caseEditor ? '<button data-evaluation="copy">Create copy</button><button data-evaluation="download-cases">Download test cases</button>' : ""}</div>
        ${caseEditor ? evaluationEditor(d, state.casePage) : '<h2>Saved results</h2><p class="muted">Compare the saved outputs and add your review notes.</p><div id="evaluation-saved-results"></div>'}`
          : '<h2>Start an evaluation</h2><p class="muted">Create a definition and add your test cases.</p>'
      }</section></div>
      ${definition ? `<section class="panel evaluation-launch"><h2>Compare loaded models</h2><div class="resource-toolbar"><button data-evaluation="models">Refresh models</button><button class="primary" data-evaluation="run">Run saved test cases</button></div><ul class="resource-list">${state.models.map((model) => `<li><label class="check"><input type="checkbox" data-evaluation-model="${model.id}" ${state.selected.has(model.id) ? "checked" : ""}><span>${esc(modelName(model))}<small>${esc(model.architecture || model.format)} · ${esc(model.path)}</small></span></label></li>`).join("") || '<li class="muted">Load a local generation model to run these tests.</li>'}</ul><p class="muted">Choose up to four models. Each receives the same saved raw prompts and settings.</p></section>` : ""}
      <section id="evaluation-live" class="panel evaluation-live" ${state.pending || state.running ? "" : "hidden"}></section></section>`;
    root.querySelector("#evaluation-library").addEventListener(
      "change",
      (event) =>
        void run(async () => {
          state.libraryKind = event.target.value;
          await load();
        }),
    );
    root
      .querySelector("#evaluation-cases-file")
      .addEventListener("change", (event) => {
        const file = event.target.files?.[0];
        event.target.value = "";
        if (!file) return;
        void run(async () => {
          requireClean();
          if (file.size > 1024 * 1024)
            throw Error("Test-case imports must be 1 MiB or smaller.");
          const bytes = new Uint8Array(await file.arrayBuffer()),
            value = decodeEvaluationDataset(bytes, file.name);
          const digest = await crypto.subtle.digest("SHA-256", bytes);
          const provenance = {
            evaluation_studio: {
              version: 1,
              operation: "case-file-import",
              filename: file.name.slice(0, 512),
              file_bytes: bytes.length,
              file_sha256: Array.from(new Uint8Array(digest), (byte) =>
                byte.toString(16).padStart(2, "0"),
              ).join(""),
            },
          };
          adoptDraft(
            newEvaluationDataset(
              file.name.replace(/\.json$/i, ""),
              value.cases,
              provenance,
            ),
          );
          await load();
          state.notice = "Test cases imported as a new draft.";
        });
      });
    root
      .querySelectorAll("[data-evaluation]")
      .forEach((control) =>
        control.addEventListener(
          "click",
          () => void action(control.dataset.evaluation),
        ),
      );
    root.querySelectorAll("[data-evaluation-uri]").forEach((control) =>
      control.addEventListener(
        "click",
        () =>
          void run(async () => {
            requireClean();
            await open(control.dataset.evaluationUri);
          }),
      ),
    );
    root.querySelector("#evaluation-namespace").addEventListener(
      "change",
      (event) =>
        void run(async () => {
          requireClean();
          const value = event.target.value;
          if (!/^[A-Za-z0-9_-]{1,64}$/.test(value))
            throw Error("Use 1–64 letters, digits, underscores or hyphens.");
          state.namespace = value;
          state.resource = null;
          state.value = null;
          await load();
        }),
    );
    root
      .querySelector("#evaluation-title")
      ?.addEventListener("input", (event) => {
        state.resource.title = event.target.value;
        dirty();
      });
    root.querySelectorAll("[data-evaluation-setting]").forEach((control) =>
      control.addEventListener("input", () => {
        state.value.settings[control.dataset.evaluationSetting] = Number(
          control.value,
        );
        dirty();
      }),
    );
    root.querySelectorAll("[data-evaluation-model]").forEach((control) =>
      control.addEventListener("change", () => {
        const id = Number(control.dataset.evaluationModel);
        if (control.checked) {
          if (state.selected.size >= 4) {
            control.checked = false;
            return;
          }
          state.selected.add(id);
        } else state.selected.delete(id);
      }),
    );
    root.querySelectorAll("[data-case-field]").forEach((control) =>
      control.addEventListener(
        control.tagName === "SELECT" ? "change" : "input",
        () => {
          const item = state.value.cases[Number(control.dataset.caseIndex)],
            key = control.dataset.caseField;
          if (["type", "expected", "pointer"].includes(key))
            item.check[key] = control.value;
          else item[key] = control.value;
          dirty();
          if (key === "type") {
            paint();
            root
              .querySelector(
                `[data-case-index="${control.dataset.caseIndex}"][data-case-field=type]`,
              )
              ?.focus();
          }
        },
      ),
    );
    root.querySelectorAll("[data-case-remove]").forEach((control) =>
      control.addEventListener("click", () => {
        state.value.cases.splice(Number(control.dataset.caseRemove), 1);
        state.casePage = Math.min(
          state.casePage,
          Math.max(
            0,
            Math.ceil(state.value.cases.length / evaluationCasePageSize) - 1,
          ),
        );
        dirty();
        paint();
        const index = Math.min(
          Number(control.dataset.caseRemove),
          state.value.cases.length - 1,
        );
        (
          root.querySelector(
            `[data-case-index="${index}"][data-case-field=label]`,
          ) || root.querySelector("[data-evaluation=case]")
        )?.focus();
      }),
    );
    if (state.busy)
      root
        .querySelectorAll("button,input,textarea,select")
        .forEach((control) => (control.disabled = true));
    if (d?.kind === "run")
      results(root.querySelector("#evaluation-saved-results"), d, true);
    live();
    evaluationDatasetPicker(root.querySelector("#evaluation-dataset-picker"), {
      namespace: state.namespace,
      api: backend,
      busy: state.busy,
      run,
      repaint,
      getSettings: () =>
        state.value?.kind === "definition"
          ? state.value.settings
          : defaultEvaluationSettings(),
      onDraft: async (prepare) => {
        requireClean();
        const resource = await prepare();
        adoptDraft(resource);
        await load();
        state.notice = "Test draft prepared. Review it before saving.";
      },
    });
  }
  function dirty() {
    state.dirty = true;
    const target = root.querySelector("#evaluation-save-state");
    if (target)
      target.textContent =
        (state.revision === "0" ? "New draft" : "Revision " + state.revision) +
        " · Unsaved changes";
  }
  function results(target, value, review) {
    target.innerHTML = `<p class="muted">${value.status} · ${value.results.length} / ${value.planned} requests recorded${value.stop_reason ? " · " + esc(value.stop_reason) : ""}</p><p class="fine-print">Latency includes local request, queue, tokenization, inference and response decoding. Observed tokens/s uses that elapsed time.</p>
      <details class="evaluation-run-details"><summary>Run details</summary><dl><dt>Saved definition</dt><dd>${esc(value.definition.namespace)} · ${esc(value.definition.uri)} · revision ${esc(value.definition.revision)}</dd><dt>Started</dt><dd>${esc(new Date(value.started_ms).toLocaleString())}</dd><dt>Finished</dt><dd>${value.finished_ms ? esc(new Date(value.finished_ms).toLocaleString()) : "—"}</dd>${Object.entries(
        value.settings,
      )
        .map(
          ([key, number]) =>
            `<dt>${esc({ temperature: "Temperature", top_p: "Top P", top_k: "Top K", max_tokens: "Maximum new tokens", seed: "Seed" }[key])}</dt><dd>${esc(number)}</dd>`,
        )
        .join(
          "",
        )}${value.models.map((model) => `<dt>${esc(modelName(model))}</dt><dd>${esc(model.path)} · ${esc(model.architecture || model.format || "Unknown format")} · ${model.file_size === null ? "Unknown size" : model.file_size.toLocaleString() + " bytes"} · ${esc(model.compute_at_start || "Unknown compute")}</dd>`).join("")}</dl><p class="fine-print">These files and compute settings were observed at launch. A changed model file or compute provider can affect a rerun.</p></details>
      <div class="evaluation-table"><table><thead><tr><th>Model</th><th>Checked</th><th>Pass / fail</th><th>Needs review</th><th>Errors</th></tr></thead><tbody>${value.models
        .map((model, index) => {
          const rows = value.results.filter((row) => row.model_index === index),
            passed = rows.filter((row) => row.status === "pass").length,
            failed = rows.filter((row) => row.status === "fail").length;
          return `<tr><td>${esc(modelName(model))}</td><td>${passed + failed}</td><td>${passed} / ${failed}</td><td>${rows.filter((row) => row.status === "unreviewed").length}</td><td>${rows.filter((row) => row.status === "error").length}</td></tr>`;
        })
        .join("")}</tbody></table></div>
      ${value.results
        .map(
          (row, index) =>
            `<article class="evaluation-result"><h3>${esc(row.label)} · ${esc(modelName(value.models[row.model_index]))}</h3><div class="resource-toolbar"><b>${esc(row.status)}</b><span>${row.latency_ms.toFixed(2)} ms request</span><span>${row.prompt_tokens ?? "—"} prompt tokens</span><span>${row.generated_tokens ?? "—"} output tokens</span><span>${row.observed_tokens_per_second === null ? "—" : row.observed_tokens_per_second.toFixed(2)} observed tokens/s</span></div><p>${esc(row.reason)}</p><details><summary>Prompt, check and output</summary><pre>${esc(row.prompt)}</pre><p>${esc(checkTypes[row.check.type])}${["exact", "contains", "pointer"].includes(row.check.type) ? " · " + esc(row.check.expected) : ""}${row.check.type === "pointer" ? " · " + esc(row.check.pointer) : ""}</p><pre class="evaluation-output">${esc(row.text)}</pre>${row.output_truncated ? `<p class="danger-action">Stored prefix only · ${row.output_bytes.toLocaleString()} original output bytes. The check used the complete returned text.</p>` : ""}</details>${
              review
                ? `${
                    row.check.type === "manual" && row.status !== "error"
                      ? `<label>Human verdict<select data-review-verdict="${index}">${[
                          ["unreviewed", "Needs review"],
                          ["pass", "Pass"],
                          ["fail", "Fail"],
                        ]
                          .map(
                            ([key, label]) =>
                              `<option value="${key}" ${row.status === key ? "selected" : ""}>${label}</option>`,
                          )
                          .join("")}</select></label>`
                      : ""
                  }<label>Review note<textarea data-review-note="${index}" rows="2">${esc(row.review_note)}</textarea></label>`
                : ""
            }</article>`,
        )
        .join("")}`;
    target.querySelectorAll("[data-review-verdict]").forEach((control) =>
      control.addEventListener("change", () => {
        const index = control.dataset.reviewVerdict,
          row = state.value.results[Number(index)];
        row.status = control.value;
        row.reason =
          control.value === "unreviewed"
            ? "Needs human review."
            : "Human review: " + control.value;
        dirty();
        // Refresh both the row and comparison counts, then restore keyboard focus.
        results(target, state.value, true);
        target.querySelector(`[data-review-verdict="${index}"]`).focus();
      }),
    );
    target.querySelectorAll("[data-review-note]").forEach((control) =>
      control.addEventListener("input", () => {
        state.value.results[Number(control.dataset.reviewNote)].review_note =
          control.value;
        dirty();
      }),
    );
    if (state.busy)
      target
        .querySelectorAll("input,textarea,select")
        .forEach((control) => (control.disabled = true));
  }
  function live() {
    if (!current()) return;
    const target = root.querySelector("#evaluation-live");
    if (!target) return;
    target.hidden = !state.running && !state.pending;
    if (target.hidden) return;
    const run = state.pending;
    target.innerHTML = `<h2>${state.running ? "Evaluation in progress" : "Unsaved results"}</h2><div class="resource-toolbar">${state.running ? `<button data-evaluation-stop ${state.stop ? "disabled" : ""}>${state.stop ? "Stopping after current request" : "Stop after current request"}</button>` : "<button data-evaluation-results-save>Save results</button>"}</div><div id="evaluation-results"></div>`;
    target
      .querySelector("[data-evaluation-stop]")
      ?.addEventListener("click", () => {
        state.stop = true;
        live();
      });
    target
      .querySelector("[data-evaluation-results-save]")
      ?.addEventListener("click", () => void action("results-save"));
    if (run) results(target.querySelector("#evaluation-results"), run, false);
    if (state.busy && !state.running)
      target
        .querySelectorAll("button")
        .forEach((control) => (control.disabled = true));
  }
  function adoptDraft(resource) {
    state.value = isDataset(resource.uri)
      ? parseEvaluationDataset(resource)
      : parseEvaluation(resource);
    state.resource = resource;
    state.revision = "0";
    state.dirty = true;
    state.casePage = 0;
    state.libraryKind = isDataset(resource.uri) ? "datasets" : "evaluations";
  }
  async function open(uri) {
    const loaded = await backend.resource(state.namespace, uri);
    if (
      isDataset(uri) &&
      loaded.resource.metadata?.evaluation_dataset === undefined
    ) {
      await selectEvaluationDataset(
        backend,
        { ...loaded.resource, revision: loaded.revision },
        state.namespace,
      );
      state.notice = "Dataset selected. Choose its test mapping below.";
      return;
    }
    const value = isDataset(uri)
      ? parseEvaluationDataset(loaded.resource)
      : parseEvaluation(loaded.resource);
    state.resource = loaded.resource;
    state.value = value;
    state.revision = loaded.revision;
    state.dirty = false;
    state.casePage = 0;
  }
  async function execute() {
    requireClean();
    if (state.value?.kind !== "definition" || state.revision === "0")
      throw Error("Save a definition before running its test cases.");
    validateDefinition(state.value);
    await models();
    const selected = state.models.filter((model) =>
      state.selected.has(model.id),
    );
    if (!selected.length || selected.length > 4)
      throw Error("Choose 1–4 loaded local models.");
    const runtime = await backend.runtime();
    const snapshots = selected.map((model) =>
      modelSnapshot(
        model,
        runtime.models?.find((item) => item.model_id === model.id)?.compute,
      ),
    );
    const reference = {
      namespace: state.namespace,
      uri: state.resource.uri,
      revision: state.revision,
    };
    state.running = true;
    state.stop = false;
    live();
    try {
      state.pending = await runEvaluation(
        backend,
        state.value,
        reference,
        snapshots,
        {
          shouldStop: () => state.stop,
          onResult: (partial) => {
            state.pending = partial;
            live();
          },
        },
      );
      state.pendingResource = evaluationResource(
        "run",
        state.resource.title + " · results",
        state.pending,
        { evaluation_studio: { version: 1, definition: reference } },
      );
      state.notice =
        "Requests finished. Review the observations and save results.";
    } finally {
      state.running = false;
    }
  }
  async function action(name) {
    // File dialogs need the user click and an enabled input, before busy repaint.
    if (name === "import-cases") {
      if (state.busy) return;
      try {
        requireClean();
        root.querySelector("#evaluation-cases-file").click();
      } catch (error) {
        state.error = error.message;
        paint();
      }
      return;
    }
    await run(async () => {
      if (name === "new") {
        requireClean();
        state.value = {
          schema: 1,
          kind: "definition",
          settings: defaultEvaluationSettings(),
          cases: [],
        };
        state.resource = evaluationResource(
          "definition",
          "Untitled evaluation",
          state.value,
        );
        state.revision = "0";
        state.dirty = true;
        state.casePage = 0;
        state.libraryKind = "evaluations";
        await load();
      } else if (name === "new-dataset") {
        requireClean();
        adoptDraft(newEvaluationDataset());
        await load();
      } else if (name === "discard") {
        state.value = null;
        state.resource = null;
        state.pending = null;
        state.pendingResource = null;
        state.dirty = false;
        state.revision = "0";
      } else if (name === "refresh") await load();
      else if (name === "next" && state.next) await load(state.next);
      else if (name === "models") await models();
      else if (name === "cases-prev")
        state.casePage = Math.max(0, state.casePage - 1);
      else if (name === "cases-next")
        state.casePage = Math.min(
          Math.max(
            0,
            Math.ceil(state.value.cases.length / evaluationCasePageSize) - 1,
          ),
          state.casePage + 1,
        );
      else if (name === "case") {
        const limit =
          state.value.kind === "definition" ? 32 : evaluationDatasetCaseLimit;
        if (state.value.cases.length >= limit)
          throw Error(`This editor supports at most ${limit} cases.`);
        state.value.cases.push(newEvaluationCase());
        state.casePage =
          state.value.kind === "definition"
            ? 0
            : Math.floor(
                (state.value.cases.length - 1) / evaluationCasePageSize,
              );
        dirty();
      } else if (name === "copy") {
        if (state.pending) throw Error("Save or discard results first.");
        const dataset = state.value.kind === "evaluation-dataset",
          value = dataset
            ? validateEvaluationDataset(state.value)
            : validateDefinition(state.value);
        const provenance = {
          evaluation_studio: {
            version: 1,
            source: {
              namespace: state.namespace,
              uri: state.resource.uri,
              revision: state.revision,
            },
            edited: state.dirty,
          },
        };
        state.resource = dataset
          ? newEvaluationDataset(
              state.resource.title + " · copy",
              value.cases,
              provenance,
            )
          : evaluationResource(
              "definition",
              state.resource.title + " · copy",
              value,
              provenance,
            );
        state.value = value;
        state.revision = "0";
        state.dirty = true;
      } else if (name === "save") {
        const value =
          state.value.kind === "definition"
            ? validateDefinition(state.value)
            : state.value.kind === "evaluation-dataset"
              ? validateEvaluationDataset(state.value)
              : validateRun(state.value);
        const resource = { ...state.resource, content: JSON.stringify(value) };
        const saved = await (
          state.revision === "0" ? backend.saveResource : backend.updateResource
        )(state.namespace, resource, state.revision);
        state.resource = resource;
        state.value = value;
        state.revision = saved.revision;
        state.dirty = false;
        state.libraryKind = isDataset(state.resource.uri)
          ? "datasets"
          : "evaluations";
        await load();
        state.notice =
          state.value.kind === "evaluation-dataset"
            ? "Test dataset saved."
            : "Evaluation saved.";
      } else if (name === "reload") {
        if (state.pending)
          throw Error("Save or discard results before reloading.");
        await open(state.resource.uri);
        state.notice = "Saved version loaded.";
      } else if (name === "download-cases") {
        const value = validateEvaluationDataset({
          schema: 1,
          kind: "evaluation-dataset",
          cases: state.value.cases,
        });
        const url = URL.createObjectURL(
            // Compact output keeps every valid download within the import cap.
            new Blob([JSON.stringify(value)], {
              type: "application/json",
            }),
          ),
          link = document.createElement("a");
        link.href = url;
        link.download = writingDownloadName(
          state.resource.title,
          "txt",
        ).replace(/\.txt$/, ".cases.json");
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        state.notice =
          "Current test cases downloaded. Saved data stays unchanged.";
      } else if (name === "export") {
        if (state.dirty)
          throw Error("Save changes before exporting a package.");
        const text = await backend.exportResource(
          state.namespace,
          state.resource.uri,
        );
        const url = URL.createObjectURL(
            new Blob([text], { type: "application/json" }),
          ),
          link = document.createElement("a");
        link.href = url;
        link.download = writingDownloadName(
          state.resource.title,
          "txt",
        ).replace(/\.txt$/, ".json");
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        state.notice = "Saved evaluation package exported.";
      } else if (name === "run") await execute();
      else if (name === "results-save") {
        if (!state.pendingResource)
          throw Error("No completed observations to save.");
        const saved = await backend.saveResource(
          state.namespace,
          state.pendingResource,
          "0",
        );
        state.resource = state.pendingResource;
        state.value = state.pending;
        state.revision = saved.revision;
        state.dirty = false;
        state.pending = null;
        state.pendingResource = null;
        state.libraryKind = "evaluations";
        await load();
        state.notice = "Evaluation results saved.";
      }
    });
    if (current() && ["case", "cases-prev", "cases-next"].includes(name)) {
      const selector =
        name === "case"
          ? `[data-case-index="${state.value.cases.length - 1}"][data-case-field=label]`
          : `[data-evaluation="${name}"]:not(:disabled)`;
      (
        root.querySelector(selector) ||
        root.querySelector("[data-case-field=label]")
      )?.focus();
    }
  }
  repaint = paint;
  paint();
  if (!state.busy)
    void run(async () => {
      await load();
      await models();
    });
}
