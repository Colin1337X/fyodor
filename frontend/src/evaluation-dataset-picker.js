// Selection belongs to this picker; its parent owns busy state and editor
// replacement. A selected revision stays fixed until the user selects again.
import { escapeHtml as esc } from "./message.js";
import { datasetPage } from "./dataset-data.js";
import {
  evaluationDatasetReference,
  readEvaluationSource,
  evaluationFromDataset,
} from "./evaluation-datasets.js";

const state = {
  namespace: null,
  open: false,
  items: [],
  next: "",
  reference: null,
  report: null,
  target: "definition",
  mapping: "manual",
  method: "range",
  start: 1,
  count: 1,
  seed: 42,
};
export function resetEvaluationDatasetPicker(namespace) {
  Object.assign(state, {
    namespace,
    items: [],
    next: "",
    reference: null,
    report: null,
    start: 1,
    count: 1,
  });
}
export async function selectEvaluationDataset(api, item, namespace) {
  if (state.namespace !== namespace) resetEvaluationDatasetPicker(namespace);
  const reference = evaluationDatasetReference(item, namespace);
  const { report } = await readEvaluationSource(api, reference);
  // Assign only after validation, preserving prior selection on failed reads.
  state.reference = reference;
  state.report = report;
  state.open = true;
  state.start = 1;
  state.count = Math.min(32, report.count);
  state.mapping = "manual";
}
export function evaluationDatasetPicker(
  root,
  { namespace, api, busy, run, repaint, onDraft, getSettings },
) {
  if (state.namespace !== namespace) resetEvaluationDatasetPicker(namespace);
  const report = state.report;
  root.innerHTML = `<details class="panel evaluation-dataset-picker" ${state.open ? "open" : ""}><summary>Create tests from a saved dataset</summary>
    <div class="resource-toolbar"><button data-evaluation-dataset="browse">Browse datasets</button><button data-evaluation-dataset="next" ${state.next ? "" : "disabled"}>Next page</button></div>
    <ul class="resource-list">${state.items.map((item) => `<li><button data-evaluation-dataset-uri="${esc(item.uri)}" aria-pressed="${state.reference?.uri === item.uri && state.reference?.revision === item.revision}"><b>${esc(item.title || "Untitled dataset")}</b><small>Revision ${esc(item.revision)}</small></button></li>`).join("") || '<li class="muted">Choose a saved training or test dataset.</li>'}</ul>
    ${
      state.reference
        ? `<p class="muted">${esc(state.reference.title)} · revision ${esc(state.reference.revision)} · ${report.count} ${report.kind === "cases" ? "test cases" : report.mode === "sft" ? "prompt/completion records" : report.mode === "dpo" ? "preference records" : "nonempty lines"}</p><div class="config-grid"><label>Create<select id="evaluation-dataset-target"><option value="definition">Evaluation definition · up to 32 cases</option><option value="dataset">Reusable test dataset · up to 512 cases</option></select></label><label>Selection<select id="evaluation-dataset-method"><option value="range">Consecutive range</option><option value="sample">Seeded sample</option></select></label><label>Case count<input id="evaluation-dataset-count" type="number" min="1" max="${state.target === "definition" ? 32 : 512}" value="${esc(state.count)}"></label>${state.method === "range" ? `<label>First record / case<input id="evaluation-dataset-start" type="number" min="1" value="${esc(state.start)}"></label>` : `<label>Selection seed<input id="evaluation-dataset-seed" type="number" min="0" max="4294967295" value="${esc(state.seed)}"></label>`}${report.kind === "training" && report.mode === "sft" ? `<label>Completion check<select id="evaluation-dataset-mapping"><option value="manual">Human review · prompt only</option><option value="exact">Exact completion text</option><option value="contains">Contains completion text</option></select></label>` : ""}</div>
      <p class="muted">${report.kind === "cases" ? "Saved prompts, checks and labels are preserved." : report.mode === "sft" ? "The first field becomes the raw prompt. Choose explicitly whether the completion is an expected answer." : report.mode === "dpo" ? "Only the prompt becomes a human-review case. Chosen/rejected text does not establish correctness." : "Each nonempty line becomes a raw prompt for human review."}</p><button class="primary" data-evaluation-dataset="prepare" ${report.count ? "" : "disabled"}>Create test draft</button><p class="fine-print">The source stays unchanged. Review and save the new draft.</p>`
        : ""
    }</details>`;
  root
    .querySelector("details")
    .addEventListener("toggle", (event) => (state.open = event.target.open));
  for (const key of ["target", "method", "mapping", "start", "count", "seed"]) {
    const control = root.querySelector("#evaluation-dataset-" + key);
    if (!control) continue;
    if (["target", "method", "mapping"].includes(key))
      control.value = state[key];
    control.addEventListener(
      control.tagName === "SELECT" ? "change" : "input",
      () => {
        state[key] =
          control.tagName === "SELECT" ? control.value : Number(control.value);
        if (key === "target" || key === "method") {
          repaint();
          focus("#evaluation-dataset-" + key);
        }
      },
    );
  }
  root.querySelectorAll("[data-evaluation-dataset-uri]").forEach((control) =>
    control.addEventListener("click", () => {
      const item = state.items.find(
        (item) => item.uri === control.dataset.evaluationDatasetUri,
      );
      void run(() => selectEvaluationDataset(api, item, namespace)).then(() =>
        focus("[data-evaluation-dataset=prepare]"),
      );
    }),
  );
  root.querySelectorAll("[data-evaluation-dataset]").forEach((control) =>
    control.addEventListener(
      "click",
      () =>
        void run(async () => {
          const name = control.dataset.evaluationDataset;
          if (name === "browse" || name === "next") {
            const page = await datasetPage(
              api,
              namespace,
              name === "next" ? state.next : "",
            );
            state.items = page.resources;
            state.next = page.next;
          } else if (name === "prepare") {
            await onDraft(() =>
              evaluationFromDataset(api, state.reference, {
                target: state.target,
                mapping: state.mapping,
                selection: {
                  method: state.method,
                  start: state.start,
                  count: state.count,
                  seed: state.seed,
                },
                settings: getSettings(),
              }),
            );
            state.open = false;
          }
        }).then(() =>
          focus(
            control.dataset.evaluationDataset === "prepare" && !state.open
              ? "#evaluation-title"
              : "[data-evaluation-dataset=" +
                  control.dataset.evaluationDataset +
                  "]",
          ),
        ),
    ),
  );
  if (busy)
    root
      .querySelectorAll("button,input,select")
      .forEach((control) => (control.disabled = true));
  function focus(selector) {
    if (
      document.body.dataset.view === "evaluations" &&
      state.namespace === namespace
    )
      document
        .querySelector(
          selector === "#evaluation-title"
            ? selector
            : "#evaluation-dataset-picker " + selector,
        )
        ?.focus();
  }
}
