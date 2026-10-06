// Render-only composition shared by definitions and reusable test datasets.
// The workspace owns draft state, listeners, native writes and conflict handling.
import { escapeHtml as esc } from "./message.js";
import { checkTypes } from "./evaluation-data.js";
import { evaluationDatasetCaseLimit } from "./evaluation-datasets.js";
export const evaluationCasePageSize = 20;
export function evaluationEditor(value, page = 0) {
  const definition = value.kind === "definition",
    limit = definition ? 32 : evaluationDatasetCaseLimit;
  const start = definition ? 0 : page * evaluationCasePageSize;
  const cases = definition
    ? value.cases
    : value.cases.slice(start, start + evaluationCasePageSize);
  return `${
    definition
      ? `<h2>Generation settings</h2><div class="config-grid">${[
          ["temperature", "Temperature", 0, 5, 0.01],
          ["top_p", "Top P", 0.000001, 1, 0.01],
          ["top_k", "Top K", 0, 1000000, 1],
          ["max_tokens", "Maximum new tokens", 1, 4096, 1],
          ["seed", "Seed", 0, Number.MAX_SAFE_INTEGER, 1],
        ]
          .map(
            ([key, label, min, max, step]) =>
              `<label>${label}<input type="number" data-evaluation-setting="${key}" value="${esc(value.settings[key])}" min="${min}" max="${max}" step="${step}"></label>`,
          )
          .join("")}</div>`
      : '<h2>Reusable test dataset</h2><p class="muted">Save prompts and checks here, then select cases for a model comparison.</p>'
  }
    <h2>Test cases · ${value.cases.length} / ${limit}</h2>${!definition ? `<div class="resource-toolbar"><button data-evaluation="cases-prev" ${page ? "" : "disabled"}>Previous cases</button><span>Page ${page + 1} / ${Math.max(1, Math.ceil(value.cases.length / evaluationCasePageSize))}</span><button data-evaluation="cases-next" ${start + evaluationCasePageSize < value.cases.length ? "" : "disabled"}>Next cases</button></div>` : ""}
    ${cases
      .map((item, offset) => {
        const index = start + offset;
        return `<article class="evaluation-case"><div class="resource-toolbar"><label>Label<input data-case-index="${index}" data-case-field="label" value="${esc(item.label)}"></label><button data-case-remove="${index}">Remove case</button></div><label>Raw prompt<textarea data-case-index="${index}" data-case-field="prompt" rows="3">${esc(item.prompt)}</textarea></label><label>Correctness check<select data-case-index="${index}" data-case-field="type">${Object.entries(
          checkTypes,
        )
          .map(
            ([key, label]) =>
              `<option value="${key}" ${item.check.type === key ? "selected" : ""}>${label}</option>`,
          )
          .join(
            "",
          )}</select></label>${["exact", "contains", "pointer"].includes(item.check.type) ? `<label>${item.check.type === "pointer" ? "Expected JSON value" : "Expected text"}<textarea data-case-index="${index}" data-case-field="expected" rows="2">${esc(item.check.expected)}</textarea></label>` : ""}${item.check.type === "pointer" ? `<label>JSON pointer<input data-case-index="${index}" data-case-field="pointer" value="${esc(item.check.pointer)}" placeholder="/answer"></label>` : ""}</article>`;
      })
      .join("")}<button data-evaluation="case">Add test case</button>`;
}
