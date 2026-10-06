// Evaluation datasets contain explicit test cases, distinct from native trainer
// text/TSV. Exact revision reads and hashes describe input bytes, not model access
// grants, ground truth, model identity or attested execution.
import {
  isDataset,
  savedDatasetMode,
  inspectDataset,
  seededShuffle,
} from "./dataset-data.js";
import {
  evaluationLimit,
  validateEvaluationCases,
  validateDefinition,
  defaultEvaluationSettings,
  evaluationResource,
} from "./evaluation-data.js";
import { boundedResourceTitle } from "./resource-text.js";

const encoder = new TextEncoder();
export const evaluationDatasetCaseLimit = 512;
const namespace = (value) =>
  typeof value === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(value);
export function validateEvaluationDataset(value) {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).some(
      (key) => !["schema", "kind", "cases"].includes(key),
    ) ||
    value.schema !== 1 ||
    value.kind !== "evaluation-dataset"
  )
    throw Error("Expected test dataset schema 1.");
  const result = {
    schema: 1,
    kind: "evaluation-dataset",
    cases: validateEvaluationCases(value.cases, evaluationDatasetCaseLimit),
  };
  if (encoder.encode(JSON.stringify(result)).length > evaluationLimit)
    throw Error("Test dataset exceeds 1 MiB. Choose fewer or shorter cases.");
  return result;
}
export function parseEvaluationDataset(resource) {
  if (
    !isDataset(resource?.uri) ||
    resource.metadata?.evaluation_dataset?.version !== 1 ||
    typeof resource.content !== "string" ||
    encoder.encode(resource.content).length > evaluationLimit
  )
    throw Error(
      "This test dataset format is unsupported. Export it from Resources.",
    );
  return validateEvaluationDataset(JSON.parse(resource.content));
}
export function newEvaluationDataset(
  title = "Untitled test dataset",
  cases = [],
  provenance = {},
) {
  const value = validateEvaluationDataset({
    schema: 1,
    kind: "evaluation-dataset",
    cases,
  });
  return {
    schema: 1,
    uri: "fyodor://datasets/" + crypto.randomUUID(),
    title: boundedResourceTitle(title),
    content: JSON.stringify(value),
    metadata: { evaluation_dataset: { version: 1 } },
    provenance,
  };
}
export function decodeEvaluationDataset(bytes, name) {
  if (!/\.json$/i.test(name) || bytes.byteLength > evaluationLimit)
    throw Error("Choose a UTF-8 test-case .json file, 1 MiB or smaller.");
  let content;
  try {
    content = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw Error("Save test cases as UTF-8 JSON and try again.");
  }
  return validateEvaluationDataset(JSON.parse(content));
}
export function evaluationDatasetReference(item, ns) {
  if (
    !isDataset(item?.uri) ||
    !namespace(ns) ||
    typeof item.revision !== "string" ||
    !/^[1-9][0-9]*$/.test(item.revision)
  )
    throw Error("Choose a saved dataset revision.");
  return {
    namespace: ns,
    uri: item.uri,
    revision: item.revision,
    title: typeof item.title === "string" ? item.title : "Untitled dataset",
  };
}
export function inspectEvaluationSource(resource) {
  if (
    !isDataset(resource?.uri) ||
    typeof resource.content !== "string" ||
    encoder.encode(resource.content).length > evaluationLimit
  )
    throw Error("Choose a supported saved dataset, 1 MiB or smaller.");
  // Unknown Evaluation annotations never fall back to interpreting JSON as
  // trainer text. Known case datasets preserve their labels, checks and IDs.
  if (resource.metadata?.evaluation_dataset !== undefined) {
    const value = parseEvaluationDataset(resource);
    return {
      kind: "cases",
      mode: null,
      count: value.cases.length,
      rows: value.cases.map((item, index) => ({ index, case: item })),
    };
  }
  const mode = savedDatasetMode(resource);
  if (!mode)
    throw Error("This dataset format is unsupported. Open it in Resources.");
  const report = inspectDataset(resource.content, mode);
  if (report.issues.length)
    throw Error(
      "Fix dataset format issues before creating test cases: " +
        report.issues[0].message,
    );
  return {
    kind: "training",
    mode,
    count: report.rows.length,
    rows: report.rows,
  };
}
export async function readEvaluationSource(api, reference) {
  const ref = evaluationDatasetReference(reference, reference?.namespace);
  const loaded = await api.resource(ref.namespace, ref.uri, ref.revision);
  if (
    loaded.deleted ||
    loaded.revision !== ref.revision ||
    loaded.resource?.uri !== ref.uri
  )
    throw Error(
      "Selected dataset revision is unavailable. Browse and select it again.",
    );
  return {
    resource: loaded.resource,
    report: inspectEvaluationSource(loaded.resource),
  };
}
async function sha256(text) {
  const bytes = await crypto.subtle.digest("SHA-256", encoder.encode(text));
  return Array.from(new Uint8Array(bytes), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}
function selectRows(report, selection, maximum) {
  if (
    !selection ||
    !["range", "sample"].includes(selection.method) ||
    !Number.isSafeInteger(selection.count) ||
    selection.count < 1 ||
    selection.count > maximum ||
    selection.count > report.count
  )
    throw Error(`Choose 1–${maximum} available cases.`);
  if (selection.method === "range") {
    if (
      !Number.isSafeInteger(selection.start) ||
      selection.start < 1 ||
      selection.start - 1 + selection.count > report.count
    )
      throw Error("The case range must fit the available records.");
    return report.rows.slice(
      selection.start - 1,
      selection.start - 1 + selection.count,
    );
  }
  return seededShuffle(report.rows, selection.seed).slice(0, selection.count);
}
export async function evaluationFromDataset(
  api,
  reference,
  {
    target = "definition",
    mapping = "manual",
    selection = { method: "range", start: 1, count: 1, seed: 42 },
    settings = defaultEvaluationSettings(),
  } = {},
) {
  if (!["definition", "dataset"].includes(target))
    throw Error("Choose a definition or test dataset.");
  const ref = evaluationDatasetReference(reference, reference?.namespace);
  const { resource, report } = await readEvaluationSource(api, ref);
  const maximum = target === "definition" ? 32 : evaluationDatasetCaseLimit;
  const rows = selectRows(report, selection, maximum);
  if (
    report.kind === "training" &&
    (!["manual", "exact", "contains"].includes(mapping) ||
      (report.mode !== "sft" && mapping !== "manual"))
  )
    throw Error(
      "Exact/contains completion checks require SFT records. Corpus and DPO prompts use human review.",
    );
  const cases = rows.map((row) =>
    report.kind === "cases"
      ? structuredClone(row.case)
      : {
          id: crypto.randomUUID(),
          label: "Record " + row.line,
          prompt: row.fields[0],
          check: {
            type: mapping,
            expected: mapping === "manual" ? "" : row.fields[1],
            pointer: "",
          },
        },
  );
  // All selected cases must fit their individual bounds; never silently truncate
  // prompts, skip invalid records or turn a preferred response into ground truth.
  validateEvaluationCases(cases, maximum);
  const selectionRecord =
    selection.method === "range"
      ? { method: "range", start: selection.start, count: selection.count }
      : {
          method: "sample",
          seed: selection.seed,
          count: selection.count,
          algorithm: "mulberry32-fisher-yates-v1",
        };
  const provenance = {
    evaluation_studio: {
      version: 1,
      operation: "dataset-mapping",
      source: {
        namespace: ref.namespace,
        uri: ref.uri,
        revision: ref.revision,
        content_sha256: await sha256(resource.content),
      },
      source_format:
        report.kind === "cases" ? "evaluation-dataset" : report.mode,
      mapping:
        report.kind === "cases"
          ? "preserve-cases"
          : report.mode === "sft"
            ? "prompt-completion-" + mapping
            : report.mode === "dpo"
              ? "prompt-only-manual"
              : "nonempty-line-prompt-manual",
      selection: selectionRecord,
      source_records: rows.map((row) =>
        report.kind === "cases"
          ? { case_index: row.index, case_id: row.case.id }
          : { line: row.line },
      ),
      mapped_cases_sha256: await sha256(JSON.stringify(cases)),
    },
  };
  if (encoder.encode(JSON.stringify(provenance)).length > 65536)
    throw Error("Mapping provenance exceeds 64 KiB. Choose fewer cases.");
  const title =
    (typeof resource.title === "string" ? resource.title : ref.title) +
    " · " +
    (target === "definition" ? "evaluation" : "test cases");
  if (target === "dataset")
    return newEvaluationDataset(title, cases, provenance);
  return evaluationResource(
    "definition",
    title,
    validateDefinition({ schema: 1, kind: "definition", settings, cases }),
    provenance,
  );
}
