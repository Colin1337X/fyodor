// Versioned frontend records use the native generic resource store. They are
// not native job receipts or immutable model identities. No invented metrics,
// regex execution, model-graded correctness or implicit Context reads live here.
import { boundedResourceTitle } from "./resource-text.js";
const encoder = new TextEncoder();
export const evaluationLimit = 1024 * 1024;
export const checkTypes = {
  manual: "Human review",
  exact: "Exact text",
  contains: "Contains text",
  json: "Valid JSON",
  pointer: "JSON value at pointer",
};
export const evaluationUri = (uri) =>
  /^fyodor:\/\/evaluations\/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(
    uri,
  );
const caseId = (value) =>
  typeof value === "string" &&
  /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(value);
const namespace = (value) =>
  typeof value === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(value);
export const defaultEvaluationSettings = () => ({
  temperature: 0,
  top_p: 1,
  top_k: 0,
  max_tokens: 64,
  seed: 42,
});
const keys = (value, allowed) =>
  value &&
  typeof value === "object" &&
  !Array.isArray(value) &&
  Object.keys(value).every((key) => allowed.includes(key));
function text(value, limit, label) {
  if (typeof value !== "string" || encoder.encode(value).length > limit)
    throw Error(`${label} exceeds its supported text bound.`);
  return value;
}
function finite(value, min, max, integer = false) {
  return (
    Number.isFinite(value) &&
    value >= min &&
    value <= max &&
    (!integer || Number.isSafeInteger(value))
  );
}

export function evaluationSettings(value) {
  if (
    !keys(value, ["temperature", "top_p", "top_k", "max_tokens", "seed"]) ||
    !finite(value.temperature, 0, 5) ||
    !finite(value.top_p, Number.MIN_VALUE, 1) ||
    !finite(value.top_k, 0, 1000000, true) ||
    !finite(value.max_tokens, 1, 4096, true) ||
    !finite(value.seed, 0, Number.MAX_SAFE_INTEGER, true)
  )
    throw Error(
      "Use bounded native generation settings and an exact nonnegative seed.",
    );
  return { ...value };
}
function structured(text) {
  const value = JSON.parse(text);
  let nodes = 0;
  const visit = (node, depth) => {
    if (++nodes > 10000 || depth > 32)
      throw Error("JSON check exceeds 32 levels or 10000 values.");
    if (node && typeof node === "object")
      Object.values(node).forEach((child) => visit(child, depth + 1));
  };
  visit(value, 0);
  return value;
}
function pointerSegments(path) {
  text(path, 512, "JSON pointer");
  if (path === "") return [];
  if (!path.startsWith("/") || /~(?:[^01]|$)/.test(path))
    throw Error("Use a JSON pointer such as /answer or /items/0.");
  const parts = path
    .slice(1)
    .split("/")
    .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"));
  if (parts.length > 32)
    throw Error("JSON pointers support at most 32 segments.");
  return parts;
}
function equal(a, b) {
  if (a === b) return true;
  if (
    !a ||
    !b ||
    typeof a !== "object" ||
    typeof b !== "object" ||
    Array.isArray(a) !== Array.isArray(b)
  )
    return false;
  const left = Object.keys(a),
    right = Object.keys(b);
  return (
    left.length === right.length &&
    left.every((key) => Object.hasOwn(b, key) && equal(a[key], b[key]))
  );
}
export function validateCheck(check) {
  if (
    !keys(check, ["type", "expected", "pointer"]) ||
    !Object.hasOwn(checkTypes, check.type)
  )
    throw Error("Choose a supported correctness check.");
  text(check.expected, 8192, "Expected value");
  text(check.pointer, 512, "JSON pointer");
  if (check.type === "contains" && !check.expected.length)
    throw Error("Contains checks need nonempty expected text.");
  if (check.type === "pointer") {
    pointerSegments(check.pointer);
    try {
      structured(check.expected);
    } catch (error) {
      throw Error("Expected JSON value: " + error.message);
    }
  }
  return { ...check };
}
export function evaluateOutput(output, check) {
  validateCheck(check);
  if (check.type === "manual")
    return { status: "unreviewed", reason: "Needs human review." };
  let passed = false;
  try {
    if (check.type === "exact") passed = output === check.expected;
    else if (check.type === "contains")
      passed = output.includes(check.expected);
    else {
      let value = structured(output);
      passed = true;
      if (check.type === "pointer") {
        for (const key of pointerSegments(check.pointer)) {
          if (
            !value ||
            typeof value !== "object" ||
            (Array.isArray(value) && !/^(0|[1-9][0-9]*)$/.test(key)) ||
            !Object.hasOwn(value, key)
          )
            return { status: "fail", reason: "JSON pointer was not present." };
          value = value[key];
        }
        passed = equal(value, structured(check.expected));
      }
    }
  } catch (error) {
    return { status: "fail", reason: "JSON check failed: " + error.message };
  }
  return {
    status: passed ? "pass" : "fail",
    reason: passed ? "Check passed." : "Output did not match the check.",
  };
}
export function validateDefinition(value) {
  if (
    !keys(value, ["schema", "kind", "settings", "cases"]) ||
    value.schema !== 1 ||
    value.kind !== "definition" ||
    !Array.isArray(value.cases) ||
    value.cases.length > 32
  )
    throw Error(
      "Expected Evaluation definition schema 1 with at most 32 cases.",
    );
  const seen = new Set();
  const cases = value.cases.map((item) => {
    if (
      !keys(item, ["id", "label", "prompt", "check"]) ||
      !caseId(item.id) ||
      seen.has(item.id)
    )
      throw Error("Each case needs a unique case ID.");
    seen.add(item.id);
    text(item.label, 512, "Case label");
    text(item.prompt, 16384, "Case prompt");
    return { ...item, check: validateCheck(item.check) };
  });
  return {
    schema: 1,
    kind: "definition",
    settings: evaluationSettings(value.settings),
    cases,
  };
}
export function newEvaluationCase() {
  return {
    id: crypto.randomUUID(),
    label: "Untitled case",
    prompt: "",
    check: { type: "manual", expected: "", pointer: "" },
  };
}
export function evaluationResource(kind, title, value, provenance = {}) {
  const content = JSON.stringify(value);
  if (encoder.encode(content).length > evaluationLimit)
    throw Error("Evaluation content exceeds the native 1 MiB resource limit.");
  return {
    schema: 1,
    uri: "fyodor://evaluations/" + crypto.randomUUID(),
    title: boundedResourceTitle(title),
    content,
    metadata: { evaluation_studio: { version: 1, kind } },
    provenance,
  };
}
export function parseEvaluation(resource) {
  const annotation = resource.metadata?.evaluation_studio;
  if (
    !evaluationUri(resource.uri) ||
    annotation?.version !== 1 ||
    !["definition", "run"].includes(annotation.kind)
  )
    throw Error(
      "This evaluation format is unsupported. Export it from Resources.",
    );
  if (encoder.encode(resource.content).length > evaluationLimit)
    throw Error("Evaluation content is too large.");
  const value = JSON.parse(resource.content);
  return annotation.kind === "definition"
    ? validateDefinition(value)
    : validateRun(value);
}
export function modelSnapshot(model, compute = null) {
  if (
    !Number.isSafeInteger(model.id) ||
    model.id < 1 ||
    model.generation_supported !== true ||
    typeof model.path !== "string"
  )
    throw Error("Choose a loaded local generation model.");
  return {
    process_id: model.id,
    path: model.path,
    file_size:
      Number.isSafeInteger(model.file_size) && model.file_size >= 0
        ? model.file_size
        : null,
    format: typeof model.format === "string" ? model.format : null,
    architecture:
      typeof model.architecture === "string" ? model.architecture : null,
    compute_at_start: typeof compute === "string" ? compute : null,
  };
}
export function validateRun(value) {
  // Narrow known records fail closed instead of rewriting unknown future fields.
  if (
    !keys(value, [
      "schema",
      "kind",
      "definition",
      "settings",
      "models",
      "started_ms",
      "finished_ms",
      "status",
      "stop_reason",
      "planned",
      "results",
    ]) ||
    value.schema !== 1 ||
    value.kind !== "run" ||
    !keys(value.definition, ["namespace", "uri", "revision"]) ||
    !namespace(value.definition.namespace) ||
    !evaluationUri(value.definition.uri) ||
    typeof value.definition.revision !== "string" ||
    !/^[1-9][0-9]*$/.test(value.definition.revision) ||
    !Array.isArray(value.models) ||
    value.models.length < 1 ||
    value.models.length > 4 ||
    !Array.isArray(value.results) ||
    value.results.length > 128 ||
    !["completed", "stopped"].includes(value.status) ||
    ![null, "requested", "result-size"].includes(value.stop_reason) ||
    !finite(value.started_ms, 0, 8640000000000000, true) ||
    !finite(value.finished_ms, 0, 8640000000000000, true) ||
    !finite(value.planned, 1, 128, true) ||
    value.results.length > value.planned
  )
    throw Error("Expected Evaluation run schema 1.");
  evaluationSettings(value.settings);
  const modelIds = new Set();
  for (const model of value.models) {
    if (
      !keys(model, [
        "process_id",
        "path",
        "file_size",
        "format",
        "architecture",
        "compute_at_start",
      ]) ||
      !Number.isSafeInteger(model.process_id) ||
      model.process_id < 1 ||
      modelIds.has(model.process_id) ||
      !(
        model.file_size === null ||
        finite(model.file_size, 0, Number.MAX_SAFE_INTEGER, true)
      )
    )
      throw Error("Invalid run model snapshot.");
    text(model.path, 16384, "Model path");
    for (const field of ["format", "architecture", "compute_at_start"])
      if (model[field] !== null) text(model[field], 512, "Model " + field);
    modelIds.add(model.process_id);
  }
  const observations = new Set();
  for (const row of value.results) {
    if (
      !keys(row, [
        "case_id",
        "label",
        "prompt",
        "check",
        "model_index",
        "text",
        "output_truncated",
        "output_bytes",
        "latency_ms",
        "prompt_tokens",
        "generated_tokens",
        "observed_tokens_per_second",
        "stop_reason",
        "status",
        "reason",
        "review_note",
      ]) ||
      !Number.isSafeInteger(row.model_index) ||
      !value.models[row.model_index] ||
      !caseId(row.case_id) ||
      observations.has(row.model_index + ":" + row.case_id) ||
      !["pass", "fail", "unreviewed", "error"].includes(row.status) ||
      typeof row.output_truncated !== "boolean" ||
      !finite(row.output_bytes, 0, 16 * evaluationLimit, true) ||
      !finite(row.latency_ms, 0, Number.MAX_VALUE) ||
      ![row.prompt_tokens, row.generated_tokens].every(
        (n) => n === null || finite(n, 0, Number.MAX_SAFE_INTEGER, true),
      ) ||
      !(
        row.observed_tokens_per_second === null ||
        finite(row.observed_tokens_per_second, 0, Number.MAX_VALUE)
      )
    )
      throw Error("Invalid evaluation measurement.");
    text(row.label, 512, "Result label");
    text(row.prompt, 16384, "Result prompt");
    text(row.text, 65536, "Saved output");
    text(row.reason, 2048, "Result reason");
    text(row.review_note, 8192, "Review note");
    validateCheck(row.check);
    if (row.stop_reason !== null) text(row.stop_reason, 512, "Stop reason");
    const savedBytes = encoder.encode(row.text).length;
    if (
      row.output_truncated
        ? row.output_bytes <= savedBytes
        : row.output_bytes !== savedBytes
    )
      throw Error("Saved output byte count is inconsistent.");
    observations.add(row.model_index + ":" + row.case_id);
  }
  if (
    value.status === "completed" &&
    (value.results.length !== value.planned || value.stop_reason !== null)
  )
    throw Error("A completed run must contain every planned result.");
  if (value.status === "stopped" && value.stop_reason === null)
    throw Error("A stopped run needs its stop reason.");
  return value;
}
function outputPrefix(value, limit = 65536) {
  // Store complete Unicode code points; checks still see the full response.
  if (encoder.encode(value).length <= limit) return value;
  let output = "",
    size = 0;
  for (const character of value) {
    const bytes = encoder.encode(character).length;
    if (size + bytes > limit) break;
    output += character;
    size += bytes;
  }
  return output;
}
const count = (value) =>
  Number.isSafeInteger(value) && value >= 0 ? value : null;
export async function runEvaluation(
  api,
  definition,
  reference,
  models,
  {
    clock = () => performance.now(),
    wallClock = () => Date.now(),
    shouldStop = () => false,
    onResult = () => {},
  } = {},
) {
  const spec = validateDefinition(definition);
  if (!spec.cases.length || !models.length || models.length > 4)
    throw Error("Add test cases and choose 1–4 models.");
  if (
    !evaluationUri(reference.uri) ||
    typeof reference.revision !== "string" ||
    !/^[1-9][0-9]*$/.test(reference.revision) ||
    !namespace(reference.namespace)
  )
    throw Error("Save the definition before running it.");
  const run = {
    schema: 1,
    kind: "run",
    definition: { ...reference },
    settings: spec.settings,
    models: structuredClone(models),
    started_ms: wallClock(),
    finished_ms: 0,
    status: "running",
    stop_reason: null,
    planned: models.length * spec.cases.length,
    results: [],
  };
  // Validate snapshots before a request can execute, using a valid empty stop.
  validateRun({
    ...run,
    status: "stopped",
    stop_reason: "requested",
    finished_ms: run.started_ms,
  });
  for (let model_index = 0; model_index < models.length; model_index++)
    for (const item of spec.cases) {
      if (shouldStop()) {
        run.status = "stopped";
        run.stop_reason = "requested";
        run.finished_ms = wallClock();
        return validateRun(run);
      }
      const start = clock();
      let response, error;
      try {
        response = await api.generate({
          model_id: models[model_index].process_id,
          prompt: item.prompt,
          ...spec.settings,
        });
        if (
          response.model_id !== models[model_index].process_id ||
          typeof response.text !== "string" ||
          response.seed !== spec.settings.seed
        )
          throw Error("Generation returned a mismatched model, seed or text.");
      } catch (failure) {
        error =
          (failure instanceof Error ? failure.message : String(failure)) ||
          "Generation failed.";
      }
      const elapsed = Math.max(0, clock() - start),
        generated = error ? null : count(response.generated_tokens),
        output = error ? "" : response.text;
      const verdict = error
        ? { status: "error", reason: error }
        : evaluateOutput(output, item.check);
      const row = {
        case_id: item.id,
        label: item.label,
        prompt: item.prompt,
        check: item.check,
        model_index,
        text: outputPrefix(output),
        output_truncated: encoder.encode(output).length > 65536,
        output_bytes: encoder.encode(output).length,
        latency_ms: elapsed,
        prompt_tokens: error ? null : count(response.prompt_tokens),
        generated_tokens: generated,
        observed_tokens_per_second:
          generated !== null && elapsed > 0
            ? (generated * 1000) / elapsed
            : null,
        stop_reason: error
          ? null
          : typeof response.stop_reason === "string"
            ? response.stop_reason
            : null,
        ...verdict,
        reason: outputPrefix(verdict.reason, 2048),
        review_note: "",
      };
      if (
        encoder.encode(
          JSON.stringify({ ...run, results: [...run.results, row] }),
        ).length >
        evaluationLimit - 4096
      ) {
        // The just-completed request is omitted from the bounded resource.
        // Earlier rows stay available; no subsequent request is scheduled.
        run.status = "stopped";
        run.stop_reason = "result-size";
        run.finished_ms = wallClock();
        return validateRun(run);
      }
      run.results.push(row);
      onResult(run, row);
    }
  run.status = "completed";
  run.finished_ms = wallClock();
  return validateRun(run);
}
export async function evaluationPage(api, namespace, after = "") {
  const resources = [],
    seen = new Set();
  let cursor = after;
  for (let i = 0; i < 25; i++) {
    seen.add(cursor);
    const page = await api.resources(namespace, cursor, "all");
    resources.push(...page.resources.filter((item) => evaluationUri(item.uri)));
    cursor = page.next;
    if (cursor && seen.has(cursor))
      throw Error("Repeated evaluation cursor. Refresh the library.");
    if (!cursor || resources.length >= 20) break;
  }
  return { resources, next: cursor };
}
