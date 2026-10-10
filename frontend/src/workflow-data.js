// Stable step IDs and explicit earlier-output references can later become graph
// edges. Today these are closed frontend records over native workflow resources,
// not an invented native job/event/permission protocol.
import { boundedResourceTitle } from "./resource-text.js";
import {
  defaultEvaluationSettings,
  evaluationSettings,
  validateCheck,
  evaluateOutput,
} from "./evaluation-data.js";
import { validateAgentReference, validateAgentRun } from "./agent-data.js";
const encoder = new TextEncoder();
const uuidPattern = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/;
const uuid = (value) =>
  typeof value === "string" &&
  uuidPattern.test(value) &&
  value !== "00000000-0000-0000-0000-000000000000";
const known = (value, keys) =>
  value &&
  typeof value === "object" &&
  !Array.isArray(value) &&
  Object.keys(value).length === keys.length &&
  Object.keys(value).every((key) => keys.includes(key));
const count = (value, min, max) =>
  Number.isSafeInteger(value) && value >= min && value <= max;
export const workflowOperations = {
  text: "Compose text",
  generate: "Generate locally",
  check: "Check text",
  agent: "Run saved agent",
};
export const workflowChecks = ["exact", "contains", "json", "pointer"];
export const workflowLimit = 1048576;
export function workflowText(value, limit, label = "Text") {
  if (
    typeof value !== "string" ||
    value.includes("\0") ||
    encoder.encode(value).length > limit
  )
    throw Error(`${label} exceeds its supported UTF-8 text bound.`);
  return value;
}
export const workflowUri = (value) =>
  typeof value === "string" &&
  value.startsWith("fyodor://workflows/") &&
  uuid(value.slice(19));
export function workflowReference(value) {
  if (
    !known(value, ["namespace", "uri", "revision"]) ||
    typeof value.namespace !== "string" ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(value.namespace) ||
    !workflowUri(value.uri) ||
    typeof value.revision !== "string" ||
    !/^[1-9][0-9]{0,18}$/.test(value.revision) ||
    BigInt(value.revision) > 9223372036854775807n
  )
    throw Error("Use an exact saved workflow revision.");
  return structuredClone(value);
}
export function newWorkflowStep(operation = "text", source = "run-input") {
  if (!Object.hasOwn(workflowOperations, operation))
    throw Error("Unsupported workflow operation.");
  return {
    id: crypto.randomUUID(),
    label: workflowOperations[operation],
    operation,
    input: { source, prefix: "", suffix: "" },
    config:
      operation === "generate"
        ? defaultEvaluationSettings()
        : operation === "check"
          ? { type: "exact", expected: "", pointer: "" }
          : operation === "agent"
            ? { namespace: "workspace", uri: "", revision: "0" }
            : {},
  };
}
export const newWorkflow = () => ({
  schema: 1,
  kind: "workflow-definition",
  steps: [newWorkflowStep()],
});
export function validateWorkflowWiring(steps) {
  if (!Array.isArray(steps) || steps.length < 1 || steps.length > 12)
    throw Error("Use 1–12 workflow steps.");
  const seen = new Set();
  for (const item of steps) {
    if (!uuid(item?.id) || seen.has(item.id))
      throw Error("Each step needs a unique stable ID.");
    if (item.input?.source !== "run-input" && !seen.has(item.input?.source))
      throw Error(
        "Step inputs may use only the run input or an earlier step output.",
      );
    seen.add(item.id);
  }
}
export function validateWorkflow(value) {
  if (
    !known(value, ["schema", "kind", "steps"]) ||
    value.schema !== 1 ||
    value.kind !== "workflow-definition"
  )
    throw Error("Unsupported workflow definition.");
  validateWorkflowWiring(value.steps);
  for (const item of value.steps) {
    if (
      !known(item, ["id", "label", "operation", "input", "config"]) ||
      !Object.hasOwn(workflowOperations, item.operation) ||
      !workflowText(item.label, 256, "Step label").trim() ||
      !known(item.input, ["source", "prefix", "suffix"])
    )
      throw Error("Unsupported workflow step.");
    workflowText(item.input.prefix, 2048, "Input prefix");
    workflowText(item.input.suffix, 2048, "Input suffix");
    if (item.operation === "text" && !known(item.config, []))
      throw Error("Compose text has no executable configuration.");
    if (item.operation === "generate") {
      if (
        !known(item.config, [
          "temperature",
          "top_p",
          "top_k",
          "max_tokens",
          "seed",
        ])
      )
        throw Error("Unsupported generation configuration.");
      evaluationSettings(item.config);
    }
    if (item.operation === "check") {
      if (
        !known(item.config, ["type", "expected", "pointer"]) ||
        !workflowChecks.includes(item.config.type)
      )
        throw Error("Use an automated literal or JSON check.");
      validateCheck(item.config);
      workflowText(item.config.expected, 8192, "Expected value");
    }
    if (item.operation === "agent") validateAgentReference(item.config);
  }
  return structuredClone(value);
}
export function workflowInput(spec, input, outputs) {
  const source = spec.source === "run-input" ? input : outputs.get(spec.source);
  if (typeof source !== "string")
    throw Error("The referenced step has no completed output.");
  return workflowText(
    spec.prefix + source + spec.suffix,
    65536,
    "Composed input",
  );
}
export function validateWorkflowRun(value) {
  if (
    !known(value, [
      "schema",
      "kind",
      "workflow",
      "definition",
      "input",
      "model",
      "status",
      "steps",
      "output",
      "error",
      "started_at",
      "finished_at",
    ]) ||
    value.schema !== 1 ||
    value.kind !== "workflow-run" ||
    !["completed", "failed", "stopped", "limit"].includes(value.status)
  )
    throw Error("Unsupported workflow run.");
  workflowReference(value.workflow);
  const definition = validateWorkflow(value.definition);
  workflowText(value.input, 8192, "Run input");
  workflowText(value.output, 65536, "Run output");
  workflowText(value.error, 4096, "Run error");
  if (
    !count(value.started_at, 1, 8640000000000000) ||
    !count(value.finished_at, value.started_at, 8640000000000000) ||
    !Array.isArray(value.steps) ||
    value.steps.length > definition.steps.length
  )
    throw Error("Unsupported workflow timing or trace.");
  if (
    value.model !== null &&
    (!known(value.model, [
      "process_id",
      "path",
      "file_size",
      "format",
      "architecture",
      "compute_at_start",
    ]) ||
      !count(value.model.process_id, 1, Number.MAX_SAFE_INTEGER) ||
      typeof value.model.path !== "string" ||
      (value.model.file_size !== null &&
        !count(value.model.file_size, 0, Number.MAX_SAFE_INTEGER)) ||
      ["format", "architecture", "compute_at_start"].some(
        (key) =>
          value.model[key] !== null && typeof value.model[key] !== "string",
      ))
  )
    throw Error("Unsupported runtime model observation.");
  if (value.model !== null) {
    workflowText(value.model.path, 8192, "Observed model path");
    for (const key of ["format", "architecture", "compute_at_start"])
      if (value.model[key] !== null)
        workflowText(value.model[key], 256, "Model observation");
    if (!definition.steps.some((step) => step.operation === "generate"))
      throw Error("Workflow has no native generation model to observe.");
  }
  const outputs = new Map();
  let last = "";
  for (let index = 0; index < value.steps.length; index++) {
    const record = value.steps[index],
      spec = definition.steps[index];
    if (
      !known(record, [
        "id",
        "status",
        "input",
        "output",
        "error",
        "request_ms",
        "prompt_tokens",
        "generated_tokens",
        "agent_run",
      ]) ||
      record.id !== spec.id ||
      !["completed", "failed", "stopped", "limit"].includes(record.status) ||
      (record.status !== "completed" && index !== value.steps.length - 1)
    )
      throw Error("Unsupported workflow step trace.");
    workflowText(record.input, 65536, "Recorded input");
    workflowText(record.output, 65536, "Recorded output");
    workflowText(record.error, 4096, "Step error");
    if (record.input !== workflowInput(spec.input, value.input, outputs))
      throw Error("Recorded input does not match its saved binding.");
    if (
      record.request_ms !== null &&
      (!Number.isFinite(record.request_ms) || record.request_ms < 0)
    )
      throw Error("Unsupported request duration.");
    if (
      ["prompt_tokens", "generated_tokens"].some(
        (key) =>
          record[key] !== null &&
          !count(record[key], 0, Number.MAX_SAFE_INTEGER),
      )
    )
      throw Error("Unsupported reported token count.");
    if (
      spec.operation !== "generate" &&
      (record.request_ms !== null ||
        record.prompt_tokens !== null ||
        record.generated_tokens !== null)
    )
      throw Error("Only local generation steps have native request metrics.");
    if (spec.operation === "generate" && value.model === null)
      throw Error("Generation trace requires its actual model observation.");
    if (record.agent_run !== null) {
      if (spec.operation !== "agent") throw Error("Unexpected agent trace.");
      validateAgentRun(record.agent_run);
      if (
        ["namespace", "uri", "revision"].some(
          (key) => record.agent_run.agent[key] !== spec.config[key],
        ) ||
        record.agent_run.input !== record.input ||
        record.agent_run.output !== record.output ||
        record.agent_run.status !== record.status
      )
        throw Error("Agent trace does not match the workflow step.");
    } else if (spec.operation === "agent" && record.status === "completed")
      throw Error("Completed agent step requires actual observations.");
    if (record.status === "completed") {
      if (
        ["text", "check"].includes(spec.operation) &&
        record.output !== record.input
      )
        throw Error("Deterministic step output does not match its input.");
      if (
        spec.operation === "check" &&
        evaluateOutput(record.input, spec.config).status !== "pass"
      )
        throw Error("Recorded check did not pass.");
      outputs.set(record.id, record.output);
      last = record.output;
    }
  }
  if (
    value.output !== last ||
    (value.status === "completed" &&
      (value.steps.length !== definition.steps.length ||
        value.steps.some((item) => item.status !== "completed")))
  )
    throw Error("Workflow completion does not match its recorded steps.");
  if (encoder.encode(JSON.stringify(value)).length > workflowLimit)
    throw Error("Workflow run exceeds 1 MiB.");
  return structuredClone(value);
}
export function workflowResource(kind, title, value) {
  if (!["definition", "run"].includes(kind))
    throw Error("Unsupported workflow resource kind.");
  const parsed =
    kind === "definition"
      ? validateWorkflow(value)
      : validateWorkflowRun(value);
  return {
    schema: 1,
    uri: "fyodor://workflows/" + crypto.randomUUID(),
    title: boundedResourceTitle(title),
    content: JSON.stringify(parsed),
    metadata: {
      workflow_studio: {
        version: 1,
        kind,
        ...(kind === "run" ? { workflow: parsed.workflow.uri } : {}),
      },
    },
    provenance: {
      workflow_studio: {
        operation:
          kind === "definition" ? "workflow-definition" : "client-workflow-run",
      },
    },
  };
}
export function parseWorkflow(resource) {
  const annotation = resource?.metadata?.workflow_studio;
  if (
    !workflowUri(resource?.uri) ||
    annotation?.version !== 1 ||
    !["definition", "run"].includes(annotation.kind)
  )
    throw Error(
      "Unsupported workflow resource. Inspect or export it from Resources.",
    );
  workflowText(resource.content, workflowLimit, "Workflow content");
  let value;
  try {
    value = JSON.parse(resource.content);
  } catch {
    throw Error("Invalid workflow JSON.");
  }
  const parsed =
    annotation.kind === "definition"
      ? validateWorkflow(value)
      : validateWorkflowRun(value);
  if (annotation.kind === "run" && annotation.workflow !== parsed.workflow.uri)
    throw Error("Workflow run annotation does not match its definition.");
  if (annotation.kind === "run" && parsed.workflow.uri === resource.uri)
    throw Error("A run cannot be its own workflow definition.");
  return parsed;
}
export async function workflowPage(
  api,
  namespace,
  kind = "definition",
  parent = null,
  after = "",
) {
  if (
    typeof namespace !== "string" ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(namespace) ||
    !["definition", "run"].includes(kind) ||
    (kind === "run" && !workflowUri(parent)) ||
    typeof after !== "string"
  )
    throw Error("Unsupported workflow catalog selection.");
  const resources = [],
    seen = new Set();
  let cursor = after,
    reads = 0;
  for (let page = 0; page < 25; page++) {
    seen.add(cursor);
    const data = await api.resources(namespace, cursor, "all");
    if (
      !Array.isArray(data.resources) ||
      data.resources.length > 100 ||
      typeof data.next !== "string"
    )
      throw Error("Unsupported resource catalog.");
    // The native title catalog omits metadata. Read only workflow identities at
    // the catalog's exact revision; never infer record kind from a title. Finish
    // the whole consumed page so a lookup budget cannot silently skip matches.
    for (const item of data.resources.filter((row) => workflowUri(row?.uri))) {
      workflowReference({ namespace, uri: item.uri, revision: item.revision });
      const loaded = await api.resource(namespace, item.uri, item.revision);
      if (
        loaded.resource?.uri !== item.uri ||
        loaded.revision !== item.revision
      )
        throw Error("Workflow catalog revision changed unexpectedly.");
      workflowText(loaded.resource.content, workflowLimit, "Workflow content");
      reads++;
      const annotation = loaded.resource.metadata?.workflow_studio;
      const definition =
        annotation?.kind === "definition" ||
        !["definition", "run"].includes(annotation?.kind);
      if (
        (kind === "definition" && definition) ||
        (kind === "run" &&
          annotation?.kind === "run" &&
          annotation.workflow === parent)
      )
        resources.push(item);
    }
    cursor = data.next;
    if (cursor && seen.has(cursor))
      throw Error("Repeated resource cursor. Refresh the workflow library.");
    if (!cursor || resources.length >= 20 || reads >= 100) break;
  }
  return { resources, next: cursor };
}
