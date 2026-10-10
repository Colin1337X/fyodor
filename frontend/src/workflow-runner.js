import {
  validateWorkflow,
  workflowReference,
  workflowText,
  workflowInput,
  validateWorkflowRun,
} from "./workflow-data.js";
import { parseAgent, agentConnection } from "./agent-data.js";
import { runAgent } from "./agent-runner.js";
import { evaluateOutput, modelSnapshot } from "./evaluation-data.js";
const encoder = new TextEncoder();
const bytes = (value) => encoder.encode(JSON.stringify(value)).length;
const errorText = (error) =>
  String(error?.message || error)
    .replace(/\0/g, "�")
    .slice(0, 1000);
const counters = (value) =>
  Number.isSafeInteger(value) && value >= 0 ? value : null;
const stopped = (signal) => {
  if (signal?.aborted)
    throw new DOMException("Workflow stopped.", "AbortError");
};
const limited = (message) => {
  const error = Error(message);
  error.limit = true;
  return error;
};

// Preflight every referenced agent before any composing/generation request. A
// workflow does not turn a stale saved tool choice into renewed authorization.
export async function workflowPreflight(
  api,
  definition,
  reference,
  provider,
  modelId,
) {
  const spec = validateWorkflow(definition),
    ref = workflowReference(reference),
    agents = new Map();
  for (const step of spec.steps.filter((item) => item.operation === "agent")) {
    if (step.config.namespace !== ref.namespace)
      throw Error("Agent references must use the workflow namespace.");
    const key = step.config.uri + "@" + step.config.revision;
    if (!agents.has(key)) {
      const head = await api.resource(ref.namespace, step.config.uri);
      if (
        head.resource?.uri !== step.config.uri ||
        head.revision !== step.config.revision
      )
        throw Error(
          "A referenced agent changed. Select its reviewed saved profile again before running.",
        );
      const value = parseAgent(head.resource);
      agentConnection(value, provider);
      agents.set(key, value);
    }
  }
  let model = null;
  if (spec.steps.some((item) => item.operation === "generate")) {
    const catalog = await api.models(),
      runtime = await api.runtime();
    if (
      catalog?.version !== 1 ||
      catalog.ok !== true ||
      !Array.isArray(catalog.models) ||
      catalog.models.length > 32 ||
      runtime?.version !== 1 ||
      runtime.ok !== true ||
      !Array.isArray(runtime.models) ||
      runtime.models.length > 32
    )
      throw Error("Unsupported native model observations.");
    const selected = catalog.models.find((item) => item.id === modelId);
    if (!selected)
      throw Error("Choose a loaded local generation model for this run.");
    model = modelSnapshot(
      selected,
      runtime.models.find((item) => item.model_id === modelId)?.compute ?? null,
    );
  }
  return { spec, ref, agents, model };
}
export async function runWorkflow({
  definition,
  reference,
  input,
  provider,
  modelId,
  api,
  signal,
  onUpdate = () => {},
}) {
  provider = { ...provider }; // Freeze this run's connection without persisting it.
  workflowText(input, 8192, "Run input");
  stopped(signal);
  const { spec, ref, agents, model } = await workflowPreflight(
    api,
    definition,
    reference,
    provider,
    modelId,
  );
  const run = {
      schema: 1,
      kind: "workflow-run",
      workflow: ref,
      definition: spec,
      input,
      model,
      status: "failed",
      steps: [],
      output: "",
      error: "",
      started_at: Date.now(),
      finished_at: Date.now(),
    },
    outputs = new Map();
  // Runs stay below the native record bound even with embedded Agent traces.
  const append = (record) => {
    if (bytes([...run.steps, record]) > 524288)
      throw limited("Workflow reached its recorded-output limit.");
    run.steps.push(record);
    if (record.status === "completed") {
      outputs.set(record.id, record.output);
      run.output = record.output;
    }
    onUpdate(structuredClone(run), null);
  };
  try {
    for (const step of spec.steps) {
      stopped(signal);
      const text = workflowInput(step.input, input, outputs);
      const record = {
        id: step.id,
        status: "completed",
        input: text,
        output: "",
        error: "",
        request_ms: null,
        prompt_tokens: null,
        generated_tokens: null,
        agent_run: null,
      };
      onUpdate(structuredClone(run), step.id);
      try {
        if (step.operation === "text") record.output = text;
        else if (step.operation === "check") {
          const verdict = evaluateOutput(text, step.config);
          if (verdict.status !== "pass") throw Error(verdict.reason);
          record.output = text;
        } else if (step.operation === "generate") {
          const start = performance.now();
          let response;
          try {
            response = await api.generate({
              model_id: model.process_id,
              prompt: text,
              ...step.config,
            });
          } finally {
            record.request_ms = Math.max(0, performance.now() - start);
          }
          if (
            response?.model_id !== model.process_id ||
            response.seed !== step.config.seed ||
            typeof response.text !== "string"
          )
            throw Error(
              "Generation returned a mismatched model, seed or text.",
            );
          record.output = workflowText(
            response.text,
            65536,
            "Generated output",
          );
          record.prompt_tokens = counters(response.prompt_tokens);
          record.generated_tokens = counters(response.generated_tokens);
        } else if (step.operation === "agent") {
          // Re-check immediately at the step, in addition to whole-run preflight.
          const head = await api.resource(ref.namespace, step.config.uri);
          if (
            head.resource?.uri !== step.config.uri ||
            head.revision !== step.config.revision
          )
            throw Error(
              "Referenced agent changed before its step. Review and select the profile again.",
            );
          record.agent_run = await runAgent({
            definition: agents.get(
              step.config.uri + "@" + step.config.revision,
            ),
            reference: step.config,
            input: text,
            provider,
            api,
            signal,
          });
          record.output = record.agent_run.output;
          record.status = record.agent_run.status;
          record.error = record.agent_run.error;
        }
      } catch (error) {
        record.status =
          error.name === "AbortError"
            ? "stopped"
            : error.limit
              ? "limit"
              : "failed";
        record.error = errorText(error);
        append(record);
        throw error;
      }
      append(record);
      if (record.status !== "completed") {
        if (record.status === "stopped")
          throw new DOMException(
            record.error || "Agent stopped.",
            "AbortError",
          );
        if (record.status === "limit") throw limited(record.error);
        throw Error(record.error || "Agent failed.");
      }
      // Native generation has no cancel contract: retain its actual response,
      // then stop scheduling. Aborting is never labelled native acknowledgement.
      stopped(signal);
    }
    run.status = "completed";
  } catch (error) {
    run.status =
      error.name === "AbortError"
        ? "stopped"
        : error.limit
          ? "limit"
          : "failed";
    run.error = errorText(error);
  }
  run.finished_at = Math.max(run.started_at, Date.now());
  const result = validateWorkflowRun(run);
  onUpdate(structuredClone(result), null);
  return result;
}
