import {
  agentConnection,
  validateAgent,
  validateAgentReference,
  agentTools,
} from "./agent-data.js";
const encoder = new TextEncoder();
const bytes = (value) => encoder.encode(JSON.stringify(value)).length;
const text = (value) =>
  typeof value === "string" &&
  !value.includes("\0") &&
  encoder.encode(value).length <= 65536;
const toolSchemas = Object.keys(agentTools).map((name) => ({
  type: "function",
  function: {
    name,
    description: agentTools[name],
    parameters: { type: "object", properties: {}, additionalProperties: false },
  },
}));
const step = (kind, status, fields = {}) => ({
  kind,
  status,
  text: "",
  tool: null,
  error: "",
  request_ms: null,
  tokens: null,
  ...fields,
});
const safeError = (error) =>
  String(error?.message || error)
    .replace(/\0/g, "�")
    .slice(0, 1000);
const stopped = (signal) => {
  if (signal?.aborted) throw new DOMException("Agent stopped.", "AbortError");
};
const id = (value) => Number.isSafeInteger(value) && value > 0;
const scalar = (value) =>
  typeof value === "string" &&
  !value.includes("\0") &&
  encoder.encode(value).length <= 256;

// Local read-only adapters return a narrow observation, never bearer tokens,
// filesystem paths, resource text or Context grants. No requested function name
// can become a property lookup, shell command or unrestricted API call.
export async function agentReadTool(api, name) {
  if (name === "local_time") return new Date().toISOString();
  if (name === "runtime_status") {
    const data = await api.runtime();
    if (
      data?.version !== 1 ||
      data.ok !== true ||
      !Array.isArray(data.models) ||
      data.models.length > 32
    )
      throw Error("Unsupported runtime observation.");
    const providers = ["cpu", "cuda", "vulkan", "rocm", "mlx"];
    if (
      providers.some((key) => typeof data.compiled?.[key] !== "boolean") ||
      data.models.some(
        (item) => !id(item?.model_id) || !providers.includes(item.compute),
      )
    )
      throw Error("Unsupported runtime observation.");
    return JSON.stringify({
      compiled: Object.fromEntries(
        providers.map((key) => [key, data.compiled?.[key] === true]),
      ),
      models: data.models.map((item) => ({
        model_id: item.model_id,
        compute: item.compute,
      })),
    });
  }
  if (name === "list_loaded_models") {
    const data = await api.models();
    if (
      data?.version !== 1 ||
      data.ok !== true ||
      !Array.isArray(data.models) ||
      data.models.length > 32
    )
      throw Error("Unsupported model catalog.");
    if (
      data.models.some(
        (item) =>
          !id(item?.id) ||
          (item.architecture != null && !scalar(item.architecture)) ||
          (item.path != null &&
            (typeof item.path !== "string" ||
              item.path.includes("\0") ||
              encoder.encode(item.path).length > 8192)),
      )
    )
      throw Error("Unsupported model catalog.");
    return JSON.stringify(
      data.models.map((item) => ({
        id: item.id,
        name:
          typeof item.path === "string"
            ? item.path.split(/[\\/]/).pop()
            : "Unnamed model",
        architecture: item.architecture ?? null,
      })),
    );
  }
  throw Error("Tool is not permitted.");
}
export async function runAgent({
  definition,
  reference,
  input,
  provider,
  api,
  signal,
  onUpdate = () => {},
}) {
  const config = validateAgent(definition),
    connection = agentConnection(config, provider);
  if (
    typeof input !== "string" ||
    input.includes("\0") ||
    encoder.encode(input).length > 8192
  )
    throw Error("Use a run input up to 8 KiB without null characters.");
  const run = {
    schema: 1,
    kind: "agent-run",
    agent: validateAgentReference(reference),
    definition: config,
    input,
    status: "failed",
    steps: [],
    output: "",
    error: "",
    started_at: Date.now(),
    finished_at: Date.now(),
  };
  const messages = [
    { role: "system", content: config.instructions },
    { role: "user", content: input },
  ];
  let toolCount = 0;
  const append = (value) => {
    if (bytes([...run.steps, value]) > 524288) {
      const error = Error("Run reached its recorded-output limit.");
      error.limit = true;
      throw error;
    }
    run.steps.push(value);
    onUpdate(structuredClone(run));
  };
  try {
    for (let turn = 0; turn < config.limits.turns; turn++) {
      stopped(signal);
      if (bytes(messages) > 786432) {
        const error = Error("Run reached its request-history limit.");
        error.limit = true;
        throw error;
      }
      const body = {
        model: config.target.model,
        messages: structuredClone(messages),
        ...config.sampling,
        stream: false,
      };
      if (config.tools.length)
        body.tools = toolSchemas.filter((schema) =>
          config.tools.includes(schema.function.name),
        );
      const started = performance.now();
      let response;
      try {
        response = await api.openaiChat(connection, body, { signal });
      } catch (error) {
        append(
          step("model", error.name === "AbortError" ? "stopped" : "failed", {
            error: safeError(error),
            request_ms: performance.now() - started,
          }),
        );
        throw error;
      }
      stopped(signal);
      const choice = response?.choices?.[0]?.message,
        calls = choice?.tool_calls ?? [];
      if (
        !choice ||
        choice.role !== "assistant" ||
        !Array.isArray(calls) ||
        calls.length > 8 ||
        !text(choice.content ?? "")
      )
        throw Error("Unsupported assistant response.");
      const tokens =
        Number.isSafeInteger(response.usage?.completion_tokens) &&
        response.usage.completion_tokens >= 0
          ? response.usage.completion_tokens
          : null;
      run.output = choice.content ?? "";
      append(
        step("model", "completed", {
          text: choice.content ?? "",
          request_ms: performance.now() - started,
          tokens,
        }),
      );
      if (!calls.length) {
        if (typeof choice.content !== "string")
          throw Error("Endpoint returned no assistant text.");
        run.status = "completed";
        break;
      }
      // Validate the entire batch before executing any member. A denied call
      // cannot sneak through beside a permitted call or exceed the saved budget.
      const ids = new Set();
      for (const call of calls) {
        if (
          call?.type !== "function" ||
          typeof call.id !== "string" ||
          call.id.includes("\0") ||
          encoder.encode(call.id).length > 128 ||
          !call.id ||
          ids.has(call.id) ||
          !config.tools.includes(call.function?.name) ||
          typeof call.function?.arguments !== "string" ||
          encoder.encode(call.function.arguments).length > 1024
        )
          throw Error("Endpoint requested a tool that is not permitted.");
        ids.add(call.id);
        let args;
        try {
          args = JSON.parse(call.function.arguments);
        } catch {
          throw Error("Tool arguments must be an empty JSON object.");
        }
        if (
          !args ||
          typeof args !== "object" ||
          Array.isArray(args) ||
          Object.keys(args).length
        )
          throw Error("Tool arguments must be an empty JSON object.");
      }
      if (toolCount + calls.length > config.limits.tool_calls) {
        const error = Error("Agent reached its tool-call limit.");
        error.limit = true;
        throw error;
      }
      messages.push({
        role: "assistant",
        content: choice.content ?? null,
        tool_calls: structuredClone(calls),
      });
      for (const call of calls) {
        stopped(signal);
        const name = call.function.name;
        toolCount++;
        let output;
        try {
          output = await agentReadTool(api, name);
          if (!text(output) || encoder.encode(output).length > 32768)
            throw Error("Tool output exceeded its supported bound.");
        } catch (error) {
          append(
            step("tool", "failed", { tool: name, error: safeError(error) }),
          );
          throw error;
        }
        append(step("tool", "completed", { tool: name, text: output }));
        stopped(signal);
        messages.push({ role: "tool", tool_call_id: call.id, content: output });
      }
      if (turn === config.limits.turns - 1) {
        run.status = "limit";
        run.error = "Agent reached its model-turn limit.";
      }
    }
  } catch (error) {
    run.status =
      error.name === "AbortError"
        ? "stopped"
        : error.limit
          ? "limit"
          : "failed";
    run.error = safeError(error);
  }
  run.finished_at = Math.max(run.started_at, Date.now());
  onUpdate(structuredClone(run));
  return run;
}
