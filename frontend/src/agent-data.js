// Frontend definitions/runs live in existing native typed resources. They are
// client execution records, not native jobs, grants or immutable model lineage.
import { boundedResourceTitle } from "./resource-text.js";
const encoder = new TextEncoder(),
  uuid = "[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}";
const uriPattern = new RegExp(
  `^fyodor://agents/(${uuid})(?:/runs/(${uuid}))?$`,
);
export const agentTools = {
  runtime_status: "Read runtime status",
  list_loaded_models: "List loaded model names",
  local_time: "Read local time",
};
export const agentRecordLimit = 1048576;
const known = (value, keys) =>
  value &&
  typeof value === "object" &&
  !Array.isArray(value) &&
  Object.keys(value).length === keys.length &&
  Object.keys(value).every((key) => keys.includes(key));
const bounded = (value, bytes, label) => {
  if (
    typeof value !== "string" ||
    value.includes("\0") ||
    encoder.encode(value).length > bytes
  )
    throw Error(`${label} exceeds its supported text bound.`);
  return value;
};
const integer = (value, min, max) =>
  Number.isSafeInteger(value) && value >= min && value <= max;
export function agentKind(uri) {
  const match = typeof uri === "string" && uri.match(uriPattern);
  if (
    !match ||
    match.slice(1).some((id) => id === "00000000-0000-0000-0000-000000000000")
  )
    return null;
  return match[2] ? "run" : "definition";
}
export function agentEndpoint(value) {
  bounded(value, 2048, "Endpoint");
  let url;
  try {
    url = new URL(value);
  } catch {
    throw Error("Use an HTTP(S) endpoint URL.");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw Error(
      "Use an HTTP(S) endpoint without credentials, query or fragment.",
    );
  return url.href.replace(/\/+$/, "");
}
export function newAgent(provider = {}) {
  return {
    schema: 1,
    kind: "agent-definition",
    instructions: "",
    target: {
      kind: "configured-endpoint",
      endpoint: provider.type === "openai" ? provider.endpoint : "",
      model: provider.type === "openai" ? provider.model : "",
    },
    tools: [],
    permissions: { context: false, resources: false },
    limits: { turns: 6, tool_calls: 8 },
    sampling: { max_tokens: 256, temperature: 0.5, top_p: 1 },
  };
}
export function validateAgent(value) {
  if (
    !known(value, [
      "schema",
      "kind",
      "instructions",
      "target",
      "tools",
      "permissions",
      "limits",
      "sampling",
    ]) ||
    value.schema !== 1 ||
    value.kind !== "agent-definition"
  )
    throw Error("Unsupported agent definition.");
  bounded(value.instructions, 16384, "Instructions");
  if (
    !known(value.target, ["kind", "endpoint", "model"]) ||
    value.target.kind !== "configured-endpoint" ||
    !bounded(value.target.model, 256, "Model").trim()
  )
    throw Error("Supply an endpoint model name.");
  const endpoint = agentEndpoint(value.target.endpoint);
  if (
    !Array.isArray(value.tools) ||
    value.tools.length > 3 ||
    new Set(value.tools).size !== value.tools.length ||
    value.tools.some((tool) => !Object.hasOwn(agentTools, tool))
  )
    throw Error("Select only supported read-only tools.");
  if (
    !known(value.permissions, ["context", "resources"]) ||
    value.permissions.context !== false ||
    value.permissions.resources !== false
  )
    throw Error(
      "Context and resource tools are not supported by this agent format.",
    );
  if (
    !known(value.limits, ["turns", "tool_calls"]) ||
    !integer(value.limits.turns, 1, 12) ||
    !integer(value.limits.tool_calls, 0, 48)
  )
    throw Error("Use 1–12 model turns and 0–48 tool calls.");
  if (
    !known(value.sampling, ["max_tokens", "temperature", "top_p"]) ||
    !integer(value.sampling.max_tokens, 1, 4096) ||
    !Number.isFinite(value.sampling.temperature) ||
    value.sampling.temperature < 0 ||
    value.sampling.temperature > 5 ||
    !Number.isFinite(value.sampling.top_p) ||
    value.sampling.top_p <= 0 ||
    value.sampling.top_p > 1
  )
    throw Error("Use supported sampling settings.");
  return { ...structuredClone(value), target: { ...value.target, endpoint } };
}
export function agentConnection(value, provider) {
  const definition = validateAgent(value);
  if (
    provider?.type !== "openai" ||
    agentEndpoint(provider.endpoint) !== definition.target.endpoint
  )
    throw Error(
      "Configure the matching OpenAI-compatible connection in Settings before running this agent.",
    );
  return {
    endpoint: definition.target.endpoint,
    model: definition.target.model,
    apiKey: typeof provider.apiKey === "string" ? provider.apiKey : "",
  };
}
export function agentResource(kind, title, value, parent) {
  if (!["definition", "run"].includes(kind))
    throw Error("Unsupported agent kind.");
  const content = JSON.stringify(
    kind === "definition" ? validateAgent(value) : validateAgentRun(value),
  );
  if (encoder.encode(content).length > agentRecordLimit)
    throw Error("Agent content exceeds 1 MiB.");
  if (
    kind === "run" &&
    (agentKind(parent) !== "definition" || value.agent.uri !== parent)
  )
    throw Error("A saved agent parent is required.");
  return {
    schema: 1,
    uri:
      kind === "definition"
        ? "fyodor://agents/" + crypto.randomUUID()
        : parent + "/runs/" + crypto.randomUUID(),
    title: boundedResourceTitle(title),
    content,
    metadata: { agent_studio: { version: 1, kind } },
    provenance: {
      agent_studio: {
        operation:
          kind === "definition" ? "agent-definition" : "client-endpoint-run",
      },
    },
  };
}
export function validateAgentRun(value) {
  if (
    !known(value, [
      "schema",
      "kind",
      "agent",
      "definition",
      "input",
      "status",
      "steps",
      "output",
      "error",
      "started_at",
      "finished_at",
    ]) ||
    value.schema !== 1 ||
    value.kind !== "agent-run" ||
    !["completed", "stopped", "failed", "limit"].includes(value.status)
  )
    throw Error("Unsupported agent run.");
  validateAgentReference(value.agent);
  validateAgent(value.definition);
  bounded(value.input, 8192, "Run input");
  bounded(value.output, 65536, "Assistant text");
  bounded(value.error, 4096, "Run error");
  if (
    !integer(value.started_at, 1, 8640000000000000) ||
    !integer(value.finished_at, value.started_at, 8640000000000000) ||
    !Array.isArray(value.steps) ||
    value.steps.length > 64
  )
    throw Error("Unsupported run timing or steps.");
  for (const step of value.steps) {
    if (
      !known(step, [
        "kind",
        "status",
        "text",
        "tool",
        "error",
        "request_ms",
        "tokens",
      ]) ||
      !["model", "tool", "limit"].includes(step.kind) ||
      !["completed", "failed", "stopped", "limit"].includes(step.status)
    )
      throw Error("Unsupported run step.");
    bounded(step.text, 65536, "Step text");
    bounded(step.error, 4096, "Step error");
    if (
      step.kind === "tool"
        ? !Object.hasOwn(agentTools, step.tool)
        : step.tool !== null
    )
      throw Error("Unsupported recorded tool.");
    if (
      step.kind !== "model" &&
      (step.tokens !== null || step.request_ms !== null)
    )
      throw Error("Only model steps have request measurements.");
    if (
      step.request_ms !== null &&
      (!Number.isFinite(step.request_ms) || step.request_ms < 0)
    )
      throw Error("Unsupported request duration.");
    if (
      step.tokens !== null &&
      !integer(step.tokens, 0, Number.MAX_SAFE_INTEGER)
    )
      throw Error("Unsupported reported token count.");
  }
  if (encoder.encode(JSON.stringify(value)).length > agentRecordLimit)
    throw Error("Agent run exceeds 1 MiB.");
  return structuredClone(value);
}
export function validateAgentReference(value) {
  if (
    !known(value, ["namespace", "uri", "revision"]) ||
    typeof value.namespace !== "string" ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(value.namespace) ||
    agentKind(value.uri) !== "definition" ||
    typeof value.revision !== "string" ||
    !/^[1-9][0-9]{0,18}$/.test(value.revision) ||
    BigInt(value.revision) > 9223372036854775807n
  )
    throw Error("Unsupported agent revision reference.");
  return structuredClone(value);
}
export function parseAgent(resource) {
  const kind = agentKind(resource?.uri),
    annotation = resource?.metadata?.agent_studio;
  if (
    !kind ||
    annotation?.version !== 1 ||
    annotation.kind !== kind ||
    typeof resource.content !== "string" ||
    encoder.encode(resource.content).length > agentRecordLimit
  )
    throw Error(
      "Unsupported agent resource. Inspect or export it from Resources.",
    );
  let value;
  try {
    value = JSON.parse(resource.content);
  } catch {
    throw Error("Invalid agent JSON.");
  }
  const parsed =
    kind === "definition" ? validateAgent(value) : validateAgentRun(value);
  if (kind === "run" && resource.uri.split("/runs/")[0] !== parsed.agent.uri)
    throw Error("Agent run parent does not match its recorded definition.");
  return parsed;
}
export async function agentPage(
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
    (kind === "run" && agentKind(parent) !== "definition") ||
    typeof after !== "string"
  )
    throw Error("Unsupported agent catalog selection.");
  const resources = [],
    seen = new Set();
  let cursor = after;
  for (let page = 0; page < 25; page++) {
    seen.add(cursor);
    const data = await api.resources(namespace, cursor, "all");
    if (
      !Array.isArray(data.resources) ||
      data.resources.length > 100 ||
      typeof data.next !== "string"
    )
      throw Error("Unsupported resource catalog.");
    resources.push(
      ...data.resources.filter(
        (item) =>
          agentKind(item.uri) === kind &&
          (kind !== "run" || item.uri.split("/runs/")[0] === parent),
      ),
    );
    cursor = data.next;
    if (cursor && seen.has(cursor))
      throw Error("Repeated resource cursor. Refresh the library.");
    if (!cursor || resources.length >= 20) break;
  }
  return { resources, next: cursor };
}
