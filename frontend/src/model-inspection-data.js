import { escapeHtml as esc } from "./message.js";

export const tokenizerTextBytes = 65536,
  tokenizerResultLimit = 131072;
const providers = new Set(["cpu", "cuda", "vulkan", "rocm", "mlx", "external"]);
export function inspectedModelId(value) {
  if (!Number.isSafeInteger(value) || value < 1)
    throw Error("Choose a loaded model with a valid runtime ID.");
  return value;
}
const text = (value) =>
  typeof value === "string" && value.length <= 8192 ? value : null;
const count = (value) =>
  Number.isSafeInteger(value) && value >= 0 ? value : null;
export function modelObservation(model, runtime) {
  const id = inspectedModelId(model?.id);
  const observation = { id };
  for (const key of [
    "path",
    "format",
    "architecture",
    "producer",
    "execution_error",
    "generation_error",
  ])
    observation[key] = text(model[key]);
  for (const key of [
    "file_size",
    "format_version",
    "tensor_count",
    "metadata_count",
    "data_offset",
  ])
    observation[key] = count(model[key]);
  for (const key of [
    "inference_supported",
    "generation_supported",
    "draft_supported",
  ])
    observation[key] = model[key] === true;
  observation.capabilities = Array.isArray(model.capabilities)
    ? model.capabilities
        .filter((value) => typeof value === "string" && value.length <= 128)
        .slice(0, 32)
    : [];
  const active = runtime?.models?.find((item) => item.model_id === id);
  observation.compute = providers.has(active?.compute) ? active.compute : null;
  observation.selectable = active?.selectable === true;
  observation.compiled = Object.fromEntries(
    ["cpu", "cuda", "vulkan", "rocm", "mlx"].map((provider) => [
      provider,
      typeof runtime?.compiled?.[provider] === "boolean"
        ? runtime.compiled[provider]
        : null,
    ]),
  );
  return observation;
}
export function nativeModelInfo(response, id) {
  if (
    response?.version !== 1 ||
    response.ok !== true ||
    response.model_id !== inspectedModelId(id)
  )
    throw Error("Unsupported model information response.");
  const result = { model_id: id };
  for (const key of ["architecture", "compute"]) {
    if (response[key] != null && text(response[key]) === null)
      throw Error("Unsupported model information text.");
    result[key] = text(response[key]);
  }
  if (result.compute !== null && !providers.has(result.compute))
    throw Error("Unsupported compute provider.");
  for (const key of [
    "context_length",
    "embedding_length",
    "layers",
    "attention_heads",
    "kv_heads",
    "vocabulary_size",
    "experts",
    "sliding_window",
  ]) {
    if (response[key] != null && count(response[key]) === null)
      throw Error("Unsupported model information counter.");
    result[key] = count(response[key]);
  }
  if (
    response.mtp_assistant != null &&
    typeof response.mtp_assistant !== "boolean"
  )
    throw Error("Unsupported assistant flag.");
  result.mtp_assistant =
    typeof response.mtp_assistant === "boolean" ? response.mtp_assistant : null;
  return result;
}
export function tokenizerText(text) {
  if (
    typeof text !== "string" ||
    text.includes("\0") ||
    text.length > tokenizerTextBytes ||
    new TextEncoder().encode(text).length > tokenizerTextBytes
  )
    throw Error("Use text without null characters, up to 64 KiB of UTF-8.");
  return text;
}
export function tokenizerResult(response, id) {
  if (
    response?.version !== 1 ||
    response.ok !== true ||
    response.model_id !== inspectedModelId(id) ||
    !Number.isSafeInteger(response.count) ||
    response.count < 0 ||
    response.count > tokenizerResultLimit ||
    !Array.isArray(response.tokens) ||
    response.tokens.length !== response.count ||
    response.tokens.some(
      (value) => !Number.isInteger(value) || value < 0 || value > 4294967295,
    )
  )
    throw Error("Unsupported tokenizer result.");
  return { count: response.count, tokens: [...response.tokens] };
}
const field = (label, value) =>
  `<div><dt>${esc(label)}</dt><dd>${esc(value ?? "Not reported")}</dd></div>`;
export function modelObservationMarkup(model, info = null) {
  const registry = [
    ["Runtime model ID", model.id],
    ["Source path", model.path],
    ["Format", model.format],
    ["Format version", model.format_version],
    ["Architecture", info?.architecture ?? model.architecture],
    ["File size (bytes)", model.file_size],
    ["Tensors", model.tensor_count],
    ["Metadata entries", model.metadata_count],
    ["Data offset (bytes)", model.data_offset],
    ["Producer", model.producer],
    [
      "Capabilities",
      model.capabilities.length ? model.capabilities.join(", ") : null,
    ],
    ["Active compute", info?.compute ?? model.compute],
  ];
  const shape = [
    ["Context tokens", info?.context_length],
    ["Embedding size", info?.embedding_length],
    ["Layers", info?.layers],
    ["Attention heads", info?.attention_heads],
    ["KV heads", info?.kv_heads],
    ["Vocabulary size", info?.vocabulary_size],
    ["Experts", info?.experts],
    ["Sliding window", info?.sliding_window],
    [
      "MTP assistant",
      info?.mtp_assistant === null || !info
        ? "Not reported"
        : info.mtp_assistant
          ? "Yes"
          : "No",
    ],
  ];
  return `<h3>Loaded file</h3><dl class="model-inspection-fields">${registry.map(([label, value]) => field(label, value)).join("")}</dl>${info ? `<h3>Native execution shape</h3><dl class="model-inspection-fields">${shape.map(([label, value]) => field(label, value)).join("")}</dl>` : ""}<h3>Compiled providers</h3><dl class="model-inspection-fields">${Object.entries(
    model.compiled,
  )
    .map(([provider, compiled]) =>
      field(
        provider.toUpperCase(),
        compiled === null ? null : compiled ? "Included" : "Not included",
      ),
    )
    .join("")}</dl>${[
    ["Tensor execution", model.execution_error],
    ["Text generation", model.generation_error],
  ]
    .filter(([, error]) => Boolean(error))
    .map(
      ([label, error]) =>
        `<p class="muted"><b>${esc(label)}:</b> ${esc(error)}</p>`,
    )
    .join("")}`;
}
