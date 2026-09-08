// Rebuild a narrow data shape when restoring untrusted persisted/imported data.
export function normalizeChat(value, newId) {
  if (!value || typeof value !== "object" || !Array.isArray(value.messages) || value.messages.length > 2000)
    throw new Error("A chat must contain a messages array (at most 2,000 entries).");
  const messages = value.messages.map(message => {
    if (!message || !["user", "assistant", "tool"].includes(message.role) || typeof message.text !== "string" || message.text.length > 2000000)
      throw new Error("Each message needs a supported role and text.");
    return { role: message.role, text: message.text, tokens: Number.isFinite(message.tokens) ? Math.max(0, message.tokens) : 0,
      seconds: Number.isFinite(message.seconds) ? Math.max(0, message.seconds) : 0 };
  });
  return { id: newId, title: typeof value.title === "string" ? value.title.slice(0, 160) : "Imported chat",
    created: Number.isFinite(value.created) && value.created > 0 && value.created < 8640000000000000 ? value.created : Date.now(),
    messages, draft: typeof value.draft === "string" ? value.draft.slice(0, 200000) : "" };
}

// Format the complete conversation: the native endpoint retains no KV session.
// Gemma's empty thought channel matches the backend text compatibility adapter.
export function formatConversation(messages, architecture) {
  const gemma = architecture === "gemma4";
  return messages.map(message => {
    const role = message.role === "assistant" && gemma ? "model" : message.role;
    return gemma ? `<|turn>${role}\n${message.content}<turn|>\n` : `${role}: ${message.content}\n`;
  }).join("") + (gemma ? "<|turn>model\n<|channel>thought\n<channel|>" : "assistant: ");
}

export function generationOptions(params) {
  const result = {};
  for (const [key, low, high, integer] of [["temperature", 0, 5, false], ["top_p", .000001, 1, false],
    ["top_k", 0, 1000000, true], ["max_tokens", 1, 4096, true]]) {
    const value = Number(params[key]);
    if (!Number.isFinite(value) || value < low || value > high || (integer && !Number.isInteger(value)))
      throw new Error(`Invalid ${key.replaceAll("_", " ")}.`);
    result[key] = value;
  }
  if (params.seed !== "") {
    const seed = Number(params.seed);
    if (!Number.isSafeInteger(seed) || seed < 0) throw new Error("Seed must be a nonnegative safe integer.");
    result.seed = seed;
  }
  return result;
}
