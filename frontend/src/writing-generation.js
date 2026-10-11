import { escapeHtml as esc } from "./message.js";
import { inspectContextReceipt, isContextIdentity } from "./context-receipt.js";
import { contextReceiptSources } from "./context-receipt-view.js";
import {
  compareRevisionText,
  textComparisonMarkup,
} from "./revision-comparison.js";

const encoder = new TextEncoder();
const limit = 1048576;
const modes = ["generate", "rewrite", "continue"];
const validRevision = (value) =>
  typeof value === "string" &&
  /^[1-9][0-9]{0,18}$/.test(value) &&
  BigInt(value) <= 9223372036854775807n;
const text = (value) =>
  typeof value === "string" &&
  !value.includes("\0") &&
  encoder.encode(value).length <= limit;
function counter(value) {
  if (value === undefined || value === null) return null;
  if (!Number.isSafeInteger(value) || value < 0)
    throw Error("Unsupported Writing generation counter.");
  return value;
}

// Keep the exact baseline separately from the mutable editor. Acceptance and
// comparison use identical concatenation, including empty output and newlines.
export function createWritingPreview(
  result,
  { draft, revision, mode, principal },
) {
  if (
    !modes.includes(mode) ||
    !validRevision(revision) ||
    !isContextIdentity(principal) ||
    !/^fyodor:\/\/writing\/(documents|notes|characters|projects)\//.test(
      draft?.uri || "",
    ) ||
    !isContextIdentity(draft?.uri.split("/").at(-1)) ||
    draft.uri.split("/").length !== 5 ||
    typeof draft.title !== "string" ||
    draft.title.includes("\0") ||
    encoder.encode(draft.title).length > 1024 ||
    !text(draft.content) ||
    !text(result?.text) ||
    !isContextIdentity(result.receipt_id)
  )
    throw Error("Unsupported Writing generation preview.");
  return {
    text: result.text,
    receipt_id: result.receipt_id,
    mode,
    target: draft.uri,
    revision,
    principal,
    base: { title: draft.title, content: draft.content },
    prompt_tokens: counter(result.prompt_tokens),
    generated_tokens: counter(result.generated_tokens),
    context_bytes: counter(result.context_bytes),
  };
}
export const writingPreviewText = (preview) =>
  preview.mode === "continue"
    ? preview.base.content + preview.text
    : preview.text;

// A preview cannot be accepted merely because an endpoint returned text. The
// permission-checked native receipt must describe this exact request/output.
export function verifyWritingReceipt(preview, receipt) {
  const report = inspectContextReceipt(receipt),
    metadata = JSON.parse(receipt.metadata_json);
  if (
    receipt.output !== preview.text ||
    metadata.writing_mode !== preview.mode ||
    report.sources[0]?.uri !== preview.target ||
    report.sources[0]?.revision !== preview.revision ||
    report.sources[0]?.layer !== "explicit"
  )
    throw Error("The Writing receipt does not match this preview.");
  for (const key of ["prompt_tokens", "generated_tokens", "context_bytes"]) {
    if (preview[key] !== null && metadata[key] !== preview[key])
      throw Error("The Writing receipt counters do not match this preview.");
  }
  return report;
}
export function writingPreviewAcceptance(
  preview,
  receipt,
  draft,
  revision,
  dirty,
  accepted,
) {
  if (accepted?.receipt_id === preview.receipt_id)
    return "Preview copied to the editor. Save to keep it.";
  if (
    dirty ||
    draft?.uri !== preview.target ||
    revision !== preview.revision ||
    draft.content !== preview.base.content
  )
    return "The editor changed after generation. Save or reload before generating another preview.";
  if (encoder.encode(writingPreviewText(preview)).length > limit)
    return "The proposed text exceeds the 1 MiB document limit.";
  if (!receipt) return "Read the saved receipt before accepting this preview.";
  try {
    verifyWritingReceipt(preview, receipt);
  } catch (error) {
    return error.message;
  }
  return "";
}
export function writingPreviewPanel(state) {
  const preview = state.generated;
  if (!preview) return "";
  const reason = writingPreviewAcceptance(
    preview,
    state.receipt,
    state.draft,
    state.revision,
    state.dirty,
    state.accepted,
  );
  const metric = (value) =>
    value === null ? "Unknown" : value.toLocaleString();
  return `<section class="panel writing-preview" aria-label="Generated prose review">
    <header><div><p class="eyebrow">REVIEW GENERATED PROSE</p><h2>${esc({ generate: "Generated preview", rewrite: "Rewrite preview", continue: "Continuation preview" }[preview.mode])}</h2><p class="muted">From saved revision ${esc(preview.revision)} · ${esc(preview.base.title)}</p></div></header>
    <div class="dataset-stats"><span>${metric(preview.prompt_tokens)} prompt tokens</span><span>${metric(preview.generated_tokens)} generated tokens</span><span>${metric(preview.context_bytes)} context bytes</span></div>
    <pre id="writing-output">${esc(preview.text)}</pre>${preview.text === "" ? '<p class="muted">The model returned empty text.</p>' : ""}
    <div class="resource-toolbar"><button class="primary" data-resource="writing-apply" ${reason ? "disabled" : ""}>${preview.mode === "continue" ? "Append preview to editor" : "Replace editor text with preview"}</button><button data-resource="writing-preview-export">Download preview (.txt)</button><button data-resource="writing-preview-discard">Discard preview</button></div>
    <p id="writing-preview-state" class="muted">${esc(reason || "Review the preview before copying it to the editor.")}</p>
    <details class="writing-preview-comparison"><summary>Compare proposed text with saved text</summary><div class="writing-preview-comparison-body"></div></details>
    <details class="writing-preview-receipt"><summary>Context used and saved receipt</summary><p>Receipt: <code id="writing-preview-receipt-id">${esc(preview.receipt_id)}</code></p><button data-resource="writing-preview-receipt">Read saved receipt again</button>
      ${state.receipt ? `<h3>Context used</h3>${contextReceiptSources(state.receipt)}<details><summary>Executed prompt and sources</summary><pre id="writing-exact-prompt">${esc(state.receipt.prompt)}</pre><pre>${esc(state.receipt.sources_json)}</pre><h4>Model and sampling</h4><pre id="writing-preview-model">${esc(state.receipt.metadata_json)}</pre></details>` : '<p class="muted">The saved receipt is not loaded.</p>'}</details>
  </section>`;
}
export function installWritingPreview(root, state, current) {
  const panel = root.querySelector(".writing-preview");
  if (!panel) return;
  const comparison = panel.querySelector(".writing-preview-comparison");
  // Expensive line work happens only on disclosure. No editor repaint or timers
  // are needed: the baseline and generated output never mutate in-place.
  comparison.addEventListener("toggle", () => {
    if (current() && panel.isConnected && comparison.open)
      comparison.querySelector("div").innerHTML = textComparisonMarkup(
        compareRevisionText(
          state.generated.base.content,
          writingPreviewText(state.generated),
        ),
      );
  });
  root.querySelector(".resource-editor")?.addEventListener("input", () => {
    if (!current() || !panel.isConnected) return;
    const reason = writingPreviewAcceptance(
      state.generated,
      state.receipt,
      state.draft,
      state.revision,
      state.dirty,
      state.accepted,
    );
    panel.querySelector('[data-resource="writing-apply"]').disabled = !!reason;
    panel.querySelector("#writing-preview-state").textContent = reason;
  });
}
