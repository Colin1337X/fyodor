import { escapeHtml as esc } from "./message.js";
import {
  inspectedModelId,
  modelObservation,
  nativeModelInfo,
  modelObservationMarkup,
  tokenizerText,
  tokenizerResult,
} from "./model-inspection-data.js";

// Observation is window-local, never a persistent identity or model annotation.
// Modal request tickets prevent dismissed/replaced results from repainting.
export function installModelInspector({ api, getState }) {
  const dialog = document.createElement("dialog");
  dialog.id = "model-inspector";
  dialog.setAttribute("aria-labelledby", "model-inspector-title");
  document.body.append(dialog);
  let ticket = 0,
    returnFocus = null,
    model = null,
    info = null,
    result = null,
    resultText = "",
    loading = false,
    tokenBusy = false;
  function skeleton() {
    dialog.innerHTML = `<section class="model-inspector-card"><header><div><p class="eyebrow">MODEL INSPECTION</p><h2 id="model-inspector-title">${esc(model.path?.split(/[\\/]/).pop() || "Loaded model")}</h2></div><button id="model-inspector-close" aria-label="Close model inspection">×</button></header><p class="fine-print">Loaded model ID applies to this runtime.</p><p id="model-info-status" role="status" aria-live="polite"></p><div id="model-info-fields">${modelObservationMarkup(model)}</div><section class="model-tokenizer"><h3>Tokenizer</h3><label for="model-tokenizer-text">Text to tokenize</label><textarea id="model-tokenizer-text" rows="4" maxlength="65536" placeholder="Try a phrase with this model’s tokenizer…"></textarea><div class="resource-toolbar"><button id="model-tokenize">Tokenize text</button><button id="model-token-export" disabled>Download token IDs</button></div><p id="model-tokenizer-status" role="status" aria-live="polite"></p><pre id="model-tokenizer-result" hidden></pre><p class="fine-print">Raw text is tokenized without a chat template. Model-specific special tokens may be included.</p></section></section>`;
    dialog
      .querySelector("#model-inspector-close")
      .addEventListener("click", () => dialog.close());
    dialog
      .querySelector("#model-tokenize")
      .addEventListener("click", () => void tokenize());
    dialog
      .querySelector("#model-token-export")
      .addEventListener("click", download);
    dialog
      .querySelector("#model-tokenizer-text")
      .addEventListener("input", () => {
        result = null;
        resultText = "";
        renderTokens();
        dialog.querySelector("#model-tokenizer-status").textContent =
          "Text changed. Tokenize to update the result.";
      });
    controls();
  }
  function controls() {
    const supported =
      model.selectable && (model.generation_supported || model.draft_supported);
    dialog.querySelector("#model-tokenize").disabled =
      loading || tokenBusy || !supported;
    dialog.querySelector("#model-tokenizer-text").disabled = tokenBusy;
    dialog.querySelector("#model-token-export").disabled = !result || tokenBusy;
    if (!supported)
      dialog.querySelector("#model-tokenizer-status").textContent =
        "Tokenization is unavailable for this loaded model.";
  }
  function renderTokens() {
    const output = dialog.querySelector("#model-tokenizer-result");
    output.hidden = !result;
    output.textContent = result ? result.tokens.slice(0, 256).join(", ") : "";
    controls();
  }
  async function tokenize() {
    if (loading || tokenBusy) return;
    const current = ticket,
      status = dialog.querySelector("#model-tokenizer-status");
    result = null;
    resultText = "";
    renderTokens();
    try {
      const text = tokenizerText(
        dialog.querySelector("#model-tokenizer-text").value,
      );
      tokenBusy = true;
      controls();
      status.className = "muted";
      status.textContent = "Tokenizing…";
      const response = tokenizerResult(
        await api.tokenize(model.id, text),
        model.id,
      );
      if (ticket !== current || !dialog.open) return;
      result = response;
      resultText = text;
      status.textContent = `${result.count} tokens${result.count > 256 ? " · first 256 IDs shown; download includes all IDs" : ""}`;
    } catch (error) {
      if (ticket === current && dialog.open) {
        status.className = "danger-action";
        status.textContent = error.message;
      }
    } finally {
      if (ticket === current) {
        tokenBusy = false;
        renderTokens();
      }
    }
  }
  function download() {
    if (!result || tokenBusy) return;
    // This explicit local export includes the user's submitted text; never
    // download implicitly or persist it in app/session storage.
    const record = {
      schema: 1,
      kind: "model-tokenization",
      model: {
        id: model.id,
        path: model.path,
        file_size: model.file_size,
        format: model.format,
      },
      text: resultText,
      ...result,
    };
    const url = URL.createObjectURL(
        new Blob([JSON.stringify(record, null, 2)], {
          type: "application/json",
        }),
      ),
      a = document.createElement("a");
    a.href = url;
    a.download = "fyodor-tokenization.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function open(id, origin) {
    if (document.querySelector("dialog[open]")) return;
    const state = getState(),
      loaded = state.models.find((item) => item.id === inspectedModelId(id));
    if (!loaded)
      throw Error("This model is no longer loaded. Refresh the model list.");
    model = modelObservation(loaded, state.runtime);
    info = null;
    result = null;
    resultText = "";
    loading = true;
    tokenBusy = false;
    returnFocus = origin || document.activeElement;
    const current = ++ticket;
    skeleton();
    dialog.showModal();
    dialog.querySelector("#model-inspector-close").focus();
    const status = dialog.querySelector("#model-info-status");
    status.textContent = model.selectable
      ? "Reading native execution details…"
      : "Native execution details are unavailable for this loaded file.";
    try {
      if (model.selectable) {
        const response = await api.modelInfo(model.id);
        if (ticket !== current || !dialog.open) return;
        info = nativeModelInfo(response, model.id);
        dialog.querySelector("#model-info-fields").innerHTML =
          modelObservationMarkup(model, info);
        status.textContent = "Native execution details loaded.";
      }
    } catch (error) {
      if (ticket === current && dialog.open) {
        status.className = "danger-action";
        status.textContent = error.message;
      }
    } finally {
      if (ticket === current) {
        loading = false;
        controls();
      }
    }
  }
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !event.isComposing) {
      event.preventDefault();
      dialog.close();
    }
  });
  dialog.addEventListener("close", () => {
    if (dialog.open) return;
    ticket++;
    loading = false;
    tokenBusy = false;
    result = null;
    resultText = "";
    dialog.replaceChildren();
    if (returnFocus?.isConnected && !returnFocus.disabled) returnFocus.focus();
    else document.querySelector("#open-switcher")?.focus();
  });
  return { open };
}
