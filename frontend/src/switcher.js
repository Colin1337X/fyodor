// Native modal semantics own focus trapping. Window-only result references are
// bounded and escaped; typed text can never become an executable command.
import { escapeHtml as esc } from "./message.js";
import { filterSwitcherItems, searchSavedTitles } from "./switcher-data.js";

export function installSwitcher({
  button,
  api,
  getItems,
  execute,
  openResource,
  onOpen,
}) {
  const dialog = document.createElement("dialog");
  dialog.id = "quick-switcher";
  dialog.setAttribute("aria-labelledby", "switcher-title");
  document.body.append(dialog);
  dialog.innerHTML = `<section class="switcher-card"><header><div><p class="eyebrow">FIND AND OPEN</p><h2 id="switcher-title">Quick switcher</h2></div><button id="switcher-close" aria-label="Close quick switcher">×</button></header>
    <div class="switcher-search"><label for="switcher-query">Search</label><input id="switcher-query" type="search" maxlength="256" autocomplete="off" placeholder="Workspaces or conversations…"><div class="resource-toolbar" role="group" aria-label="Search type"><button data-switcher-mode="commands" aria-pressed="true">Workspaces & chats</button><button data-switcher-mode="resources" aria-pressed="false">Saved resource titles</button></div>
    <div id="switcher-resource-controls" hidden><label>Namespace<input id="switcher-namespace" value="workspace" maxlength="64"></label><button id="switcher-search">Search saved titles</button><p class="fine-print">Search saved titles and resource URIs in this namespace.</p></div>
    <p id="switcher-status" role="status" aria-live="polite"></p></div><ul id="switcher-results" aria-label="Matches"></ul><footer><span>↑ ↓ to choose · Enter to open · Esc to close</span><kbd>Ctrl / Cmd + K</kbd></footer></section>`;
  const query = dialog.querySelector("#switcher-query"),
    namespace = dialog.querySelector("#switcher-namespace"),
    status = dialog.querySelector("#switcher-status"),
    list = dialog.querySelector("#switcher-results");
  let mode = "commands",
    items = [],
    next = "",
    completedQuery = "",
    completedNamespace = "",
    busy = false,
    generation = 0,
    returnFocus = null,
    selection = null,
    focusPending = false;
  function controls() {
    dialog.querySelector("#switcher-resource-controls").hidden =
      mode !== "resources";
    query.placeholder =
      mode === "resources"
        ? "Saved title or fyodor:// URI…"
        : "Workspaces or conversations…";
    dialog.querySelectorAll("[data-switcher-mode]").forEach((n) => {
      n.setAttribute("aria-pressed", String(n.dataset.switcherMode === mode));
      n.disabled = busy;
    });
    query.disabled = busy;
    namespace.disabled = busy;
    dialog.querySelector("#switcher-search").disabled = busy;
  }
  function render() {
    controls();
    list.innerHTML =
      items
        .map(
          (item, index) =>
            `<li><button data-switcher-index="${index}" ${busy ? "disabled" : ""}><b>${esc(mode === "resources" ? item.title || "Untitled" : item.label)}</b><small>${esc(mode === "resources" ? item.namespace + " · revision " + item.revision : item.detail)}</small>${mode === "resources" ? `<span>${esc(item.uri)}</span>` : ""}</button></li>`,
        )
        .join("") +
      (mode === "resources" && next
        ? `<li><button id="switcher-more" ${busy ? "disabled" : ""}>Continue searching</button></li>`
        : "");
    list
      .querySelectorAll("[data-switcher-index]")
      .forEach((n) =>
        n.addEventListener("click", () =>
          choose(Number(n.dataset.switcherIndex)),
        ),
      );
    list
      .querySelector("#switcher-more")
      ?.addEventListener("click", () => void search(true));
  }
  function commands() {
    try {
      const found = filterSwitcherItems(getItems(), query.value);
      items = found.items;
      status.className = "muted";
      status.textContent = found.total
        ? `${items.length} of ${found.total} matches`
        : "No matching workspaces or conversations.";
    } catch (error) {
      items = [];
      errorMessage(error.message);
    }
    render();
  }
  function errorMessage(message) {
    status.className = "danger-action";
    status.textContent = message;
  }
  async function search(more = false) {
    if (busy) return;
    const ticket = ++generation;
    const text = more ? completedQuery : query.value,
      ns = more ? completedNamespace : namespace.value;
    busy = true;
    status.className = "muted";
    status.textContent = "Searching saved titles…";
    render();
    try {
      const found = await searchSavedTitles(api, {
        namespace: ns,
        query: text,
        after: more ? next : "",
      });
      if (ticket !== generation || !dialog.open) return;
      items = found.items;
      next = found.next;
      completedQuery = text;
      completedNamespace = ns;
      status.textContent = `${items.length} matches for “${text}” · ${found.scanned} records scanned${next ? " · more records available" : ""}`;
    } catch (error) {
      if (ticket === generation && dialog.open) {
        items = [];
        next = "";
        errorMessage(error.message);
      }
    } finally {
      if (ticket === generation) {
        busy = false;
        render();
        if (dialog.open) query.focus();
      }
    }
  }
  function choose(index) {
    if (busy || !items[index]) return;
    try {
      // Queue guarded inspection before closing; a dirty/busy editor keeps this
      // dialog open and is never silently replaced by a search result.
      if (mode === "resources") {
        openResource(items[index]);
        dialog.close();
      } else {
        const item = items[index];
        dialog.close();
        execute(item);
      }
    } catch (error) {
      if (!dialog.open) dialog.showModal();
      errorMessage(error.message);
    }
  }
  function restoreFocus() {
    const target =
      returnFocus?.isConnected &&
      !returnFocus.disabled &&
      !returnFocus.closest("[inert]")
        ? returnFocus
        : button;
    target.focus();
    if (selection && target === returnFocus)
      try {
        target.setSelectionRange(
          selection.start,
          selection.end,
          selection.direction,
        );
      } catch {}
  }
  function open() {
    if (dialog.open) {
      query.focus();
      return;
    }
    if (document.querySelector("dialog[open]")) return;
    // Native close restores the element before its queued close event. Restore
    // our saved caret before a same-task reopen captures that element again.
    if (focusPending) restoreFocus();
    returnFocus = document.activeElement;
    selection =
      returnFocus && typeof returnFocus.selectionStart === "number"
        ? {
            start: returnFocus.selectionStart,
            end: returnFocus.selectionEnd,
            direction: returnFocus.selectionDirection,
          }
        : null;
    onOpen();
    generation++;
    busy = false;
    mode = "commands";
    query.value = "";
    items = [];
    next = "";
    dialog.showModal();
    focusPending = true;
    commands();
    query.focus();
  }
  button.addEventListener("click", open);
  dialog
    .querySelector("#switcher-close")
    .addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    // The native close event is queued. A same-task reopen owns the new
    // generation and focus; the previous dismissal must not invalidate it.
    if (dialog.open) return;
    generation++;
    busy = false;
    focusPending = false;
    restoreFocus();
  });
  dialog.querySelectorAll("[data-switcher-mode]").forEach((n) =>
    n.addEventListener("click", () => {
      if (busy) return;
      mode = n.dataset.switcherMode;
      items = [];
      next = "";
      status.textContent =
        mode === "resources" ? "Enter a title, then search." : "";
      if (mode === "commands") commands();
      else render();
      query.focus();
    }),
  );
  query.addEventListener("input", () => {
    if (mode === "commands") commands();
    else {
      items = [];
      next = "";
      status.textContent = "Search to update the matches.";
      render();
    }
  });
  namespace.addEventListener("input", () => {
    items = [];
    next = "";
    status.textContent = "Search to update the matches.";
    render();
  });
  dialog
    .querySelector("#switcher-search")
    .addEventListener("click", () => void search());
  dialog.addEventListener("keydown", (event) => {
    if (event.isComposing) return;
    // A populated native search field can consume Escape to clear itself.
    // The modal's advertised shortcut must dismiss it on the first press.
    if (event.key === "Escape") {
      event.preventDefault();
      dialog.close();
      return;
    }
    if (event.target === query) {
      if (event.key === "Enter") {
        event.preventDefault();
        mode === "resources" ? void search() : choose(0);
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        list.querySelector("button:not(:disabled)")?.focus();
      }
    } else if (
      event.target.closest("#switcher-results") &&
      ["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)
    ) {
      event.preventDefault();
      const buttons = [...list.querySelectorAll("button:not(:disabled)")],
        index = buttons.indexOf(document.activeElement);
      if (event.key === "ArrowUp" && index === 0) {
        query.focus();
        return;
      }
      const nextIndex =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? buttons.length - 1
            : Math.max(
                0,
                Math.min(
                  buttons.length - 1,
                  index + (event.key === "ArrowDown" ? 1 : -1),
                ),
              );
      buttons[nextIndex]?.focus();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (
      (event.ctrlKey || event.metaKey) &&
      !event.altKey &&
      !event.shiftKey &&
      !event.isComposing &&
      event.key.toLowerCase() === "k"
    ) {
      if (document.querySelector("dialog[open]") && !dialog.open) return;
      event.preventDefault();
      open();
    }
  });
  return { open };
}
