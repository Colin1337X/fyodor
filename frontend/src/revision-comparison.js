import { escapeHtml as esc } from "./message.js";

const encoder = new TextEncoder();
export const comparisonLimits = Object.freeze({
  bytes: 1048576,
  lines: 20000,
  cells: 250000,
  renderedLines: 1000,
});
const linesOf = (text) => (text === "" ? [] : text.split("\n"));

// Trim unchanged edges before bounded LCS. A hard cell budget avoids quadratic
// work on unrelated novels. Failure never masquerades as a complete comparison.
export function compareRevisionText(before, after) {
  if (typeof before !== "string" || typeof after !== "string")
    throw Error("Comparison requires text.");
  if (
    [before, after].some(
      (text) =>
        text.length > comparisonLimits.bytes ||
        encoder.encode(text).length > comparisonLimits.bytes,
    )
  )
    return {
      available: false,
      reason: "Text exceeds the 1 MiB comparison limit.",
    };
  const a = linesOf(before),
    b = linesOf(after);
  if (a.length > comparisonLimits.lines || b.length > comparisonLimits.lines)
    return {
      available: false,
      reason: "Text exceeds the 20,000-line comparison limit.",
    };
  let prefix = 0,
    suffix = 0;
  while (prefix < a.length && prefix < b.length && a[prefix] === b[prefix])
    prefix++;
  while (
    suffix < a.length - prefix &&
    suffix < b.length - prefix &&
    a[a.length - 1 - suffix] === b[b.length - 1 - suffix]
  )
    suffix++;
  const oldCount = a.length - prefix - suffix,
    newCount = b.length - prefix - suffix;
  if (oldCount * newCount > comparisonLimits.cells)
    return {
      available: false,
      reason: "The changed region exceeds the detailed comparison limit.",
    };
  const width = newCount + 1,
    table = new Uint32Array((oldCount + 1) * width);
  for (let i = oldCount - 1; i >= 0; i--)
    for (let j = newCount - 1; j >= 0; j--)
      table[i * width + j] =
        a[prefix + i] === b[prefix + j]
          ? 1 + table[(i + 1) * width + j + 1]
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
  const rows = [];
  let oldLine = 0,
    newLine = 0,
    added = 0,
    removed = 0;
  const put = (kind, text) => {
    rows.push({
      kind,
      text,
      oldLine: kind === "added" ? null : ++oldLine,
      newLine: kind === "removed" ? null : ++newLine,
    });
    if (kind === "added") added++;
    if (kind === "removed") removed++;
  };
  for (let n = 0; n < prefix; n++) put("same", a[n]);
  let i = 0,
    j = 0;
  while (i < oldCount || j < newCount) {
    if (i < oldCount && j < newCount && a[prefix + i] === b[prefix + j]) {
      put("same", a[prefix + i]);
      i++;
      j++;
    } else if (
      i < oldCount &&
      (j === newCount || table[(i + 1) * width + j] >= table[i * width + j + 1])
    )
      put("removed", a[prefix + i++]);
    else put("added", b[prefix + j++]);
  }
  for (let n = a.length - suffix; n < a.length; n++) put("same", a[n]);
  // Context is display-only. Keep the complete operation list for exact counts
  // and reconstruction; omitted unchanged lines receive explicit separators.
  const keep = new Set();
  rows.forEach((row, index) => {
    if (row.kind !== "same")
      for (
        let n = Math.max(0, index - 3);
        n <= Math.min(rows.length - 1, index + 3);
        n++
      )
        keep.add(n);
  });
  const displayed = [];
  let skipped = 0;
  for (let n = 0; n < rows.length; n++) {
    if (!keep.has(n)) {
      skipped++;
      continue;
    }
    if (skipped) {
      displayed.push({ kind: "omitted", count: skipped });
      skipped = 0;
    }
    displayed.push(rows[n]);
  }
  if (skipped && keep.size) displayed.push({ kind: "omitted", count: skipped });
  return {
    available: true,
    added,
    removed,
    rows,
    displayed:
      displayed.length <= comparisonLimits.renderedLines ? displayed : null,
  };
}

export function revisionComparisonMarkup(snapshot) {
  const result = compareRevisionText(
    snapshot.saved.content,
    snapshot.draft.content,
  );
  const titleChanged = snapshot.saved.title !== snapshot.draft.title;
  const title = titleChanged
    ? `<div class="revision-title-change"><p><b>Saved title:</b> ${esc(snapshot.saved.title)}</p><p><b>Editor title:</b> ${esc(snapshot.draft.title)}</p></div>`
    : '<p class="muted">Title unchanged.</p>';
  const label = `Saved revision ${snapshot.savedRevision} → ${snapshot.dirty ? "unsaved editor" : "editor revision " + snapshot.editorRevision}`;
  if (!result.available)
    return `<p>${esc(label)}</p>${title}<p role="status">${esc(result.reason)} Read the saved text above and the editor directly.</p>`;
  const summary =
    result.added || result.removed
      ? `${result.added} added lines · ${result.removed} removed lines`
      : "Text unchanged.";
  const changes =
    result.displayed === null
      ? "<p>Too many changed lines to display. Read the saved text above and the editor directly.</p>"
      : result.displayed.length
        ? `<ol class="revision-lines" aria-label="Line comparison">${result.displayed
            .map((row) =>
              row.kind === "omitted"
                ? `<li class="revision-omitted">${row.count} unchanged lines omitted</li>`
                : `<li class="revision-${row.kind}"><span class="revision-line-number" aria-hidden="true">${row.oldLine ?? "–"} / ${row.newLine ?? "–"}</span><span class="revision-line-kind">${row.kind === "same" ? "Unchanged" : row.kind === "added" ? "Added" : "Removed"}</span><code>${esc(row.text.endsWith("\r") ? row.text.slice(0, -1) : row.text) || '<span class="muted">(empty line)</span>'}${row.text.endsWith("\r") ? '<span class="revision-cr"> [CR]</span>' : ""}</code></li>`,
            )
            .join("")}</ol>`
        : "";
  return `<p>${esc(label)}</p>${title}<p role="status">${esc(summary)}</p>${changes}`;
}

// Mount on nodes owned by this paint, not the persistent root: rerenders must
// not accumulate listeners. Late debounce callbacks cannot paint another view.
export function installRevisionComparison(root, { getSnapshot, current }) {
  const history = root.querySelector(".resource-history"),
    editor = root.querySelector(".resource-editor");
  if (!history || !editor || !getSnapshot()?.saved) return;
  const details = document.createElement("details");
  details.className = "revision-comparison";
  details.innerHTML =
    '<summary>Compare saved revision with editor</summary><div class="revision-comparison-body"></div>';
  history.append(details);
  const body = details.querySelector("div");
  let timer;
  function render() {
    if (details.isConnected && current() && details.open)
      body.innerHTML = revisionComparisonMarkup(getSnapshot());
  }
  details.addEventListener("toggle", () => {
    clearTimeout(timer);
    render();
  });
  editor.addEventListener("input", (event) => {
    if (!event.target.matches("input,textarea") || !details.open) return;
    clearTimeout(timer);
    body.innerHTML = '<p role="status">Updating comparison…</p>';
    timer = setTimeout(render, 180);
  });
}
