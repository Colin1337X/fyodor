import test from "node:test";
import assert from "node:assert/strict";
import {
  compareRevisionText,
  revisionComparisonMarkup,
  comparisonLimits,
} from "../src/revision-comparison.js";

test("Exact line operations reconstruct both revisions including blanks, CRLF and final newline", () => {
  for (const [before, after] of [
    ["", ""],
    ["", "first\n"],
    ["a\nb\na\n", "a\na\nb\n"],
    ["a\r\n😀\n", "a\n😀"],
    ["removed", ""],
    ["\n", "\n\n"],
  ]) {
    const diff = compareRevisionText(before, after);
    assert.equal(diff.available, true);
    assert.deepEqual(
      diff.rows.filter((row) => row.kind !== "added").map((row) => row.text),
      before === "" ? [] : before.split("\n"),
    );
    assert.deepEqual(
      diff.rows.filter((row) => row.kind !== "removed").map((row) => row.text),
      after === "" ? [] : after.split("\n"),
    );
    assert.equal(
      diff.added,
      diff.rows.filter((row) => row.kind === "added").length,
    );
    assert.equal(
      diff.removed,
      diff.rows.filter((row) => row.kind === "removed").length,
    );
    assert.deepEqual(diff, compareRevisionText(before, after));
  }
});
test("Long unchanged edges avoid quadratic work and omitted context retains exact source line numbers", () => {
  const prefix = Array.from({ length: 700 }, (_, i) => `Line ${i}`).join("\n");
  const diff = compareRevisionText(
    prefix + "\nold\nend",
    prefix + "\nnew\nend",
  );
  assert.equal(diff.available, true);
  assert.equal(diff.added, 1);
  assert.equal(diff.removed, 1);
  assert.equal(diff.displayed[0].kind, "omitted");
  assert.equal(diff.displayed[0].count, 697);
  assert.deepEqual(
    diff.rows.find((row) => row.kind === "removed"),
    { kind: "removed", text: "old", oldLine: 701, newLine: null },
  );
  assert.deepEqual(
    diff.rows.find((row) => row.kind === "added"),
    { kind: "added", text: "new", oldLine: null, newLine: 701 },
  );
});
test("Byte, line, changed-region and rendered-row bounds distinguish unavailable comparisons from exact counts", () => {
  assert.equal(
    compareRevisionText("😀".repeat(comparisonLimits.bytes / 4 + 1), "")
      .available,
    false,
  );
  assert.equal(
    compareRevisionText("\n".repeat(comparisonLimits.lines), "").available,
    false,
  );
  assert.equal(
    compareRevisionText(
      Array(501).fill("old").join("\n"),
      Array(501).fill("new").join("\n"),
    ).available,
    false,
  );
  const removed = compareRevisionText(Array(1100).fill("old").join("\n"), "");
  assert.equal(removed.available, true);
  assert.equal(removed.removed, 1100);
  assert.equal(removed.displayed, null);
  assert.throws(() => compareRevisionText(null, ""), /requires text/);
});
test("Revision presentation escapes titles/text, labels direction and unsaved state, and exposes CR changes without color dependence", () => {
  const snapshot = {
    saved: { title: "<img src=x>", content: "<script>old</script>\r\n" },
    draft: { title: "New & title", content: "<script>new</script>\n" },
    savedRevision: "9007199254740993",
    editorRevision: "9007199254740994",
    dirty: true,
  };
  const html = revisionComparisonMarkup(snapshot);
  assert.match(html, /Saved revision 9007199254740993 → unsaved editor/);
  assert.match(html, /&lt;img src=x&gt;/);
  assert.match(html, /&lt;script&gt;new&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<script>|<img/);
  assert.match(html, />Removed</);
  assert.match(html, />Added</);
  assert.match(html, /\[CR\]/);
  assert.match(
    revisionComparisonMarkup({
      ...snapshot,
      draft: snapshot.saved,
      dirty: false,
    }),
    /Text unchanged/,
  );
});
