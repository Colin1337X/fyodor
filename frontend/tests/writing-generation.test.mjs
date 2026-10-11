import test from "node:test";
import assert from "node:assert/strict";
import {
  createWritingPreview,
  verifyWritingReceipt,
  writingPreviewText,
  writingPreviewAcceptance,
  writingPreviewPanel,
} from "../src/writing-generation.js";
import {
  compareRevisionText,
  textComparisonMarkup,
} from "../src/revision-comparison.js";
const uri = "fyodor://writing/documents/11111111-2222-4333-8444-555555555555";
const principal = "66666666-7777-4888-8999-aaaaaaaaaaaa";
const id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
const draft = { uri, title: "A <script>story</script>", content: "한글\r\n" };
function preview(mode = "continue", output = "New\n") {
  return createWritingPreview(
    {
      text: output,
      receipt_id: id,
      prompt_tokens: 2,
      generated_tokens: 3,
      context_bytes: 2,
    },
    { draft, revision: "9007199254740993", mode, principal },
  );
}
function receipt(value = preview()) {
  return {
    prompt: "ab\n\nUSER",
    output: value.text,
    metadata_json: JSON.stringify({
      schema: 1,
      context_bytes: 2,
      prompt_tokens: 2,
      generated_tokens: 3,
      writing_mode: value.mode,
    }),
    sources_json: `[{"uri":"${uri}","revision":9007199254740993,"offset":0,"length":2,"original_length":9,"layer":"explicit"}]`,
  };
}
test("Preview retains exact baseline/revision and literal append or replacement, including empty output", () => {
  const value = preview();
  assert.equal(value.revision, "9007199254740993");
  assert.equal(writingPreviewText(value), "한글\r\nNew\n");
  const changed = { ...draft, content: "later" };
  assert.notEqual(changed.content, value.base.content);
  for (const mode of ["generate", "rewrite"])
    assert.equal(writingPreviewText(preview(mode, "")), "");
  assert.equal(writingPreviewText(preview("continue", "")), draft.content);
  const diff = compareRevisionText(
    value.base.content,
    writingPreviewText(value),
  );
  assert.deepEqual(
    diff.rows.filter((row) => row.kind !== "removed").map((row) => row.text),
    writingPreviewText(value).split("\n"),
  );
});
test("Native receipt must bind actual output, mode, first source revision and reported counters", () => {
  const value = preview(),
    actual = receipt(value);
  assert.equal(
    verifyWritingReceipt(value, actual).sources[0].revision,
    value.revision,
  );
  for (const patch of [
    { output: "other" },
    {
      sources_json: actual.sources_json.replace(
        "9007199254740993",
        "9007199254740994",
      ),
    },
    { sources_json: actual.sources_json.replace("explicit", "workspace") },
    { sources_json: "[]" },
    { metadata_json: actual.metadata_json.replace("continue", "rewrite") },
    {
      metadata_json: actual.metadata_json.replace(
        '"generated_tokens":3',
        '"generated_tokens":4',
      ),
    },
  ])
    assert.throws(
      () => verifyWritingReceipt(value, { ...actual, ...patch }),
      /match/,
    );
  assert.throws(
    () => verifyWritingReceipt(value, { ...actual, metadata_json: "null" }),
    /boundary/,
  );
});
test("Acceptance prevents dirty/stale/replaced/oversized editors and duplicate copying; missing receipts fail closed", () => {
  const value = preview(),
    actual = receipt(value),
    accept = (
      d = draft,
      revision = value.revision,
      dirty = false,
      saved = actual,
      accepted = null,
    ) => writingPreviewAcceptance(value, saved, d, revision, dirty, accepted);
  assert.equal(accept(), "");
  assert.match(accept(draft, value.revision, true), /editor changed/);
  assert.match(accept({ ...draft, content: "edited" }), /editor changed/);
  assert.match(accept(draft, "2"), /editor changed/);
  assert.match(
    accept({ ...draft, uri: uri.replace("documents", "notes") }),
    /editor changed/,
  );
  assert.match(
    accept(draft, value.revision, false, null),
    /Read the saved receipt/,
  );
  assert.match(
    accept(draft, value.revision, true, actual, { receipt_id: id }),
    /copied/,
  );
  const big = preview("continue", "x");
  big.base.content = "한".repeat(349525) + "a";
  assert.match(
    writingPreviewAcceptance(
      big,
      actual,
      { ...draft, content: big.base.content },
      big.revision,
      false,
      null,
    ),
    /1 MiB/,
  );
});
test("Untrusted native responses reject malformed identity, NUL, UTF-8 bounds and counters without inventing usage", () => {
  const context = { draft, revision: "1", mode: "generate", principal },
    result = { text: "", receipt_id: id };
  assert.equal(createWritingPreview(result, context).generated_tokens, null);
  for (const bad of [
    { text: "a\0b" },
    { text: "한".repeat(349526) },
    { receipt_id: "00000000-0000-0000-0000-000000000000" },
    { prompt_tokens: -1 },
    { generated_tokens: 1.5 },
  ])
    assert.throws(() => createWritingPreview({ ...result, ...bad }, context));
  for (const bad of [
    { revision: "9223372036854775808" },
    { revision: 1 },
    { mode: "script" },
    { principal: "other" },
    {
      draft: {
        ...draft,
        uri: uri.replace(uri.split("/").at(-1), "-".repeat(36)),
      },
    },
  ])
    assert.throws(() => createWritingPreview(result, { ...context, ...bad }));
});
test("Review escapes output/title/prompt and exposes actual source ranges, unknown usage and bounded comparison", () => {
  const value = preview("rewrite", "<script>x</script>"),
    actual = receipt(value);
  const html = writingPreviewPanel({
    generated: value,
    receipt: actual,
    draft,
    revision: value.revision,
    dirty: false,
    accepted: null,
  });
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;x/);
  assert.match(html, /Prefix included/);
  assert.match(html, /9007199254740993/);
  assert.match(html, /3 generated tokens/);
  const empty = createWritingPreview(
    { text: "", receipt_id: id },
    { draft, revision: "1", mode: "rewrite", principal },
  );
  const missing = writingPreviewPanel({
    generated: empty,
    receipt: null,
    draft,
    revision: "1",
    dirty: false,
  });
  assert.match(missing, /Unknown generated tokens/);
  assert.match(missing, /empty text/);
  assert.match(missing, /writing-apply" disabled/);
  assert.match(
    textComparisonMarkup(compareRevisionText("x".repeat(1048577), "")),
    /1 MiB/,
  );
});
