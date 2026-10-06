import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { newDataset } from "../src/dataset-data.js";
import {
  newEvaluationCase,
  parseEvaluation,
  defaultEvaluationSettings,
} from "../src/evaluation-data.js";
import {
  validateEvaluationDataset,
  parseEvaluationDataset,
  newEvaluationDataset,
  decodeEvaluationDataset,
  evaluationDatasetReference,
  inspectEvaluationSource,
  evaluationFromDataset,
} from "../src/evaluation-datasets.js";
import { evaluationEditor } from "../src/evaluation-editor.js";
const ref = (resource) =>
  evaluationDatasetReference(
    { ...resource, revision: "9007199254740993" },
    "workspace",
  );
const read = (resource) => ({
  resource: async (namespace, uri, revision) => {
    assert.equal(namespace, "workspace");
    assert.equal(uri, resource.uri);
    assert.equal(revision, "9007199254740993");
    return { resource, revision };
  },
});
const hash = (text) => createHash("sha256").update(text).digest("hex");
const selection = (start, count) => ({ method: "range", start, count });
const cases = (count) =>
  Array.from({ length: count }, (_, index) => ({
    ...newEvaluationCase(),
    label: "Case " + (index + 1),
    prompt: "a",
    check: { type: "exact", expected: "bc", pointer: "" },
  }));

test("Reusable case datasets enforce known formats, UTF-8, UUIDs, text/count/content bounds and distinct trainer annotations", () => {
  const resource = newEvaluationDataset("Quality cases", cases(40));
  const value = parseEvaluationDataset(resource);
  assert.equal(value.cases.length, 40);
  assert.equal(inspectEvaluationSource(resource).kind, "cases");
  assert.equal(
    decodeEvaluationDataset(
      new TextEncoder().encode("\ufeff" + resource.content),
      "cases.json",
    ).cases.length,
    40,
  );
  assert.throws(
    () => decodeEvaluationDataset(new Uint8Array([0xff]), "cases.json"),
    /UTF-8/,
  );
  assert.throws(
    () => decodeEvaluationDataset(new Uint8Array(), "cases.tsv"),
    /json/,
  );
  assert.throws(
    () => validateEvaluationDataset({ ...value, settings: {} }),
    /schema/,
  );
  assert.throws(
    () =>
      validateEvaluationDataset({
        ...value,
        cases: [value.cases[0], value.cases[0]],
      }),
    /unique/,
  );
  assert.throws(() => newEvaluationDataset("Too many", cases(513)), /512/);
  assert.throws(
    () =>
      newEvaluationDataset(
        "Too large",
        cases(512).map((item) => ({ ...item, prompt: "x".repeat(2200) })),
      ),
    /1 MiB/,
  );
  assert.throws(
    () =>
      inspectEvaluationSource({
        ...resource,
        metadata: {
          ...resource.metadata,
          dataset_studio: { version: 1, mode: "cpt" },
          evaluation_dataset: { version: 2 },
        },
      }),
    /unsupported/,
  );
});
test("SFT mapping reads the exact revision, respects physical row grammar and hashes complete source/case bytes", async () => {
  const resource = newDataset(
    "sft",
    "Training examples",
    "a\tbc\r\n\r\nb\tca\n",
  );
  const mapped = await evaluationFromDataset(read(resource), ref(resource), {
    mapping: "exact",
    selection: selection(2, 1),
    settings: { ...defaultEvaluationSettings(), max_tokens: 2 },
  });
  const value = parseEvaluation(mapped);
  assert.equal(value.cases[0].label, "Record 3");
  assert.equal(value.cases[0].prompt, "b");
  assert.equal(value.cases[0].check.expected, "ca");
  const origin = mapped.provenance.evaluation_studio;
  assert.equal(origin.source.revision, "9007199254740993");
  assert.equal(origin.source.content_sha256, hash(resource.content));
  assert.equal(origin.mapped_cases_sha256, hash(JSON.stringify(value.cases)));
  assert.deepEqual(origin.source_records, [{ line: 3 }]);
  assert.equal(origin.mapping, "prompt-completion-exact");
  assert.equal(resource.content, "a\tbc\r\n\r\nb\tca\n");
});
test("Seeded sampling repeats the same source rows without replacement and ranges never silently exceed available records", async () => {
  const resource = newDataset(
    "sft",
    "Sample records",
    Array.from({ length: 60 }, (_, i) => "prompt" + i + "\tanswer" + i).join(
      "\n",
    ),
  );
  const options = {
    target: "dataset",
    mapping: "contains",
    selection: { method: "sample", seed: 42, count: 40 },
  };
  const a = await evaluationFromDataset(read(resource), ref(resource), options),
    b = await evaluationFromDataset(read(resource), ref(resource), options);
  assert.deepEqual(
    a.provenance.evaluation_studio.source_records,
    b.provenance.evaluation_studio.source_records,
  );
  assert.equal(
    new Set(
      a.provenance.evaluation_studio.source_records.map((row) => row.line),
    ).size,
    40,
  );
  assert.deepEqual(
    parseEvaluationDataset(a).cases.map(({ prompt, check }) => ({
      prompt,
      check,
    })),
    parseEvaluationDataset(b).cases.map(({ prompt, check }) => ({
      prompt,
      check,
    })),
  );
  await assert.rejects(
    evaluationFromDataset(read(resource), ref(resource), {
      selection: selection(60, 2),
    }),
    /range/,
  );
  await assert.rejects(
    evaluationFromDataset(read(resource), ref(resource), {
      selection: selection(1, 33),
    }),
    /32/,
  );
  await assert.rejects(
    evaluationFromDataset(read(resource), ref(resource), {
      selection: { method: "sample", count: 2, seed: -1 },
    }),
    /Seed/,
  );
});
test("Corpus/DPO mapping creates only prompt-review cases, and invalid/oversized/unavailable sources reject without truncating", async () => {
  for (const resource of [
    newDataset("cpt", "Corpus", "a\r\n\nb"),
    newDataset("dpo", "Preferences", "a\tpreferred\trejected"),
  ]) {
    const mapped = await evaluationFromDataset(read(resource), ref(resource), {
      selection: selection(1, 1),
    });
    const item = parseEvaluation(mapped).cases[0];
    assert.equal(item.prompt, "a");
    assert.deepEqual(item.check, { type: "manual", expected: "", pointer: "" });
    await assert.rejects(
      evaluationFromDataset(read(resource), ref(resource), {
        mapping: "exact",
        selection: selection(1, 1),
      }),
      /require SFT/,
    );
  }
  const invalid = newDataset("sft", "Invalid", "q\t");
  await assert.rejects(
    evaluationFromDataset(read(invalid), ref(invalid)),
    /format issues/,
  );
  const oversized = newDataset("cpt", "Too long", "a".repeat(16385));
  await assert.rejects(
    evaluationFromDataset(read(oversized), ref(oversized)),
    /prompt/,
  );
  const valid = newDataset("sft", "Available", "a\tbc");
  await assert.rejects(
    evaluationFromDataset(
      { resource: async () => ({ resource: valid, revision: "1" }) },
      ref(valid),
    ),
    /unavailable/,
  );
});
test("Saved case selection preserves IDs, labels and structured checks while recording exact origin indices", async () => {
  const items = cases(42);
  items[20].check = { type: "pointer", expected: "true", pointer: "/ok" };
  const resource = newEvaluationDataset("Reusable", items);
  const mapped = await evaluationFromDataset(read(resource), ref(resource), {
      selection: selection(21, 2),
    }),
    value = parseEvaluation(mapped);
  assert.deepEqual(value.cases, items.slice(20, 22));
  assert.equal(mapped.provenance.evaluation_studio.mapping, "preserve-cases");
  assert.deepEqual(mapped.provenance.evaluation_studio.source_records, [
    { case_index: 20, case_id: items[20].id },
    { case_index: 21, case_id: items[21].id },
  ]);
  assert.equal(
    resource.content,
    JSON.stringify({ schema: 1, kind: "evaluation-dataset", cases: items }),
  );
});
test("Case-editor paging keeps absolute indices and all resource text inert while definitions retain generation controls", () => {
  const items = cases(30);
  items[20].prompt = "</textarea><script>bad</script>";
  const markup = evaluationEditor(
    { kind: "evaluation-dataset", cases: items },
    1,
  );
  assert.equal((markup.match(/class="evaluation-case"/g) || []).length, 10);
  assert.ok(markup.includes('data-case-index="20"'));
  assert.ok(!markup.includes('data-case-index="0"'));
  assert.ok(!markup.includes("<script>"));
  assert.ok(markup.includes("&lt;/textarea&gt;"));
  assert.ok(!markup.includes("data-evaluation-setting"));
  assert.ok(
    evaluationEditor({
      kind: "definition",
      settings: defaultEvaluationSettings(),
      cases: items,
    }).includes("data-evaluation-setting"),
  );
});
