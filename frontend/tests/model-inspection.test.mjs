import test from "node:test";
import assert from "node:assert/strict";
import {
  modelObservation,
  modelObservationMarkup,
  nativeModelInfo,
  inspectedModelId,
  tokenizerText,
  tokenizerResult,
  tokenizerResultLimit,
} from "../src/model-inspection-data.js";

test("Model observations retain reported zero counters and distinguish unknown registry/compiler values from capabilities", () => {
  const model = modelObservation(
    {
      id: 3,
      path: "C:/<img>/model.gguf",
      file_size: 0,
      architecture: null,
      capabilities: ["inspect", "<script>"],
      generation_supported: true,
    },
    {
      compiled: { cpu: true, cuda: false },
      models: [{ model_id: 3, compute: "cpu", selectable: true }],
    },
  );
  assert.equal(model.file_size, 0);
  assert.equal(model.tensor_count, null);
  assert.equal(model.compiled.rocm, null);
  assert.equal(model.compiled.cuda, false);
  assert.equal(model.selectable, true);
  assert.equal(modelObservation({ id: 3 }, null).compute, null);
  for (const id of [0, -1, 1.5, "3", Number.MAX_SAFE_INTEGER + 1])
    assert.throws(() => inspectedModelId(id), /valid runtime ID/);
});
test("Native shape information binds to the requested model and rejects unsupported response values without guessing missing metadata", () => {
  const good = {
    version: 1,
    ok: true,
    model_id: 7,
    architecture: "llama",
    compute: "cpu",
    context_length: 32,
    experts: 0,
    mtp_assistant: false,
  };
  const info = nativeModelInfo(good, 7);
  assert.equal(info.context_length, 32);
  assert.equal(info.experts, 0);
  assert.equal(info.sliding_window, null);
  assert.equal(info.mtp_assistant, false);
  for (const patch of [
    { version: 2 },
    { ok: false },
    { model_id: 8 },
    { context_length: -1 },
    { layers: 1.5 },
    { vocabulary_size: Number.MAX_SAFE_INTEGER + 1 },
    { mtp_assistant: "false" },
    { compute: "imaginary" },
    { architecture: {} },
  ])
    assert.throws(() => nativeModelInfo({ ...good, ...patch }, 7));
});
test("Tokenizer requests preserve raw Unicode/whitespace and enforce UTF-8 byte and native string bounds", () => {
  assert.equal(tokenizerText(" 한글\n😀 "), " 한글\n😀 ");
  assert.equal(tokenizerText(""), "");
  assert.equal(tokenizerText("a".repeat(65536)).length, 65536);
  for (const text of ["\0", "😀".repeat(16385), "a".repeat(65537), null])
    assert.throws(() => tokenizerText(text), /64 KiB/);
});
test("Tokenizer results require exact model/count/uint32 IDs and bound the full downloadable token list", () => {
  const good = {
    version: 1,
    ok: true,
    model_id: 2,
    count: 3,
    tokens: [0, 17, 4294967295],
  };
  const parsed = tokenizerResult(good, 2);
  assert.deepEqual(parsed, { count: 3, tokens: [0, 17, 4294967295] });
  parsed.tokens[0] = 12;
  assert.equal(good.tokens[0], 0);
  for (const patch of [
    { model_id: 3 },
    { count: 2 },
    { count: tokenizerResultLimit + 1 },
    { tokens: [-1, 0, 1] },
    { tokens: [0, 1, 4294967296] },
    { tokens: [0, 1, 1.5] },
    { version: 2 },
    { tokens: null },
  ])
    assert.throws(() => tokenizerResult({ ...good, ...patch }, 2));
});
test("Inspector markup keeps metadata/errors inert and reports unknown fields without filename-derived quantization or identity", () => {
  const model = modelObservation(
      {
        id: 1,
        path: "<img>.gguf",
        producer: "<script>",
        capabilities: ["inspect"],
        generation_error: "<svg>",
      },
      {},
    ),
    html = modelObservationMarkup(model);
  assert.match(html, /&lt;img&gt;\.gguf/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /&lt;svg&gt;/);
  assert.doesNotMatch(html, /<img>|<script>|<svg>/);
  assert.match(html, /Not reported/);
  assert.match(html, /Runtime model ID/);
  assert.doesNotMatch(html, /Quantization|Stable model identity/);
});
