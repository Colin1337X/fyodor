import { test } from "node:test";
import assert from "node:assert/strict";
import {
  evaluateOutput,
  validateDefinition,
  newEvaluationCase,
  defaultEvaluationSettings,
  modelSnapshot,
  runEvaluation,
  validateRun,
  evaluationPage,
  parseEvaluation,
  evaluationResource,
} from "../src/evaluation-data.js";
const reference = {
  namespace: "workspace",
  uri: "fyodor://evaluations/11111111-2222-4333-8444-555555555555",
  revision: "9007199254740993",
};
const check = (type, expected = "", pointer = "") => ({
  type,
  expected,
  pointer,
});
const definition = () => ({
  schema: 1,
  kind: "definition",
  settings: { ...defaultEvaluationSettings(), max_tokens: 2 },
  cases: [
    {
      ...newEvaluationCase(),
      label: "First",
      prompt: "a",
      check: check("exact", "bc"),
    },
    {
      ...newEvaluationCase(),
      label: "Second",
      prompt: "b",
      check: check("manual"),
    },
  ],
});
const models = [
  modelSnapshot(
    {
      id: 1,
      path: "model.gguf",
      generation_supported: true,
      file_size: 1024,
      architecture: "llama",
    },
    "cpu",
  ),
];
const timers = () => {
  let time = 0;
  return { clock: () => (time += 25), wallClock: () => 123 };
};
test("Text and structured checks compare actual values with bounded JSON pointers and no inherited properties", () => {
  assert.equal(evaluateOutput("bc", check("exact", "bc")).status, "pass");
  assert.equal(evaluateOutput(" bc", check("exact", "bc")).status, "fail");
  assert.equal(evaluateOutput("abc", check("contains", "b")).status, "pass");
  assert.throws(() => evaluateOutput("abc", check("contains")), /nonempty/);
  assert.equal(evaluateOutput('{"ok":true}', check("json")).status, "pass");
  assert.equal(evaluateOutput("not JSON", check("json")).status, "fail");
  assert.equal(
    evaluateOutput(
      '{"a/b":[{"y":2,"x":1}]}',
      check("pointer", '{"x":1,"y":2}', "/a~1b/0"),
    ).status,
    "pass",
  );
  assert.equal(
    evaluateOutput("{}", check("pointer", "1", "/constructor")).status,
    "fail",
  );
  assert.equal(
    evaluateOutput("[1]", check("pointer", "1", "/length")).status,
    "fail",
  );
  assert.equal(
    evaluateOutput("anything", check("manual")).status,
    "unreviewed",
  );
  assert.equal(
    evaluateOutput("[".repeat(34) + "0" + "]".repeat(34), check("json")).status,
    "fail",
  );
});
test("Definitions reject unsupported fields, duplicate case IDs, invalid settings and ambiguous check formats", () => {
  const d = definition();
  assert.deepEqual(validateDefinition(d), d);
  assert.throws(() => validateDefinition({ ...d, unexpected: 1 }), /schema/);
  assert.throws(
    () => validateDefinition({ ...d, cases: [d.cases[0], d.cases[0]] }),
    /unique/,
  );
  assert.throws(
    () =>
      validateDefinition({
        ...d,
        settings: { ...d.settings, seed: Number.MAX_SAFE_INTEGER + 1 },
      }),
    /seed/,
  );
  assert.throws(
    () =>
      validateDefinition({
        ...d,
        cases: [
          { ...d.cases[0], check: check("pointer", "not JSON", "/answer") },
        ],
      }),
    /Expected JSON/,
  );
});
test("Runs send exact raw prompts/settings and preserve reported counters versus observed request timing", async () => {
  const requests = [],
    d = definition();
  let partial;
  const run = await runEvaluation(
    {
      generate: async (request) => {
        requests.push(request);
        return {
          model_id: 1,
          text: "bc",
          seed: 42,
          prompt_tokens: 3,
          generated_tokens: 2,
          stop_reason: "length",
        };
      },
    },
    d,
    reference,
    models,
    {
      ...timers(),
      onResult: (r) => {
        assert.equal(r.status, "running");
        partial = r;
      },
    },
  );
  assert.equal(run.status, "completed");
  assert.equal(partial, run);
  assert.equal(run.definition.revision, reference.revision);
  assert.equal(requests[0].prompt, "a");
  assert.deepEqual(Object.keys(requests[0]).sort(), [
    "max_tokens",
    "model_id",
    "prompt",
    "seed",
    "temperature",
    "top_k",
    "top_p",
  ]);
  assert.equal(run.results[0].latency_ms, 25);
  assert.equal(run.results[0].observed_tokens_per_second, 80);
  assert.equal(run.results[0].status, "pass");
  assert.equal(run.results[1].status, "unreviewed");
  assert.deepEqual(d, validateDefinition(d));
  assert.equal(validateRun(run), run);
});
test("Missing metrics remain null; failed generation records errors and stop never fabricates remaining outputs", async () => {
  let calls = 0,
    stop = false;
  const run = await runEvaluation(
    {
      generate: async () => {
        calls++;
        return { model_id: 1, text: "bc", seed: 42 };
      },
    },
    definition(),
    reference,
    models,
    { ...timers(), onResult: () => (stop = true), shouldStop: () => stop },
  );
  assert.equal(calls, 1);
  assert.equal(run.status, "stopped");
  assert.equal(run.stop_reason, "requested");
  assert.equal(run.results.length, 1);
  assert.equal(run.planned, 2);
  assert.equal(run.results[0].prompt_tokens, null);
  assert.equal(run.results[0].observed_tokens_per_second, null);
  const errors = await runEvaluation(
    {
      generate: async () => {
        throw Error("Native context overflow");
      },
    },
    definition(),
    reference,
    models,
    timers(),
  );
  assert.equal(errors.results[0].status, "error");
  assert.equal(errors.results[0].generated_tokens, null);
  assert.equal(errors.results[0].text, "");
});
test("Output prefixes are explicit and complete-code-point bounded; model comparisons share a saved definition", async () => {
  const d = definition();
  d.cases = [{ ...d.cases[0], check: check("contains", "👋") }];
  const run = await runEvaluation(
    {
      generate: async (request) => ({
        model_id: request.model_id,
        text: "👋".repeat(17000),
        seed: 42,
        generated_tokens: 17000,
      }),
    },
    d,
    reference,
    [...models, { ...models[0], process_id: 2, path: "other.gguf" }],
    timers(),
  );
  assert.equal(run.results.length, 2);
  assert.equal(run.results[0].output_truncated, true);
  assert.equal(run.results[0].output_bytes, 68000);
  assert.equal(new TextEncoder().encode(run.results[0].text).length, 65536);
  assert.equal(run.results[0].status, "pass");
  const resource = evaluationResource("run", "Results", run);
  assert.equal(parseEvaluation(resource).results.length, 2);
  const corrupt = structuredClone(run);
  corrupt.results[0].latency_ms = -1;
  assert.throws(() => validateRun(corrupt), /measurement/);
});
test("Evaluation catalog paging excludes other resources, preserves sparse cursors and detects repeats", async () => {
  let count = 0;
  assert.deepEqual(
    await evaluationPage(
      {
        resources: async () =>
          ++count < 3
            ? { resources: [], next: String(count) }
            : {
                resources: [
                  { uri: reference.uri },
                  {
                    uri: "fyodor://datasets/11111111-2222-4333-8444-555555555555",
                  },
                ],
                next: "",
              },
      },
      "workspace",
    ),
    { resources: [{ uri: reference.uri }], next: "" },
  );
  await assert.rejects(
    evaluationPage(
      { resources: async () => ({ resources: [], next: "repeat" }) },
      "workspace",
    ),
    /Repeated/,
  );
});

test("Run size limits stop scheduling and malformed imported observations fail closed", async () => {
  const d = definition();
  d.cases = Array.from({ length: 32 }, () => ({
    ...newEvaluationCase(),
    prompt: "a",
    check: check("manual"),
  }));
  let calls = 0;
  const run = await runEvaluation(
    {
      generate: async () => {
        calls++;
        return { model_id: 1, text: "x".repeat(65536), seed: 42 };
      },
    },
    d,
    reference,
    models,
    timers(),
  );
  assert.equal(run.status, "stopped");
  assert.equal(run.stop_reason, "result-size");
  assert.ok(calls < 32);
  assert.equal(calls, run.results.length + 1);
  assert.ok(new TextEncoder().encode(JSON.stringify(run)).length < 1024 * 1024);
  const malformed = structuredClone(run);
  malformed.results[0].output_bytes = 1;
  assert.throws(() => validateRun(malformed), /inconsistent/);
  const badNamespace = structuredClone(run);
  badNamespace.definition.namespace = "../other";
  assert.throws(() => validateRun(badNamespace), /schema/);
  const duplicate = structuredClone(run);
  duplicate.results.push(duplicate.results[0]);
  assert.throws(() => validateRun(duplicate), /measurement/);
  await assert.rejects(
    runEvaluation(
      {
        generate: async () => {
          throw Error("Must not call");
        },
      },
      d,
      reference,
      [...models, ...models],
      timers(),
    ),
    /snapshot/,
  );
});
