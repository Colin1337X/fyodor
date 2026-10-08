import test from "node:test";
import assert from "node:assert/strict";
import {
  newAgent,
  agentKind,
  agentEndpoint,
  agentConnection,
  agentResource,
  parseAgent,
  validateAgent,
  validateAgentRun,
  agentPage,
} from "../src/agent-data.js";
import { runAgent, agentReadTool } from "../src/agent-runner.js";
const provider = {
  type: "openai",
  endpoint: "https://example.test/v1",
  model: "test-model",
  apiKey: "secret-fixture",
};
const uri = "fyodor://agents/12345678-1234-1234-1234-123456789012";
const reference = { namespace: "workspace", uri, revision: "9007199254740993" };
const definition = () => ({ ...newAgent(provider), tools: ["runtime_status"] });
const runtime = () => ({
  version: 1,
  ok: true,
  compiled: { cpu: true, cuda: false, vulkan: false, rocm: false, mlx: false },
  models: [{ model_id: 1, compute: "cpu", private: "secret" }],
  token: "secret",
});
const call = (name = "runtime_status", args = "{}", id = "call-1") => ({
  type: "function",
  id,
  function: { name, arguments: args },
});
const reply = (content, calls = [], usage = undefined) => ({
  choices: [{ message: { role: "assistant", content, tool_calls: calls } }],
  usage,
});
const execute = (api, extra = {}) =>
  runAgent({
    definition: definition(),
    reference,
    input: "Review runtime.",
    provider,
    api,
    ...extra,
  });

test("agent identities, definitions and endpoint credential binding are strict", () => {
  assert.equal(agentKind(uri), "definition");
  assert.equal(
    agentKind(uri + "/runs/87654321-1234-1234-1234-123456789012"),
    "run",
  );
  for (const bad of [
    uri + "/",
    uri.toUpperCase(),
    "fyodor://agents/00000000-0000-0000-0000-000000000000",
    uri + "/runs/00000000-0000-0000-0000-000000000000",
  ])
    assert.equal(agentKind(bad), null);
  assert.equal(agentEndpoint("HTTPS://EXAMPLE.TEST/v1/"), provider.endpoint);
  for (const bad of [
    "file:///tmp",
    "https://user:pass@example.test",
    "https://example.test?secret=x",
    "https://example.test/#x",
  ])
    assert.throws(() => agentEndpoint(bad));
  assert.deepEqual(newAgent(provider).tools, []);
  assert.equal(
    agentConnection(definition(), provider).apiKey,
    "secret-fixture",
  );
  assert.throws(
    () =>
      agentConnection(definition(), {
        ...provider,
        endpoint: "https://other.test/v1",
      }),
    /matching/,
  );
  for (const edit of [
    (d) => d.tools.push("constructor"),
    (d) => d.tools.push("runtime_status"),
    (d) => (d.permissions.context = true),
    (d) => (d.limits.turns = 13),
    (d) => (d.sampling.max_tokens = NaN),
    (d) => (d.instructions = "😀".repeat(4097)),
    (d) => (d.apiKey = "secret"),
    (d) => delete d.permissions,
  ]) {
    const d = definition();
    edit(d);
    assert.throws(() => validateAgent(d));
  }
  const resource = agentResource(
    "definition",
    "<script>inert</script>",
    definition(),
  );
  assert.deepEqual(parseAgent(resource), definition());
  assert.equal(JSON.stringify(resource).includes("secret-fixture"), false);
  assert.throws(() => parseAgent({ ...resource, metadata: {} }), /Unsupported/);
  assert.throws(() => agentResource("unknown", "x", definition()));
});

test("agent read-only tools project real observations without paths or credentials", async () => {
  assert.deepEqual(
    JSON.parse(
      await agentReadTool({ runtime: async () => runtime() }, "runtime_status"),
    ),
    { compiled: runtime().compiled, models: [{ model_id: 1, compute: "cpu" }] },
  );
  const models = JSON.parse(
    await agentReadTool(
      {
        models: async () => ({
          version: 1,
          ok: true,
          models: [
            {
              id: 2,
              path: "C:\\private\\test.gguf",
              architecture: "llama",
              key: "secret",
            },
          ],
        }),
      },
      "list_loaded_models",
    ),
  );
  assert.deepEqual(models, [
    { id: 2, name: "test.gguf", architecture: "llama" },
  ]);
  assert.ok(Number.isFinite(Date.parse(await agentReadTool({}, "local_time"))));
  await assert.rejects(agentReadTool({}, "__proto__"), /not permitted/);
  for (const edit of [
    (r) => delete r.compiled.cpu,
    (r) => (r.models[0].model_id = {}),
    (r) => (r.models[0].compute = "unknown"),
    (r) => (r.version = 2),
  ]) {
    const r = runtime();
    edit(r);
    await assert.rejects(
      agentReadTool({ runtime: async () => r }, "runtime_status"),
    );
  }
});

test("agent loop forwards only granted tools, actual tool output and nullable endpoint usage", async () => {
  const requests = [],
    updates = [];
  const run = await execute(
    {
      runtime: async () => runtime(),
      openaiChat: async (connection, body) => {
        assert.equal(connection.apiKey, provider.apiKey);
        requests.push(body);
        return requests.length === 1
          ? reply("Reading runtime.", [call()], { completion_tokens: 2 })
          : reply("<script>inert</script>");
      },
    },
    { onUpdate: (value) => updates.push(value) },
  );
  assert.equal(run.status, "completed");
  assert.equal(run.output, "<script>inert</script>");
  assert.deepEqual(
    run.steps.map((s) => s.kind),
    ["model", "tool", "model"],
  );
  assert.deepEqual(
    run.steps.map((s) => s.tokens),
    [2, null, null],
  );
  assert.equal(updates[0].output, "Reading runtime.");
  assert.deepEqual(
    requests[0].tools.map((t) => t.function.name),
    ["runtime_status"],
  );
  assert.equal(requests[1].messages.at(-1).content, run.steps[1].text);
  assert.equal(requests[1].messages.at(-1).tool_call_id, "call-1");
  assert.ok(run.steps[0].request_ms >= 0);
  assert.equal(run.agent.revision, reference.revision);
  assert.equal(JSON.stringify(run).includes("secret-fixture"), false);
  assert.deepEqual(validateAgentRun(run), run);
  const packageRun = agentResource("run", "Run", run, uri);
  assert.deepEqual(parseAgent(packageRun), run);
  assert.throws(
    () => agentResource("run", "Run", run, uri.replace("12345678", "87654321")),
    /parent/,
  );
  assert.throws(() =>
    validateAgentRun({
      ...run,
      agent: { ...reference, revision: "9223372036854775808" },
    }),
  );
});

test("denied, malformed or over-budget tool batches execute no local read", async () => {
  const batches = [
    [call(), call("read_resource", "{}", "denied")],
    [call("runtime_status", '{"path":"x"}')],
    [call(), call()],
    [call("runtime_status", "[]")],
  ];
  for (const calls of batches) {
    let reads = 0;
    const run = await execute({
      runtime: async () => {
        reads++;
        return runtime();
      },
      openaiChat: async () => reply(null, calls),
    });
    assert.equal(run.status, "failed");
    assert.equal(reads, 0);
  }
  let reads = 0;
  const d = definition();
  d.limits.tool_calls = 0;
  const run = await execute(
    {
      runtime: async () => {
        reads++;
        return runtime();
      },
      openaiChat: async () => reply(null, [call()]),
    },
    { definition: d },
  );
  assert.equal(run.status, "limit");
  assert.equal(reads, 0);
});

test("model-turn limits, malformed responses and tool errors remain explicit", async () => {
  const d = definition();
  d.limits.turns = 1;
  const limited = await execute(
    {
      runtime: async () => runtime(),
      openaiChat: async () => reply(null, [call()]),
    },
    { definition: d },
  );
  assert.equal(limited.status, "limit");
  assert.equal(limited.steps.length, 2);
  const bad = await execute({
    openaiChat: async () => reply({ unsupported: true }),
  });
  assert.equal(bad.status, "failed");
  assert.equal(bad.output, "");
  const failed = await execute({
    runtime: async () => {
      throw Error("Offline");
    },
    openaiChat: async () => reply(null, [call()]),
  });
  assert.equal(failed.status, "failed");
  assert.equal(failed.steps[1].error, "Offline");
  assert.equal(failed.steps[1].tokens, null);
  const oversized = await execute({
    openaiChat: async () => reply("a".repeat(65537)),
  });
  assert.equal(oversized.status, "failed");
  assert.equal(oversized.output, "");
  const many = definition();
  many.tools = ["local_time"];
  many.limits.turns = 12;
  const bounded = await execute(
    { openaiChat: async () => reply("a".repeat(65536), [call("local_time")]) },
    { definition: many },
  );
  assert.equal(bounded.status, "limit");
  assert.match(bounded.error, /recorded-output limit/);
  assert.ok(new TextEncoder().encode(JSON.stringify(bounded)).length < 1048576);
  assert.deepEqual(validateAgentRun(bounded), bounded);
  const oddError = await execute({
    openaiChat: async () => {
      throw Error("Error\0detail");
    },
  });
  assert.equal(oddError.error.includes("\0"), false);
  assert.deepEqual(validateAgentRun(oddError), oddError);
});

test("Stop aborts the endpoint request and preserves completed reads without another turn", async () => {
  const controller = new AbortController();
  const stopped = await execute(
    {
      openaiChat: async (_connection, _body, { signal }) => {
        assert.equal(signal, controller.signal);
        controller.abort();
        throw new DOMException("Stopped", "AbortError");
      },
    },
    { signal: controller.signal },
  );
  assert.equal(stopped.status, "stopped");
  assert.equal(stopped.steps[0].status, "stopped");
  assert.equal(stopped.steps[0].tokens, null);
  const duringTool = new AbortController();
  let requests = 0;
  const observed = await execute(
    {
      openaiChat: async () => {
        requests++;
        return reply(null, [call()]);
      },
      runtime: async () => {
        duringTool.abort();
        return runtime();
      },
    },
    { signal: duringTool.signal },
  );
  assert.equal(requests, 1);
  assert.equal(observed.status, "stopped");
  assert.equal(observed.steps[1].status, "completed");
});

test("invalid inputs and revision references fail before any endpoint request", async () => {
  let requests = 0;
  const api = {
    openaiChat: async () => {
      requests++;
      return reply("done");
    },
  };
  for (const extra of [
    { input: "😀".repeat(2049) },
    { input: "x\0y" },
    { reference: { ...reference, revision: 1 } },
    { provider: { ...provider, endpoint: "https://other.test" } },
  ])
    await assert.rejects(execute(api, extra));
  assert.equal(requests, 0);
  const noTools = await execute(
    {
      openaiChat: async (_connection, body) => {
        assert.equal(Object.hasOwn(body, "tools"), false);
        return reply("done");
      },
    },
    { definition: newAgent(provider) },
  );
  assert.equal(noTools.status, "completed");
});

test("agent paging consumes sparse catalogs, preserves final-page matches and detects cycles", async () => {
  const d = agentResource("definition", "Agent", definition());
  let calls = 0;
  const page = await agentPage(
    {
      resources: async () => {
        calls++;
        return calls < 3
          ? { resources: [{ uri: "fyodor://models/1" }], next: String(calls) }
          : { resources: Array.from({ length: 25 }, () => d), next: "3" };
      },
    },
    "workspace",
  );
  assert.equal(calls, 3);
  assert.equal(page.resources.length, 25);
  assert.equal(page.next, "3");
  await assert.rejects(
    agentPage(
      { resources: async () => ({ resources: [], next: "same" }) },
      "workspace",
    ),
    /Repeated/,
  );
  calls = 0;
  const sparse = await agentPage(
    { resources: async () => ({ resources: [], next: String(++calls) }) },
    "workspace",
  );
  assert.equal(calls, 25);
  assert.equal(sparse.next, "25");
  await assert.rejects(agentPage({}, "invalid namespace"));
});
