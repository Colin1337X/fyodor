import test from "node:test";
import assert from "node:assert/strict";
import {
  newWorkflow,
  newWorkflowStep,
  validateWorkflow,
  validateWorkflowRun,
  workflowResource,
  parseWorkflow,
  workflowUri,
  workflowPage,
} from "../src/workflow-data.js";
import { runWorkflow } from "../src/workflow-runner.js";
import { newAgent, agentResource } from "../src/agent-data.js";
const uri = "fyodor://workflows/12345678-1234-1234-1234-123456789012";
const reference = { namespace: "workspace", uri, revision: "9007199254740993" };
const provider = {
  type: "openai",
  endpoint: "https://example.test/v1",
  model: "test",
  apiKey: "secret-fixture",
};
const profile = agentResource("definition", "Agent", newAgent(provider));
const nativeApi = () => ({
  models: async () => ({
    version: 1,
    ok: true,
    models: [
      { id: 1, generation_supported: true, path: "test.gguf", format: "gguf" },
    ],
  }),
  runtime: async () => ({
    version: 1,
    ok: true,
    models: [{ model_id: 1, compute: "cpu" }],
  }),
  generate: async (payload) => ({
    model_id: payload.model_id,
    seed: payload.seed,
    text: "bc",
    prompt_tokens: 1,
    generated_tokens: 2,
  }),
  resource: async () => ({ resource: profile, revision: "1" }),
  openaiChat: async () => ({
    choices: [{ message: { role: "assistant", content: "Agent result" } }],
  }),
});
const execute = (definition, extra = {}) =>
  runWorkflow({
    definition,
    reference,
    input: "a",
    provider,
    modelId: 1,
    api: nativeApi(),
    ...extra,
  });
const pipeline = () => {
  const gen = newWorkflowStep("generate"),
    check = newWorkflowStep("check", gen.id),
    agent = newWorkflowStep("agent", check.id);
  gen.config.max_tokens = 2;
  check.config.expected = "bc";
  agent.config = { namespace: "workspace", uri: profile.uri, revision: "1" };
  return { schema: 1, kind: "workflow-definition", steps: [gen, check, agent] };
};
test("workflow resources and stable step bindings reject cycles, dangling IDs and executable extensions", () => {
  assert.equal(workflowUri(uri), true);
  assert.equal(
    workflowUri(uri + "/runs/12345678-1234-1234-1234-123456789012"),
    false,
  );
  assert.equal(
    workflowUri("fyodor://workflows/00000000-0000-0000-0000-000000000000"),
    false,
  );
  const d = pipeline();
  assert.deepEqual(validateWorkflow(d), d);
  for (const edit of [
    (d) => (d.steps[0].input.source = d.steps[1].id),
    (d) => (d.steps[1].id = d.steps[0].id),
    (d) => (d.steps[1].operation = "shell"),
    (d) => (d.steps[1].config.type = "manual"),
    (d) => (d.steps[0].input.prefix = "😀".repeat(513)),
    (d) => (d.steps[0].config.max_tokens = 0),
    (d) => (d.permissions = { filesystem: true }),
  ]) {
    const v = pipeline();
    edit(v);
    assert.throws(() => validateWorkflow(v));
  }
  const packageDef = workflowResource("definition", "Workflow", d);
  assert.deepEqual(parseWorkflow(packageDef), d);
  assert.throws(() => parseWorkflow({ ...packageDef, metadata: {} }));
});
test("workflow composition runs actual generation, literal check and saved agent with exact bindings", async () => {
  const definition = pipeline(),
    sent = [],
    updates = [],
    api = nativeApi(),
    mutableProvider = { ...provider };
  api.generate = async (payload) => {
    sent.push(payload);
    mutableProvider.endpoint = "https://changed.test";
    mutableProvider.apiKey = "changed-key";
    return {
      model_id: 1,
      seed: payload.seed,
      text: "bc",
      prompt_tokens: 0,
      generated_tokens: 2,
    };
  };
  api.openaiChat = async (connection, body) => {
    assert.equal(connection.apiKey, provider.apiKey);
    assert.equal(connection.endpoint, provider.endpoint);
    assert.equal(body.messages[1].content, "bc");
    return {
      choices: [
        { message: { role: "assistant", content: "<script>inert</script>" } },
      ],
    };
  };
  const run = await execute(definition, {
    api,
    provider: mutableProvider,
    onUpdate: (value) => updates.push(value),
  });
  assert.equal(run.status, "completed");
  assert.equal(mutableProvider.endpoint, "https://changed.test");
  assert.equal(run.output, "<script>inert</script>");
  assert.deepEqual(
    run.steps.map((item) => item.input),
    ["a", "bc", "bc"],
  );
  assert.equal(sent[0].max_tokens, 2);
  assert.equal(run.steps[0].prompt_tokens, 0);
  assert.equal(run.steps[0].generated_tokens, 2);
  assert.equal(run.steps[2].agent_run.steps[0].tokens, null);
  assert.equal(updates[1].output, "bc");
  assert.equal(run.workflow.revision, reference.revision);
  assert.equal(JSON.stringify(run).includes("secret-fixture"), false);
  assert.deepEqual(validateWorkflowRun(run), run);
  const record = workflowResource("run", "Result", run);
  assert.equal(record.metadata.workflow_studio.workflow, uri);
  assert.deepEqual(parseWorkflow(record), run);
  assert.throws(() =>
    parseWorkflow({
      ...record,
      metadata: {
        workflow_studio: { version: 1, kind: "run", workflow: record.uri },
      },
    }),
  );
});
test("workflow preflight blocks stale, mismatched or cross-namespace agents before any generation", async () => {
  for (const option of ["revision", "identity", "namespace", "endpoint"]) {
    const definition = pipeline(),
      api = nativeApi();
    let requests = 0;
    api.generate = async () => {
      requests++;
      throw Error("Must not run");
    };
    if (option === "revision")
      api.resource = async () => ({ resource: profile, revision: "2" });
    if (option === "identity")
      api.resource = async () => ({
        resource: {
          ...profile,
          uri: profile.uri.replace(
            "fyodor://agents/",
            "fyodor://agents/00000000-",
          ),
        },
        revision: "1",
      });
    if (option === "namespace") definition.steps[2].config.namespace = "other";
    const selected =
      option === "endpoint"
        ? { ...provider, endpoint: "https://other.test" }
        : provider;
    await assert.rejects(execute(definition, { api, provider: selected }));
    assert.equal(requests, 0);
  }
});
test("checks stop the pipeline, malformed native observations fail and metrics stay unknown", async () => {
  const definition = pipeline();
  definition.steps[1].config.expected = "wrong";
  const api = nativeApi();
  let calls = 0;
  api.openaiChat = async () => {
    calls++;
    throw Error("Must not run");
  };
  const failed = await execute(definition, { api });
  assert.equal(failed.status, "failed");
  assert.equal(failed.steps.length, 2);
  assert.equal(failed.output, "bc");
  assert.equal(calls, 0);
  const single = { ...pipeline(), steps: [newWorkflowStep("generate")] };
  api.generate = async () => ({ model_id: 2, seed: 42, text: "wrong" });
  const wrong = await execute(single, { api });
  assert.equal(wrong.status, "failed");
  assert.equal(wrong.steps[0].output, "");
  api.generate = async (payload) => ({
    model_id: 1,
    seed: payload.seed,
    text: "real",
  });
  const missing = await execute(single, { api });
  assert.equal(missing.steps[0].generated_tokens, null);
  assert.equal(missing.steps[0].prompt_tokens, null);
  assert.throws(() => validateWorkflowRun({ ...missing, output: "forged" }));
});
test("workflow Stop retains an in-flight native result and prevents every downstream request", async () => {
  const signal = new AbortController(),
    api = nativeApi();
  let agents = 0;
  api.generate = async (payload) => {
    signal.abort();
    return {
      model_id: 1,
      seed: payload.seed,
      text: "actual",
      generated_tokens: 1,
    };
  };
  api.openaiChat = async () => {
    agents++;
    throw Error("Must not run");
  };
  const run = await execute(pipeline(), { api, signal: signal.signal });
  assert.equal(run.status, "stopped");
  assert.equal(run.steps.length, 1);
  assert.equal(run.steps[0].status, "completed");
  assert.equal(run.output, "actual");
  assert.equal(agents, 0);
  assert.deepEqual(validateWorkflowRun(run), run);
});
test("workflow rechecks the agent at its step and preserves aborted nested trace", async () => {
  const api = nativeApi();
  let reads = 0;
  api.resource = async () => ({
    resource: profile,
    revision: ++reads === 1 ? "1" : "2",
  });
  const stale = await execute(pipeline(), { api });
  assert.equal(stale.status, "failed");
  assert.equal(stale.steps[2].agent_run, null);
  assert.match(stale.error, /changed before its step/);
  const controller = new AbortController();
  const abortApi = nativeApi();
  abortApi.openaiChat = async (_connection, _body, { signal }) => {
    assert.equal(signal, controller.signal);
    controller.abort();
    throw new DOMException("Stopped", "AbortError");
  };
  const stopped = await execute(pipeline(), {
    api: abortApi,
    signal: controller.signal,
  });
  assert.equal(stopped.status, "stopped");
  assert.equal(stopped.steps[2].agent_run.status, "stopped");
  assert.equal(stopped.output, "bc");
  assert.deepEqual(validateWorkflowRun(stopped), stopped);
});
test("composed text and trace limits fail explicitly without truncating saved results", async () => {
  const d = newWorkflow();
  d.steps[0].input.prefix = "prefix:";
  const literal = await execute(d, { input: "${process.exit()}" });
  assert.equal(literal.output, "prefix:${process.exit()}");
  assert.equal(literal.model, null);
  assert.equal(literal.steps[0].request_ms, null);
  const many = {
      schema: 1,
      kind: "workflow-definition",
      steps: Array.from({ length: 12 }, () => newWorkflowStep("generate")),
    },
    api = nativeApi();
  api.generate = async (payload) => ({
    model_id: 1,
    seed: payload.seed,
    text: "a".repeat(65536),
  });
  const bounded = await execute(many, { api });
  assert.equal(bounded.status, "limit");
  assert.ok(bounded.steps.length < 12);
  assert.deepEqual(validateWorkflowRun(bounded), bounded);
  await assert.rejects(execute(d, { input: "😀".repeat(2049) }));
  assert.throws(() =>
    validateWorkflowRun({
      ...literal,
      steps: [{ ...literal.steps[0], generated_tokens: 1 }],
    }),
  );
});
test("workflow catalogs filter run annotations, preserve complete pages and detect repeated cursors", async () => {
  const run = await execute(newWorkflow()),
    record = workflowResource("run", "Result", run);
  let calls = 0;
  const page = await workflowPage(
    {
      resource: async (namespace, requested, revision) => {
        assert.equal(namespace, "workspace");
        assert.equal(requested, record.uri);
        assert.equal(revision, "1");
        return { resource: record, revision: "1" };
      },
      resources: async () => ({
        resources:
          ++calls === 1
            ? [profile]
            : Array.from({ length: 25 }, () => ({
                uri: record.uri,
                title: record.title,
                revision: "1",
              })),
        next: calls === 1 ? "second" : "third",
      }),
    },
    "workspace",
    "run",
    uri,
  );
  assert.equal(page.resources.length, 25);
  assert.equal(page.next, "third");
  let reads = 0,
    pages = 0;
  const other = {
    ...record,
    metadata: {
      workflow_studio: {
        version: 1,
        kind: "run",
        workflow: uri.replace("12345678", "87654321"),
      },
    },
  };
  const bounded = await workflowPage(
    {
      resources: async () => ({
        resources: Array.from({ length: ++pages === 1 ? 99 : 100 }, () => ({
          uri: record.uri,
          revision: "1",
        })),
        next: String(pages),
      }),
      resource: async () => {
        reads++;
        return { resource: other, revision: "1" };
      },
    },
    "workspace",
    "run",
    uri,
  );
  assert.equal(reads, 199);
  assert.equal(pages, 2);
  assert.equal(bounded.next, "2");
  assert.equal(bounded.resources.length, 0);
  await assert.rejects(
    workflowPage(
      { resources: async () => ({ resources: [], next: "repeat" }) },
      "workspace",
    ),
    /Repeated/,
  );
});
