// Authored protocol fixture, not evidence of model reasoning quality. It uses
// actual HTTP and checks that the browser passes a real native tool observation.
import { createServer } from "node:http";
import assert from "node:assert/strict";
export async function agentEndpointFixture() {
  const observations = [],
    errors = [],
    timers = new Set();
  let delayed = 0;
  const server = createServer(async (request, response) => {
    response.setHeader("Access-Control-Allow-Origin", "http://localhost:5179");
    response.setHeader(
      "Access-Control-Allow-Headers",
      "Authorization, Content-Type",
    );
    response.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    if (request.method === "OPTIONS") {
      response.writeHead(204);
      response.end();
      return;
    }
    if (request.url === "/qa/status" && request.method === "GET") {
      response.end(JSON.stringify({ delayed }));
      return;
    }
    try {
      assert.equal(request.url, "/v1/chat/completions");
      assert.equal(request.method, "POST");
      assert.equal(request.headers.authorization, "Bearer agent-fixture-key");
      let text = "";
      for await (const chunk of request) {
        text += chunk;
        assert.ok(text.length < 1048576);
      }
      const body = JSON.parse(text);
      assert.equal(body.model, "fixture-agent-model");
      assert.equal(body.stream, false);
      assert.deepEqual(
        body.tools.map((item) => item.function.name),
        ["runtime_status"],
      );
      const input = body.messages.find((item) => item.role === "user").content;
      if (input === "Stop this request") {
        delayed++;
        const timer = setTimeout(() => {
          timers.delete(timer);
          if (!response.destroyed)
            response.end(
              JSON.stringify({
                choices: [
                  { message: { role: "assistant", content: "Late response" } },
                ],
              }),
            );
        }, 30000);
        timers.add(timer);
        response.on("close", () => {
          clearTimeout(timer);
          timers.delete(timer);
        });
        return;
      }
      const tool = body.messages.at(-1);
      let result;
      if (tool.role === "tool") {
        assert.equal(tool.tool_call_id, "runtime-read");
        const data = JSON.parse(tool.content);
        assert.equal(data.compiled.cpu, true);
        assert.equal(data.models.length, 2);
        assert.ok(
          data.models.every(
            (item) =>
              item.compute === "cpu" && Number.isSafeInteger(item.model_id),
          ),
        );
        assert.deepEqual(Object.keys(data).sort(), ["compiled", "models"]);
        observations.push(data);
        result = {
          choices: [
            {
              message: {
                role: "assistant",
                content: "<script>Runtime reviewed</script>",
              },
            },
          ],
          usage: { completion_tokens: 5 },
        };
      } else
        result = {
          choices: [
            {
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    type: "function",
                    id: "runtime-read",
                    function: { name: "runtime_status", arguments: "{}" },
                  },
                ],
              },
            },
          ],
          usage: { completion_tokens: 2 },
        };
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify(result));
    } catch (error) {
      errors.push(error.message);
      response.writeHead(500);
      response.end(
        JSON.stringify({
          error: { message: "Agent protocol fixture assertion failed" },
        }),
      );
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    endpoint: `http://127.0.0.1:${server.address().port}/v1`,
    observations,
    errors,
    get delayed() {
      return delayed;
    },
    close: async () => {
      timers.forEach(clearTimeout);
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
