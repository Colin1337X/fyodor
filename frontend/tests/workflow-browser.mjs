import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { configureAgentFixture } from "./endpoint-browser-helpers.mjs";
export async function workflowBrowser({
  evaluate,
  waitFor,
  reload,
  call,
  sessionId,
  output,
  agentUri,
}) {
  const click = async (name, index = null) => {
    const selector =
      index === null
        ? `[data-workflow="${name}"]`
        : `[data-workflow-step]:nth-child(${index + 1}) [data-workflow="${name}"]`;
    await waitFor(
      `!!document.querySelector(${JSON.stringify(selector + ":not(:disabled)")})`,
    );
    await evaluate(
      `document.querySelector(${JSON.stringify(selector)}).click()`,
    );
  };
  const field = async (selector, value, change = false) =>
    evaluate(
      `(()=>{const n=document.querySelector(${JSON.stringify(selector)});n.value=${JSON.stringify(value)};n.dispatchEvent(new Event(${JSON.stringify(change ? "change" : "input")},{bubbles:true}));})()`,
    );
  const stepField = (index, fieldName) =>
    `[data-workflow-step]:nth-child(${index + 1}) [data-workflow-field="${fieldName}"]`;
  const setting = (index, key) =>
    `[data-workflow-step]:nth-child(${index + 1}) [data-workflow-setting="${key}"]`;
  const add = async (operation) => {
    await field("#workflow-new-operation", operation, true);
    await click("add");
    await waitFor('!document.querySelector("[data-workflow=save]").disabled');
  };
  await evaluate('document.querySelector("[data-view=workflows]").click()');
  await click("new");
  await waitFor('!!document.querySelector("#workflow-title:not(:disabled)")');
  await field("#workflow-title", "Browser pipeline");
  await field(stepField(0, "operation"), "generate", true);
  await waitFor(
    '!!document.querySelector("[data-workflow-setting=max_tokens]:not(:disabled)")',
  );
  await field(setting(0, "max_tokens"), "2");
  await field(stepField(0, "label"), "Generate a line");
  // Real native keyboard insertion must retain the caret without a repaint.
  await evaluate(
    `document.querySelector(${JSON.stringify(stepField(0, "label"))}).focus()`,
  );
  await call("Input.insertText", { text: "!" }, sessionId);
  assert.equal(
    await evaluate(
      `document.activeElement===document.querySelector(${JSON.stringify(stepField(0, "label"))})`,
    ),
    true,
  );
  await add("check");
  await field(setting(1, "expected"), "bc");
  const ids = await evaluate(
    'Array.from(document.querySelectorAll("[data-workflow-step]"),item=>item.dataset.workflowStep)',
  );
  await click("remove", 0);
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("earlier step")',
  );
  assert.equal(
    await evaluate('document.querySelectorAll("[data-workflow-step]").length'),
    2,
  );
  await click("up", 1);
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("earlier step")',
  );
  assert.deepEqual(
    await evaluate(
      'Array.from(document.querySelectorAll("[data-workflow-step]"),item=>item.dataset.workflowStep)',
    ),
    ids,
  );
  await add("agent");
  await field(stepField(2, "agent"), agentUri, true);
  await waitFor(
    'document.querySelector("[data-workflow-step]:nth-child(3)")?.textContent.includes("Read runtime status") && !document.querySelector("[data-workflow=save]").disabled',
  );
  await click("save");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("Workflow saved")',
  );
  const workflow_uri = await evaluate(
    'document.querySelector("[data-workflow-uri][aria-pressed=true]").dataset.workflowUri',
  );
  await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');const head=await backend.resource('workspace',${JSON.stringify(workflow_uri)});const d=JSON.parse(head.resource.content);d.steps[0].label='<script>Native stage</script>';await backend.updateResource('workspace',{...head.resource,content:JSON.stringify(d)},head.revision);})()`,
  );
  await field("#workflow-input", "a");
  await field("#workflow-model", "1", true);
  await click("run");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("changed in another client")',
  );
  await field(stepField(0, "label"), "Local unsaved stage");
  await click("save");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.toLowerCase().includes("conflict")',
  );
  assert.equal(
    await evaluate(
      `document.querySelector(${JSON.stringify(stepField(0, "label"))}).value`,
    ),
    "Local unsaved stage",
  );
  await click("new");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("Save or discard")',
  );
  await click("reload");
  await waitFor(
    'document.querySelector("#workflow-save-state")?.textContent==="Revision 2"',
  );
  assert.equal(
    await evaluate(
      'document.querySelector("[data-workflow-uri][aria-pressed=true] small").textContent',
    ),
    "Revision 2",
  );
  await field("#workflow-input", "a");
  await field("#workflow-model", "1", true);
  await click("run");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("Workflow completed")',
  );
  assert.equal(
    await evaluate('document.querySelector("#workflow-output").textContent'),
    "<script>Runtime reviewed</script>",
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".workflow-workspace script")!==null',
    ),
    false,
  );
  assert.equal(
    await evaluate('document.querySelectorAll(".workflow-trace>ol>li").length'),
    3,
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".workflow-trace").textContent.includes("2 generated tokens")',
    ),
    true,
  );
  await click("new");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("Save or discard")',
  );
  assert.equal(
    await evaluate('document.querySelector("#workflow-output").textContent'),
    "<script>Runtime reviewed</script>",
  );
  await click("save-run");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("run saved")',
  );
  const workflow_run_uri = await evaluate(
    'document.querySelector("[data-workflow-run-uri][aria-pressed=true]").dataset.workflowRunUri',
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "workflows"],
    ["fyodor-dark", 1280, "workflows-dark"],
    ["fyodor", 390, "workflows-mobile"],
  ]) {
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}})`,
    );
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    for (const [area, selector] of [
      ["definition", ".workflow-workspace"],
      ["result", ".workflow-launch"],
    ]) {
      await evaluate(
        `(()=>{const main=document.querySelector('.stage>main'),n=document.querySelector(${JSON.stringify(selector)});main.scrollTop=main.scrollTop+n.getBoundingClientRect().top-main.getBoundingClientRect().top;})()`,
      );
      await new Promise((resolve) => setTimeout(resolve, 200));
      assert.equal(
        await evaluate(
          'document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth',
        ),
        false,
      );
      await writeFile(
        path.join(output, label + "-" + area + ".png"),
        Buffer.from(
          (await call("Page.captureScreenshot", { format: "png" }, sessionId))
            .data,
          "base64",
        ),
      );
    }
  }
  await reload();
  await waitFor(
    '!!document.querySelector("[data-workflow-uri]:not(:disabled)")',
  );
  await evaluate(
    `document.querySelector('[data-workflow-uri="'+${JSON.stringify(workflow_uri)}+'"]').click()`,
  );
  await waitFor(
    '!!document.querySelector("[data-workflow-run-uri]:not(:disabled)")',
  );
  await click("inspect-agent", 2);
  await waitFor(
    'document.querySelector("[data-workflow-step]:nth-child(3)")?.textContent.includes("Read runtime status") && !document.querySelector("[data-workflow=save]").disabled',
  );
  assert.equal(
    await evaluate(
      'document.querySelector("#workflow-save-state").textContent',
    ),
    "Revision 2",
  );
  await evaluate(
    `document.querySelector('[data-workflow-run-uri="'+${JSON.stringify(workflow_run_uri)}+'"]').click()`,
  );
  await waitFor(
    'document.querySelector("#workflow-output")?.textContent==="<script>Runtime reviewed</script>"',
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".workflow-trace").textContent.includes("5 reported tokens")',
    ),
    true,
  );
  await configureAgentFixture({ evaluate, waitFor });
  await waitFor('!!document.querySelector("#workflow-input:not(:disabled)")');
  await field(stepField(2, "prefix"), "Stop this request");
  await click("save");
  await waitFor(
    'document.querySelector("#workflow-save-state")?.textContent==="Revision 3"',
  );
  await field("#workflow-input", "a");
  await field("#workflow-model", "1", true);
  await click("run");
  let started = false;
  for (let i = 0; i < 100; i++) {
    if (
      (
        await (
          await fetch(new URL("/qa/status", process.env.FYODOR_AGENT_ENDPOINT))
        ).json()
      ).delayed === 2
    ) {
      started = true;
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.ok(started, "Workflow Agent request did not start");
  await click("stop");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("Workflow stopped")',
  );
  assert.equal(
    await evaluate('document.querySelector("#workflow-output").textContent'),
    "bc",
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".workflow-trace").textContent.includes("Tokens not reported")',
    ),
    true,
  );
  await click("save-run");
  await waitFor(
    'document.querySelector("#workflow-status")?.textContent.includes("run saved")',
  );
  const workflow_stopped_uri = await evaluate(
    'document.querySelector("[data-workflow-run-uri][aria-pressed=true]").dataset.workflowRunUri',
  );
  assert.notEqual(workflow_run_uri, workflow_stopped_uri);
  return { workflow_uri, workflow_run_uri, workflow_stopped_uri };
}
