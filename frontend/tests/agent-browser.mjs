import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import path from "node:path";
export async function agentBrowser({
  evaluate,
  waitFor,
  reload,
  call,
  sessionId,
  output,
}) {
  const endpoint = process.env.FYODOR_AGENT_ENDPOINT;
  assert.ok(endpoint);
  const click = async (name) => {
    await waitFor(
      `!!document.querySelector('[data-agent="${name}"]:not(:disabled)')`,
    );
    await evaluate(`document.querySelector('[data-agent="${name}"]').click()`);
  };
  const input = async (selector, value) =>
    evaluate(
      `(()=>{const n=document.querySelector(${JSON.stringify(selector)});n.value=${JSON.stringify(value)};n.dispatchEvent(new Event('input',{bubbles:true}));})()`,
    );
  const configure = async () => {
    await evaluate('document.querySelector("#open-settings").click()');
    await evaluate(
      `(()=>{const d=document.querySelector('#settings');d.querySelector('#provider-type').value='openai';d.querySelector('#provider-type').dispatchEvent(new Event('input',{bubbles:true}));d.querySelector('#provider-endpoint').value=${JSON.stringify(endpoint)};d.querySelector('#provider-model').value='fixture-agent-model';d.querySelector('#provider-key').value='agent-fixture-key';d.querySelector('#remember-key').checked=false;d.querySelector('#agent-enabled').checked=false;d.querySelector('#save-settings').click();})()`,
    );
    await waitFor('!document.querySelector("#settings").open');
  };
  await configure();
  await evaluate('document.querySelector("[data-view=agents]").click()');
  await click("new");
  await waitFor('!!document.querySelector("#agent-title:not(:disabled)")');
  await input("#agent-title", "Browser agent");
  await input(
    "#agent-instructions",
    "Review only the allowed runtime observation.",
  );
  assert.equal(
    await evaluate(
      'document.querySelectorAll("[data-agent-tool]:checked").length',
    ),
    0,
  );
  await evaluate(
    'document.querySelector("[data-agent-tool=runtime_status]").click()',
  );
  await click("save");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("profile saved")',
  );
  const agent_uri = await evaluate(
    'document.querySelector("[data-agent-uri][aria-pressed=true]").dataset.agentUri',
  );
  // Actual independent native writer: both the start snapshot and CAS save guard
  // must preserve local edits rather than run an obsolete permission profile.
  await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');const head=await backend.resource('workspace',${JSON.stringify(agent_uri)});const d=JSON.parse(head.resource.content);d.instructions='Reviewed profile from another client.';await backend.updateResource('workspace',{...head.resource,content:JSON.stringify(d)},head.revision);})()`,
  );
  await input("#agent-input", "Review runtime");
  await click("run");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("changed in another client")',
  );
  await input("#agent-instructions", "Local unsaved edit");
  await click("save");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.toLowerCase().includes("conflict")',
  );
  assert.equal(
    await evaluate('document.querySelector("#agent-instructions").value'),
    "Local unsaved edit",
  );
  await click("new");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("Save or discard")',
  );
  assert.equal(
    await evaluate('document.querySelector("#agent-title").value'),
    "Browser agent",
  );
  await click("reload");
  await waitFor(
    'document.querySelector("#agent-save-state")?.textContent==="Revision 2"',
  );
  assert.equal(
    await evaluate('document.querySelector("#agent-instructions").value'),
    "Reviewed profile from another client.",
  );
  assert.equal(
    await evaluate(
      'document.querySelector("[data-agent-uri][aria-pressed=true] small").textContent',
    ),
    "Revision 2",
  );
  await input("#agent-input", "Review runtime");
  await click("run");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("Run completed")',
  );
  assert.equal(
    await evaluate('document.querySelector("#agent-output").textContent'),
    "<script>Runtime reviewed</script>",
  );
  assert.equal(
    await evaluate('document.querySelector(".agent-result script")!==null'),
    false,
  );
  assert.equal(
    await evaluate('document.querySelectorAll(".agent-steps li").length'),
    3,
  );
  await click("new");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("Save or discard")',
  );
  assert.equal(
    await evaluate('document.querySelector("#agent-output").textContent'),
    "<script>Runtime reviewed</script>",
  );
  await click("save-run");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("run saved")',
  );
  const agent_run_uri = await evaluate(
    'document.querySelector("[data-agent-run-uri][aria-pressed=true]").dataset.agentRunUri',
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "agents"],
    ["fyodor-dark", 1280, "agents-dark"],
    ["fyodor", 390, "agents-mobile"],
  ]) {
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}})`,
    );
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    for (const [suffix, selector] of [
      ["profile", ".agent-workspace"],
      ["result", ".agent-launch"],
    ]) {
      await evaluate(
        `(()=>{const main=document.querySelector('.stage>main'), n=document.querySelector(${JSON.stringify(selector)});main.scrollTop=main.scrollTop+n.getBoundingClientRect().top-main.getBoundingClientRect().top;})()`,
      );
      await new Promise((resolve) => setTimeout(resolve, 200));
      assert.equal(
        await evaluate(
          'document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth',
        ),
        false,
      );
      await writeFile(
        path.join(output, label + "-" + suffix + ".png"),
        Buffer.from(
          (await call("Page.captureScreenshot", { format: "png" }, sessionId))
            .data,
          "base64",
        ),
      );
    }
  }
  await reload();
  await waitFor('!!document.querySelector("[data-agent-uri]:not(:disabled)")');
  await evaluate(
    `document.querySelector('[data-agent-uri="'+${JSON.stringify(agent_uri)}+'"]').click()`,
  );
  await waitFor(
    '!!document.querySelector("[data-agent-run-uri]:not(:disabled)")',
  );
  await evaluate(
    `document.querySelector('[data-agent-run-uri="'+${JSON.stringify(agent_run_uri)}+'"]').click()`,
  );
  await waitFor(
    'document.querySelector("#agent-output")?.textContent==="<script>Runtime reviewed</script>"',
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".agent-result").textContent.includes("5 reported tokens")',
    ),
    true,
  );
  // Keys were not remembered; run uses only the current configured credential.
  await configure();
  await waitFor('!!document.querySelector("#agent-input:not(:disabled)")');
  await input("#agent-input", "Stop this request");
  await click("run");
  let started = false;
  for (let i = 0; i < 100; i++) {
    if (
      (await (await fetch(new URL("/qa/status", endpoint))).json()).delayed ===
      1
    ) {
      started = true;
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.ok(started, "Delayed endpoint request did not start");
  await click("stop");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("Run stopped")',
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".agent-result").textContent.includes("Tokens not reported")',
    ),
    true,
  );
  await click("save-run");
  await waitFor(
    'document.querySelector("#agent-status")?.textContent.includes("run saved")',
  );
  const agent_stopped_uri = await evaluate(
    'document.querySelector("[data-agent-run-uri][aria-pressed=true]").dataset.agentRunUri',
  );
  assert.notEqual(agent_stopped_uri, agent_run_uri);
  return { agent_uri, agent_run_uri, agent_stopped_uri };
}
