// Used by the owned browser/native harness. No user models or workspace data.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export async function evaluationBrowser({
  evaluate,
  waitFor,
  reload,
  call,
  sessionId,
  output,
  profile,
}) {
  await call(
    "Emulation.setDeviceMetricsOverride",
    { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false },
    sessionId,
  );
  await evaluate('document.querySelector("[data-view=evaluations]").click()');
  await waitFor(
    '!!document.querySelector("[data-evaluation=new]:not(:disabled)")',
  );
  await evaluate('document.querySelector("[data-evaluation=new]").click()');
  await waitFor('!!document.querySelector("#evaluation-title:not(:disabled)")');
  await evaluate(
    `(()=>{const title=document.querySelector('#evaluation-title');title.value='Local model quality';title.dispatchEvent(new Event('input'));const tokens=document.querySelector('[data-evaluation-setting=max_tokens]');tokens.value='2';tokens.dispatchEvent(new Event('input'));})()`,
  );
  for (const item of [
    { label: "Exact continuation", prompt: "a", type: "exact", expected: "bc" },
    { label: "Human review", prompt: "a", type: "manual" },
    { label: "Context overflow", prompt: "a".repeat(100), type: "manual" },
  ]) {
    const index = await evaluate(
      'document.querySelectorAll(".evaluation-case").length',
    );
    await evaluate('document.querySelector("[data-evaluation=case]").click()');
    await waitFor(
      `document.querySelectorAll('.evaluation-case').length===${index + 1} && !document.querySelector('[data-evaluation=case]').disabled`,
    );
    await evaluate(
      `(()=>{const data=${JSON.stringify(item)};for(const field of ['label','prompt','type']){const n=document.querySelector('[data-case-index="${index}"][data-case-field="'+field+'"]');n.value=data[field];n.dispatchEvent(new Event(field==='type'?'change':'input'));}if(data.expected){const n=document.querySelector('[data-case-index="${index}"][data-case-field=expected]');n.value=data.expected;n.dispatchEvent(new Event('input'));}})()`,
    );
  }
  await evaluate('document.querySelector("[data-evaluation=save]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Evaluation saved") && !document.querySelector("[data-evaluation=run]").disabled',
  );
  const evaluation_definition_uri = await evaluate(
    'document.querySelector("[data-evaluation-uri]").dataset.evaluationUri',
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "evaluations-definition"],
    ["fyodor-dark", 1280, "evaluations-definition-dark"],
    ["fyodor", 390, "evaluations-definition-mobile"],
  ]) {
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.stage>main').scrollTop=0`,
    );
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    await new Promise((r) => setTimeout(r, 250));
    assert.equal(
      await evaluate(
        'document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth',
      ),
      false,
    );
    await writeFile(
      path.join(output, label + ".png"),
      Buffer.from(
        (await call("Page.captureScreenshot", { format: "png" }, sessionId))
          .data,
        "base64",
      ),
    );
  }
  await call(
    "Emulation.setDeviceMetricsOverride",
    { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false },
    sessionId,
  );
  assert.equal(
    await evaluate(
      'document.querySelectorAll("[data-evaluation-model]").length',
    ),
    2,
  );
  await evaluate(
    'document.querySelectorAll("[data-evaluation-model]").forEach(n=>{if(!n.checked)n.click();});document.querySelector("[data-evaluation=run]").click()',
  );
  await waitFor(
    '!!document.querySelector("[data-evaluation-results-save]:not(:disabled)")',
  );
  assert.equal(
    await evaluate(
      'document.querySelectorAll("#evaluation-results .evaluation-result").length',
    ),
    6,
  );
  assert.deepEqual(
    await evaluate(
      'Array.from(document.querySelectorAll("#evaluation-results .evaluation-result .resource-toolbar b")).map(n=>n.textContent)',
    ),
    ["pass", "unreviewed", "error", "pass", "unreviewed", "error"],
  );
  await evaluate(
    'document.querySelector("[data-evaluation-results-save]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Evaluation results saved")',
  );
  const evaluation_run_uri = await evaluate(
    'document.querySelector("[data-evaluation-uri][aria-pressed=true]").dataset.evaluationUri',
  );
  const observations = await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');return JSON.parse((await backend.resource('workspace',${JSON.stringify(evaluation_run_uri)})).resource.content);})()`,
  );
  assert.equal(observations.definition.uri, evaluation_definition_uri);
  assert.equal(observations.definition.revision, "1");
  for (const row of observations.results) {
    assert.ok(row.latency_ms >= 0);
    if (row.status === "error") {
      assert.equal(row.generated_tokens, null);
      assert.equal(row.prompt_tokens, null);
      assert.equal(row.observed_tokens_per_second, null);
    } else {
      assert.equal(row.text, "bc");
      assert.equal(row.generated_tokens, 2);
      assert.ok(row.prompt_tokens > 0);
      assert.equal(
        row.observed_tokens_per_second,
        row.latency_ms > 0 ? 2000 / row.latency_ms : null,
      );
    }
  }
  await reload();
  await waitFor(
    `!!document.querySelector('[data-evaluation-uri="'+${JSON.stringify(evaluation_run_uri)}+'"]:not(:disabled)')`,
  );
  await evaluate(
    `document.querySelector('[data-evaluation-uri="'+${JSON.stringify(evaluation_run_uri)}+'"]').click()`,
  );
  await waitFor(
    '!!document.querySelector("[data-review-verdict]:not(:disabled)")',
  );
  await evaluate('document.querySelector(".evaluation-run-details").open=true');
  assert.equal(
    await evaluate(
      'document.querySelector(".evaluation-run-details").textContent.includes(' +
        JSON.stringify(evaluation_definition_uri) +
        ")",
    ),
    true,
  );
  await evaluate(
    '(()=>{const n=document.querySelector("[data-review-verdict]");n.value="pass";n.dispatchEvent(new Event("change"));})()',
  );
  assert.equal(
    await evaluate("document.activeElement?.dataset.reviewVerdict"),
    "1",
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".evaluation-table tbody tr").children[2].textContent',
    ),
    "2 / 0",
  );
  // Advance the saved head independently, then ensure a failed CAS retains review.
  await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');const {resource}=await backend.resource('workspace',${JSON.stringify(evaluation_run_uri)});await backend.updateResource('workspace',{...resource,title:'Independent title'},'1');})()`,
  );
  await evaluate('document.querySelector("[data-evaluation=save]").click()');
  await waitFor(
    '!document.querySelector("[data-evaluation=save]").disabled && document.querySelector("[role=status]").textContent.length>0',
  );
  assert.equal(
    await evaluate('document.querySelector("[data-review-verdict]").value'),
    "pass",
  );
  assert.equal(
    await evaluate(
      'document.querySelector("#evaluation-save-state").textContent.includes("Unsaved")',
    ),
    true,
  );
  await evaluate('document.querySelector("[data-evaluation=reload]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Saved version loaded")',
  );
  assert.equal(
    await evaluate('document.querySelector("[data-review-verdict]").value'),
    "unreviewed",
  );
  await evaluate(
    '(()=>{const n=document.querySelector("[data-review-verdict]");n.value="pass";n.dispatchEvent(new Event("change"));const note=document.querySelector("[data-review-verdict]").closest("article").querySelector("[data-review-note]");note.value="<script>inert</script> reviewed";note.dispatchEvent(new Event("input"));document.querySelector("[data-evaluation=save]").click();})()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Evaluation saved")',
  );
  assert.equal(
    await evaluate(
      'document.querySelector("#evaluation-save-state").textContent',
    ),
    "Revision 3",
  );
  assert.equal(
    await evaluate('document.querySelector(".evaluation-workspace script")'),
    null,
  );
  await evaluate('document.querySelector("[data-evaluation=export]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("package exported")',
  );
  let downloaded;
  for (let i = 0; i < 100; i++) {
    try {
      downloaded = JSON.parse(
        await readFile(path.join(profile, "Independent title.json"), "utf8"),
      );
      break;
    } catch {}
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.equal(downloaded.uri, evaluation_run_uri);
  assert.equal(
    JSON.parse(downloaded.content).results[1].review_note,
    "<script>inert</script> reviewed",
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "evaluations"],
    ["fyodor-dark", 1280, "evaluations-dark"],
    ["fyodor", 390, "evaluations-mobile"],
  ]) {
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.stage>main').scrollTop=0;document.querySelector('.evaluation-result details').open=true`,
    );
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    await new Promise((r) => setTimeout(r, 250));
    assert.equal(
      await evaluate(
        'document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth',
      ),
      false,
    );
    await writeFile(
      path.join(output, label + ".png"),
      Buffer.from(
        (await call("Page.captureScreenshot", { format: "png" }, sessionId))
          .data,
        "base64",
      ),
    );
  }
  return { evaluation_definition_uri, evaluation_run_uri };
}
