// Exercise persisted case datasets and exact-revision mappings in the owned
// browser/native fixture. All files/resources are synthetic and disposable.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export async function evaluationDatasetsBrowser({
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
  const input = async (selector, value, type = "input") =>
    evaluate(
      `(()=>{const n=document.querySelector(${JSON.stringify(selector)});n.value=${JSON.stringify(value)};n.dispatchEvent(new Event(${JSON.stringify(type)}));})()`,
    );
  const imported = async (content, name) =>
    evaluate(
      `(()=>{const transfer=new DataTransfer();transfer.items.add(new File([${JSON.stringify(content)}],${JSON.stringify(name)},{type:'application/json'}));const n=document.querySelector('#evaluation-cases-file');n.files=transfer.files;n.dispatchEvent(new Event('change'));})()`,
    );
  await evaluate('document.querySelector("[data-view=datasets]").click()');
  await waitFor(
    '!!document.querySelector("[data-dataset=new]:not(:disabled)")',
  );
  await evaluate('document.querySelector("[data-dataset=new]").click()');
  await waitFor('!!document.querySelector("#dataset-mode:not(:disabled)")');
  await input("#dataset-mode", "sft", "change");
  const trainingBytes = Array.from({ length: 40 }, () => "a\tbc").join("\n");
  await input("#dataset-title", "Evaluation seed records");
  await input("#dataset-content", trainingBytes);
  await evaluate('document.querySelector("[data-dataset=save]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Dataset saved")',
  );
  const evaluation_source_uri = await evaluate(
    'Array.from(document.querySelectorAll("[data-dataset-uri]")).find(n=>n.querySelector("b").textContent==="Evaluation seed records").dataset.datasetUri',
  );
  await evaluate('document.querySelector("[data-view=evaluations]").click()');
  await waitFor(
    '!!document.querySelector("[data-evaluation-dataset=browse]:not(:disabled)")',
  );
  await evaluate(
    '(()=>{const n=document.querySelector(".evaluation-dataset-picker");n.open=true;n.dispatchEvent(new Event("toggle"));document.querySelector("[data-evaluation-dataset=browse]").click();})()',
  );
  await waitFor(
    `!!document.querySelector('[data-evaluation-dataset-uri="'+${JSON.stringify(evaluation_source_uri)}+'"]:not(:disabled)')`,
  );
  await evaluate(
    `document.querySelector('[data-evaluation-dataset-uri="'+${JSON.stringify(evaluation_source_uri)}+'"]').click()`,
  );
  await waitFor(
    '!!document.querySelector("#evaluation-dataset-target:not(:disabled)")',
  );
  await input("#evaluation-dataset-target", "dataset", "change");
  await input("#evaluation-dataset-count", "40");
  await input("#evaluation-dataset-mapping", "exact", "change");
  await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');await backend.updateResource('workspace',{uri:${JSON.stringify(evaluation_source_uri)},title:'Evaluation seed records',content:'a\tWRONG'},'1');})()`,
  );
  await evaluate(
    'document.querySelector("[data-evaluation-dataset=prepare]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Test draft prepared")',
  );
  assert.equal(
    await evaluate('document.querySelectorAll(".evaluation-case").length'),
    20,
  );
  assert.equal(
    await evaluate(
      'document.querySelector("[data-case-field=expected]").value',
    ),
    "bc",
  );
  await evaluate(
    'document.querySelector("[data-evaluation=cases-next]").click()',
  );
  await waitFor(
    "!!document.querySelector('[data-case-index=\"20\"]:not(:disabled)')",
  );
  await input(
    '[data-case-index="20"][data-case-field=label]',
    "<script>inert</script> test case",
  );
  await evaluate('document.querySelector("[data-evaluation=save]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Test dataset saved")',
  );
  const evaluation_case_dataset_uri = await evaluate(
    'document.querySelector("[data-evaluation-uri][aria-pressed=true]").dataset.evaluationUri',
  );
  await reload();
  await waitFor(
    '!!document.querySelector("#evaluation-library:not(:disabled)")',
  );
  await input("#evaluation-library", "datasets", "change");
  await waitFor(
    `!!document.querySelector('[data-evaluation-uri="'+${JSON.stringify(evaluation_case_dataset_uri)}+'"]:not(:disabled)')`,
  );
  await evaluate(
    `document.querySelector('[data-evaluation-uri="'+${JSON.stringify(evaluation_case_dataset_uri)}+'"]').click()`,
  );
  await waitFor(
    '!!document.querySelector("[data-evaluation=cases-next]:not(:disabled)")',
  );
  await evaluate(
    'document.querySelector("[data-evaluation=cases-next]").click()',
  );
  await waitFor(
    "!!document.querySelector('[data-case-index=\"20\"]:not(:disabled)')",
  );
  assert.equal(
    await evaluate(
      "document.querySelector('[data-case-index=\"20\"][data-case-field=label]').value",
    ),
    "<script>inert</script> test case",
  );
  assert.equal(
    await evaluate('document.querySelector(".evaluation-workspace script")'),
    null,
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "evaluation-datasets"],
    ["fyodor-dark", 1280, "evaluation-datasets-dark"],
    ["fyodor", 390, "evaluation-datasets-mobile"],
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
  await evaluate(
    'document.querySelector("[data-evaluation=download-cases]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("test cases downloaded")',
  );
  let downloaded;
  for (let i = 0; i < 100; i++) {
    try {
      downloaded = await readFile(
        path.join(profile, "Evaluation seed records · test cases.cases.json"),
        "utf8",
      );
      break;
    } catch {}
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.equal(JSON.parse(downloaded).cases.length, 40);
  await evaluate('document.querySelector("[data-evaluation=discard]").click()');
  await waitFor(
    '!document.querySelector("[data-evaluation=import-cases]").disabled',
  );
  await imported(downloaded, "Round-trip cases.json");
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Test cases imported")',
  );
  await evaluate('document.querySelector("[data-evaluation=save]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Test dataset saved")',
  );
  const evaluation_imported_dataset_uri = await evaluate(
    'document.querySelector("[data-evaluation-uri][aria-pressed=true]").dataset.evaluationUri',
  );
  await imported(
    '{"schema":2,"kind":"evaluation-dataset","cases":[]}',
    "Future cases.json",
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("schema 1")',
  );
  assert.equal(
    await evaluate('document.querySelector("#evaluation-title").value'),
    "Round-trip cases",
  );
  await evaluate(
    'document.querySelector("[data-evaluation-dataset=browse]").click()',
  );
  await waitFor(
    `!!document.querySelector('[data-evaluation-dataset-uri="'+${JSON.stringify(evaluation_imported_dataset_uri)}+'"]:not(:disabled)')`,
  );
  await evaluate(
    `document.querySelector('[data-evaluation-dataset-uri="'+${JSON.stringify(evaluation_imported_dataset_uri)}+'"]').click()`,
  );
  await waitFor(
    '!!document.querySelector("#evaluation-dataset-target:not(:disabled)")',
  );
  await input("#evaluation-dataset-target", "definition", "change");
  await input("#evaluation-dataset-count", "2");
  await input("#evaluation-dataset-method", "sample", "change");
  await input("#evaluation-dataset-seed", "7");
  await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');const {resource}=await backend.resource('workspace',${JSON.stringify(evaluation_imported_dataset_uri)});const value=JSON.parse(resource.content);value.cases.forEach(item=>item.prompt='b');await backend.updateResource('workspace',{...resource,content:JSON.stringify(value)},'1');})()`,
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "evaluation-dataset-selection"],
    ["fyodor-dark", 1280, "evaluation-dataset-selection-dark"],
    ["fyodor", 390, "evaluation-dataset-selection-mobile"],
  ]) {
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('#evaluation-dataset-picker').scrollIntoView({block:'start'})`,
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
  await evaluate(
    'document.querySelector("[data-evaluation-dataset=prepare]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Test draft prepared")',
  );
  assert.equal(
    await evaluate('document.querySelectorAll(".evaluation-case").length'),
    2,
  );
  assert.equal(
    await evaluate('document.querySelector("[data-case-field=prompt]").value'),
    "a",
  );
  await evaluate(
    'document.querySelector("[data-evaluation-dataset=prepare]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Save or discard")',
  );
  assert.equal(
    await evaluate('document.querySelectorAll(".evaluation-case").length'),
    2,
  );
  await input("[data-evaluation-setting=max_tokens]", "2");
  await evaluate('document.querySelector("[data-evaluation=save]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Evaluation saved")',
  );
  const evaluation_mapped_definition_uri = await evaluate(
    'document.querySelector("[data-evaluation-uri][aria-pressed=true]").dataset.evaluationUri',
  );
  await evaluate(
    'document.querySelectorAll("[data-evaluation-model]").forEach(n=>{if(!n.checked)n.click();});document.querySelector("[data-evaluation=run]").click()',
  );
  await waitFor(
    '!!document.querySelector("[data-evaluation-results-save]:not(:disabled)")',
  );
  assert.deepEqual(
    await evaluate(
      'Array.from(document.querySelectorAll("#evaluation-results .evaluation-result .resource-toolbar b")).map(n=>n.textContent)',
    ),
    ["pass", "pass", "pass", "pass"],
  );
  await evaluate(
    'document.querySelector("[data-evaluation-results-save]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Evaluation results saved")',
  );
  const evaluation_mapped_run_uri = await evaluate(
    'document.querySelector("[data-evaluation-uri][aria-pressed=true]").dataset.evaluationUri',
  );
  return {
    evaluation_source_uri,
    evaluation_case_dataset_uri,
    evaluation_imported_dataset_uri,
    evaluation_mapped_definition_uri,
    evaluation_mapped_run_uri,
  };
}
