import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

// Exercise real completed native Writing requests. Receipt revocation is a
// separate native permission write, never a fabricated denial response.
export async function writingGenerationBrowser({
  evaluate,
  waitFor,
  call,
  sessionId,
  profile,
  output,
}) {
  const base = await evaluate(
    'document.querySelector("#resource-content").value',
  );
  const outputText = await evaluate(
    'document.querySelector("#writing-output").textContent',
  );
  assert.equal(
    await evaluate(
      'getComputedStyle(document.querySelector("#writing-output")).fontFamily',
    ),
    await evaluate("getComputedStyle(document.body).fontFamily"),
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".writing-preview").parentElement.className',
    ),
    "writing-main",
  );
  assert.equal(
    await evaluate(
      'document.querySelector("[data-resource=writing-apply]").disabled',
    ),
    false,
  );
  const metrics = await evaluate(
    'JSON.parse(document.querySelector("#writing-preview-model").textContent)',
  );
  assert.equal(metrics.writing_mode, "continue");
  assert.equal(metrics.generated_tokens, 2);
  assert.ok(
    (
      await evaluate(
        'document.querySelector(".writing-preview .dataset-stats").textContent',
      )
    ).includes(`${metrics.prompt_tokens} prompt tokens`),
  );
  await evaluate(
    'document.querySelector(".writing-preview-comparison").open=true',
  );
  await waitFor(
    '!!document.querySelector(".writing-preview-comparison-body .revision-lines")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    base,
  );
  assert.equal(
    await evaluate('document.querySelector(".writing-preview script")'),
    null,
  );
  assert.ok(
    (
      await evaluate(
        'document.querySelector(".writing-preview .context-source-details").textContent',
      )
    ).includes("Prefix included"),
  );
  await evaluate(
    'document.querySelector("[data-resource=writing-preview-export]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Generated preview downloaded")',
  );
  let download;
  const filename = await evaluate(
    `(async()=>{const {writingDownloadName}=await import('/src/writing-files.js');return writingDownloadName(document.querySelector('#resource-title').value+' preview','txt');})()`,
  );
  for (let i = 0; i < 100; i++) {
    try {
      download = await readFile(path.join(profile, filename), "utf8");
      break;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.equal(download, outputText);
  const target = await evaluate(
    'document.querySelector("[data-resource-uri*=characters]").dataset.resourceUri',
  );
  await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');await backend.contextPermissions('workspace',localStorage.getItem('fyodor-context-identity'),${JSON.stringify(target)},0);document.querySelector('[data-resource=writing-preview-receipt]').click();})()`,
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Context access denied")',
  );
  assert.equal(
    await evaluate(
      'document.querySelector("[data-resource=writing-apply]").disabled',
    ),
    true,
  );
  assert.equal(
    await evaluate('document.querySelector("#writing-exact-prompt")'),
    null,
  );
  assert.equal(
    await evaluate('document.querySelector("#writing-output").textContent'),
    outputText,
  );
  await evaluate(
    'document.querySelector("[data-resource=writing-grant]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("grants set to read only")',
  );
  await evaluate(
    'document.querySelector("[data-resource=writing-preview-receipt]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Saved Writing receipt loaded")',
  );
  assert.equal(
    await evaluate(
      'document.querySelector("[data-resource=writing-apply]").disabled',
    ),
    false,
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "writing-review"],
    ["fyodor-dark", 1280, "writing-review-dark"],
    ["fyodor", 390, "writing-review-mobile"],
  ]) {
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.writing-preview-comparison').open=true;document.querySelector('.writing-preview').scrollIntoView({block:'start'})`,
    );
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    await new Promise((resolve) => setTimeout(resolve, 250));
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
    await evaluate(
      "document.querySelector('.writing-preview-receipt').open=true;document.querySelector('.writing-preview .context-source-details details').open=true;document.querySelector('.writing-preview-receipt').scrollIntoView({block:'start'})",
    );
    // Opening a previously hidden disclosure after a palette change starts its
    // background transition. Capture the settled control, not that interim frame.
    await new Promise((resolve) => setTimeout(resolve, 250));
    const colors = await evaluate(
      "(()=>{const button=document.querySelector('[data-resource=writing-preview-receipt]'),style=getComputedStyle(button),probe=document.createElement('span');probe.style.color='var(--text)';probe.style.background='var(--surface)';button.after(probe);const expected=getComputedStyle(probe),result={ink:style.color,background:style.backgroundColor,expectedInk:expected.color,expectedBackground:expected.backgroundColor};probe.remove();return result;})()",
    );
    assert.equal(colors.ink, colors.expectedInk);
    assert.equal(colors.background, colors.expectedBackground);
    assert.equal(
      await evaluate(
        'document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth',
      ),
      false,
    );
    await writeFile(
      path.join(output, label + "-sources.png"),
      Buffer.from(
        (await call("Page.captureScreenshot", { format: "png" }, sessionId))
          .data,
        "base64",
      ),
    );
    await evaluate(
      "document.querySelector('.writing-preview-receipt').open=false",
    );
  }
  await call(
    "Emulation.setDeviceMetricsOverride",
    { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false },
    sessionId,
  );
  await evaluate('window.fyodorAppearance.set({theme:"fyodor"})');
  await evaluate(
    "(()=>{const n=document.querySelector('#resource-content');n.focus();n.setSelectionRange(n.value.length,n.value.length);})()",
  );
  await call("Input.insertText", { text: " My edit." }, sessionId);
  assert.equal(
    await evaluate(
      'document.querySelector("[data-resource=writing-apply]").disabled',
    ),
    true,
  );
  assert.ok(
    (
      await evaluate(
        'document.querySelector("#writing-preview-state").textContent',
      )
    ).includes("editor changed"),
  );
  await evaluate('document.querySelector("[data-writing-focus]").click()');
  assert.equal(
    await evaluate(
      'document.querySelector("[data-resource=writing-apply]").disabled',
    ),
    true,
  );
  await evaluate(
    'document.querySelector("[data-resource=writing-preview-discard]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Preview discarded")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    base + " My edit.",
  );
  assert.equal(
    await evaluate('document.querySelector(".writing-preview")'),
    null,
  );
  await evaluate('document.querySelector("[data-resource=reload]").click()');
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Saved version loaded")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    base,
  );
  await evaluate(
    'document.querySelector("[data-writing-mode=continue]").click()',
  );
  await waitFor(
    'document.querySelector("[role=status]")?.textContent.includes("Preview generated and receipt saved")',
  );
  assert.equal(
    await evaluate('document.querySelector("#writing-output").textContent'),
    outputText,
  );
}
