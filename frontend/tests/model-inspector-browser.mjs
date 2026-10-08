import assert from "node:assert/strict";
import { readFile, writeFile, readdir } from "node:fs/promises";
import path from "node:path";

export async function modelInspectorBrowser({
  evaluate,
  waitFor,
  call,
  sessionId,
  output,
  profile,
}) {
  const pressEscape = async () => {
    await call(
      "Input.dispatchKeyEvent",
      {
        type: "keyDown",
        key: "Escape",
        code: "Escape",
        windowsVirtualKeyCode: 27,
      },
      sessionId,
    );
    await call(
      "Input.dispatchKeyEvent",
      {
        type: "keyUp",
        key: "Escape",
        code: "Escape",
        windowsVirtualKeyCode: 27,
      },
      sessionId,
    );
    await waitFor('!document.querySelector("#model-inspector").open');
  };
  await call(
    "Emulation.setDeviceMetricsOverride",
    { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false },
    sessionId,
  );
  await evaluate('document.querySelector("[data-view=models]").click()');
  await waitFor('document.querySelectorAll("[data-inspect]").length===2');
  const id = await evaluate(
    'Number(document.querySelector("[data-inspect]").dataset.inspect)',
  );
  const selectedBefore = await evaluate(
    'document.querySelector("#global-model").value',
  );
  const expected = await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');return {info:await backend.modelInfo(${id}),tokens:await backend.tokenize(${id},'a')};})()`,
  );
  await evaluate('document.querySelector("[data-inspect]").click()');
  await waitFor(
    'document.querySelector("#model-info-status")?.textContent.includes("details loaded")',
  );
  const fields = await evaluate(
    'Object.fromEntries(Array.from(document.querySelectorAll("#model-info-fields dl>div")).map(n=>[n.querySelector("dt").textContent,n.querySelector("dd").textContent]))',
  );
  assert.equal(fields["Context tokens"], String(expected.info.context_length));
  assert.equal(
    fields["Vocabulary size"],
    String(expected.info.vocabulary_size),
  );
  assert.equal(fields["Runtime model ID"], String(id));
  assert.equal(fields["Active compute"], "cpu");
  await evaluate(
    '(()=>{const n=document.querySelector("#model-tokenizer-text");n.value="a";n.dispatchEvent(new Event("input",{bubbles:true}));document.querySelector("#model-tokenize").click();})()',
  );
  await waitFor('!document.querySelector("#model-token-export").disabled');
  assert.equal(
    await evaluate(
      'document.querySelector("#model-tokenizer-result").textContent',
    ),
    expected.tokens.tokens.join(", "),
  );
  await evaluate('document.querySelector("#model-token-export").click()');
  let exported;
  for (let i = 0; i < 100; i++) {
    try {
      exported = JSON.parse(
        await readFile(path.join(profile, "fyodor-tokenization.json"), "utf8"),
      );
      break;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.equal(exported.text, "a");
  assert.equal(exported.model.id, id);
  assert.equal(exported.count, expected.tokens.count);
  assert.deepEqual(exported.tokens, expected.tokens.tokens);
  await evaluate(
    '(()=>{const n=document.querySelector("#model-tokenizer-text");n.value="b";n.dispatchEvent(new Event("input",{bubbles:true}));})()',
  );
  assert.equal(
    await evaluate(
      'document.querySelector("#model-token-export").disabled && document.querySelector("#model-tokenizer-result").hidden',
    ),
    true,
  );
  await evaluate(
    '(()=>{const n=document.querySelector("#model-tokenizer-text");n.value="a\u0000b";n.dispatchEvent(new Event("input",{bubbles:true}));document.querySelector("#model-tokenize").click();})()',
  );
  assert.ok(
    await evaluate(
      'document.querySelector("#model-tokenizer-status").textContent.includes("null characters")',
    ),
  );
  const long = await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');return backend.tokenize(${id},'a'.repeat(300));})()`,
  );
  await evaluate(
    '(()=>{const n=document.querySelector("#model-tokenizer-text");n.value="a".repeat(300);n.dispatchEvent(new Event("input",{bubbles:true}));document.querySelector("#model-tokenize").click();})()',
  );
  await waitFor('!document.querySelector("#model-token-export").disabled');
  assert.ok(long.count > 256);
  assert.equal(
    await evaluate(
      'document.querySelector("#model-tokenizer-result").textContent',
    ),
    long.tokens.slice(0, 256).join(", "),
  );
  assert.ok(
    await evaluate(
      'document.querySelector("#model-tokenizer-status").textContent.includes("first 256 IDs")',
    ),
  );
  await evaluate('document.querySelector("#model-token-export").click()');
  let longExport;
  for (let i = 0; i < 100; i++) {
    const files = (await readdir(profile)).filter((name) =>
      /^fyodor-tokenization.*\.json$/.test(name),
    );
    // CDP download policy may overwrite the original filename or allocate a
    // suffix. Accept only complete bytes for the new submitted text.
    for (const file of files) {
      try {
        const record = JSON.parse(
          await readFile(path.join(profile, file), "utf8"),
        );
        if (record.text === "a".repeat(300)) {
          longExport = record;
          break;
        }
      } catch {}
    }
    if (longExport) break;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.ok(longExport, "Complete long tokenizer download was not received.");
  assert.equal(longExport.text, "a".repeat(300));
  assert.equal(longExport.count, long.count);
  assert.deepEqual(longExport.tokens, long.tokens);
  for (const [theme, width, label] of [
    ["fyodor", 1280, "model-inspector"],
    ["fyodor-dark", 1280, "model-inspector-dark"],
    ["fyodor", 390, "model-inspector-mobile"],
  ]) {
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 1000, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('#model-inspector').scrollTop=0`,
    );
    await new Promise((resolve) => setTimeout(resolve, 250));
    assert.equal(
      await evaluate(
        'document.documentElement.scrollWidth>innerWidth || document.querySelector("#model-inspector").scrollWidth>document.querySelector("#model-inspector").clientWidth',
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
      'document.querySelector(".model-tokenizer").scrollIntoView({block:"start"})',
    );
    await new Promise((resolve) => setTimeout(resolve, 100));
    await writeFile(
      path.join(
        output,
        label.replace("model-inspector", "model-tokenizer") + ".png",
      ),
      Buffer.from(
        (await call("Page.captureScreenshot", { format: "png" }, sessionId))
          .data,
        "base64",
      ),
    );
  }
  await pressEscape();
  await waitFor(
    'document.querySelector("#model-inspector").children.length===0',
  );
  assert.equal(
    await evaluate("document.activeElement.dataset.inspect"),
    String(id),
  );
  assert.equal(
    await evaluate(
      'document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth',
    ),
    false,
  );
  // Delay one actual native HTTP response to exercise dismissal/replacement.
  // The response contents are not mocked; this checks only frontend lifetime.
  await evaluate(
    '(()=>{const original=window.fetch;window.__restoreInfoFetch=()=>{window.fetch=original;delete window.__restoreInfoFetch;delete window.__releaseInfo;};let delayed=false;window.fetch=async(...args)=>{const response=await original(...args);if(!delayed&&String(args[0]).endsWith("/api/v1/model/info")){delayed=true;return new Promise(resolve=>window.__releaseInfo=()=>resolve(response));}return response;};})()',
  );
  try {
    await evaluate('document.querySelector("[data-inspect]").click()');
    await waitFor('typeof window.__releaseInfo==="function"');
    await pressEscape();
    await evaluate('document.querySelectorAll("[data-inspect]")[1].click()');
    await waitFor(
      'document.querySelector("#model-info-status")?.textContent.includes("details loaded")',
    );
    const second = await evaluate(
      'document.querySelector("#model-inspector-title").textContent',
    );
    assert.equal(second, "comparison.trained.gguf");
    await evaluate("window.__releaseInfo()");
    await new Promise((resolve) => setTimeout(resolve, 250));
    assert.equal(
      await evaluate(
        'document.querySelector("#model-inspector-title").textContent',
      ),
      second,
    );
    await pressEscape();
  } finally {
    await evaluate("window.__restoreInfoFetch?.()");
  }
  assert.equal(
    await evaluate('document.querySelector("#global-model").value'),
    selectedBefore,
  );
}
