// Real modal/keyboard behavior and guarded native revision reads. This helper
// operates only inside the owned synthetic browser/backend integration harness.
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import path from "node:path";
export async function switcherBrowser({
  evaluate: runExpression,
  waitFor,
  call,
  sessionId,
  output,
  sourceUri,
}) {
  const evaluate = async (expression) => {
    try {
      return await runExpression(expression);
    } catch (error) {
      throw Error(`${error.message}\nSwitcher expression: ${expression}`, {
        cause: error,
      });
    }
  };
  // Enter includes its character event so Chromium exercises native button
  // activation as well as the input's explicit keydown handler.
  const press = async (key, code, virtual, modifiers = 0) => {
    await call(
      "Input.dispatchKeyEvent",
      {
        type: "keyDown",
        key,
        code,
        windowsVirtualKeyCode: virtual,
        modifiers,
        ...(key === "Enter" ? { text: "\r", unmodifiedText: "\r" } : {}),
      },
      sessionId,
    );
    await call(
      "Input.dispatchKeyEvent",
      { type: "keyUp", key, code, windowsVirtualKeyCode: virtual, modifiers },
      sessionId,
    );
  };
  const input = async (text) =>
    evaluate(
      `(()=>{const n=document.querySelector('#switcher-query');n.value=${JSON.stringify(text)};n.dispatchEvent(new Event('input'));})()`,
    );
  const open = async () => {
    await press("k", "KeyK", 75, 2);
    await waitFor('document.querySelector("#quick-switcher").open');
  };
  await call(
    "Emulation.setDeviceMetricsOverride",
    { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false },
    sessionId,
  );
  await evaluate('document.querySelector("[data-view=writing]").click()');
  await waitFor(
    '!!document.querySelector("[data-resource=new-document]:not(:disabled)")',
  );
  await evaluate(
    'document.querySelector("[data-resource=new-document]").click()',
  );
  await evaluate(
    '(()=>{const n=document.querySelector("#resource-content");n.value="Unsaved writing for quick navigation.";n.dispatchEvent(new Event("input"));n.focus();n.setSelectionRange(3,9);})()',
  );
  await open();
  assert.equal(await evaluate("document.activeElement.id"), "switcher-query");
  await evaluate(
    '(()=>{document.querySelector("#quick-switcher").close();document.querySelector("#open-switcher").click();})()',
  );
  await waitFor(
    'document.querySelector("#quick-switcher").open && document.activeElement.id==="switcher-query"',
  );
  await press("Escape", "Escape", 27);
  assert.deepEqual(
    await evaluate(
      "({id:document.activeElement.id,start:document.activeElement.selectionStart,end:document.activeElement.selectionEnd})",
    ),
    { id: "resource-content", start: 3, end: 9 },
  );
  await open();
  await input("datasets");
  await press("ArrowDown", "ArrowDown", 40);
  assert.equal(
    await evaluate('document.activeElement.querySelector("b").textContent'),
    "Datasets",
  );
  await press("Enter", "Enter", 13);
  try {
    await waitFor(
      'document.body.dataset.view==="datasets" && !document.querySelector("#quick-switcher").open',
    );
  } catch (error) {
    throw Error(
      error.message +
        " " +
        JSON.stringify(
          await evaluate(
            '({view:document.body.dataset.view,open:document.querySelector("#quick-switcher").open,status:document.querySelector("#switcher-status").textContent,active:document.activeElement.outerHTML.slice(0,500)})',
          ),
        ),
    );
  }
  await open();
  await input("writing");
  await press("Enter", "Enter", 13);
  await waitFor(
    'document.body.dataset.view==="writing" && !!document.querySelector("#resource-content:not(:disabled)")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    "Unsaved writing for quick navigation.",
  );
  await open();
  await input("Chat training examples");
  await press("Enter", "Enter", 13);
  await waitFor(
    'document.querySelector("#chat-title")?.value==="Chat training examples"',
  );
  assert.equal(
    await evaluate('document.querySelectorAll(".message").length'),
    5,
  );
  await open();
  await input("settings");
  await press("Enter", "Enter", 13);
  await waitFor('document.querySelector("#settings").open');
  assert.equal(
    await evaluate('document.querySelectorAll("dialog[open]").length'),
    1,
  );
  await press("k", "KeyK", 75, 2);
  assert.equal(
    await evaluate('document.querySelector("#quick-switcher").open'),
    false,
  );
  await press("Escape", "Escape", 27);
  await open();
  await evaluate(
    'document.querySelector("[data-switcher-mode=resources]").click()',
  );
  await input("Evaluation seed records");
  await press("Enter", "Enter", 13);
  await waitFor(
    'document.querySelector("#switcher-status").textContent.includes("matches for") && !document.querySelector("#switcher-query").disabled',
  );
  assert.equal(
    await evaluate('document.querySelectorAll("[data-switcher-index]").length'),
    2,
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "quick-switcher"],
    ["fyodor-dark", 1280, "quick-switcher-dark"],
    ["fyodor", 390, "quick-switcher-mobile"],
  ]) {
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}})`,
    );
    await new Promise((r) => setTimeout(r, 250));
    assert.equal(
      await evaluate(
        'document.documentElement.scrollWidth>innerWidth || document.querySelector("#quick-switcher").scrollWidth>document.querySelector("#quick-switcher").clientWidth',
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
  // The result pins rev 2; advancing its head must not substitute rev 3.
  await evaluate(
    `(async()=>{const {backend}=await import('/src/api.js');await backend.updateResource('workspace',{uri:${JSON.stringify(sourceUri)},title:'Evaluation seed records',content:'a\tAFTER'},'2');})()`,
  );
  await evaluate(
    'Array.from(document.querySelectorAll("[data-switcher-index]")).find(n=>n.querySelector("b").textContent==="Evaluation seed records").click()',
  );
  await waitFor(
    'document.body.dataset.view==="resources" && document.querySelector("[role=status]")?.textContent.includes("Selected saved revision opened")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    "a\tWRONG",
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".resource-editor .fine-print").textContent',
    ),
    "Revision 2",
  );
  await evaluate(
    '(()=>{const n=document.querySelector("#resource-content");n.value="Attempted stale write.";n.dispatchEvent(new Event("input"));document.querySelector("[data-resource=save]").click();})()',
  );
  await waitFor(
    '!document.querySelector("[data-resource=save]").disabled && document.querySelector("[role=status]").textContent.length>0',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    "Attempted stale write.",
  );
  assert.equal(
    await evaluate(
      'document.querySelector(".resource-editor .fine-print").textContent.includes("Unsaved")',
    ),
    true,
  );
  await open();
  await evaluate(
    'document.querySelector("[data-switcher-mode=resources]").click()',
  );
  await input("Evaluation seed records");
  await press("Enter", "Enter", 13);
  await waitFor(
    'document.querySelector("#switcher-status").textContent.includes("matches for") && !document.querySelector("#switcher-query").disabled',
  );
  await evaluate('document.querySelector("[data-switcher-index]").click()');
  assert.equal(
    await evaluate('document.querySelector("#quick-switcher").open'),
    true,
  );
  assert.ok(
    await evaluate(
      'document.querySelector("#switcher-status").textContent.includes("Save or discard")',
    ),
  );
  await press("Escape", "Escape", 27);
  await evaluate('document.querySelector("[data-resource=discard]").click()');
  await evaluate('document.querySelector("[data-view=writing]").click()');
  await waitFor(
    '!!document.querySelector("[data-resource=discard]:not(:disabled)")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    "Unsaved writing for quick navigation.",
  );
  await evaluate('document.querySelector("[data-resource=discard]").click()');
  return { switcher_source_uri: sourceUri };
}
