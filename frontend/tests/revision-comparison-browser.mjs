import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import path from "node:path";

// Uses an already-saved synthetic character. All comparison edits are discarded
// with explicit native reload; this helper does not advance a resource revision.
export async function revisionComparisonBrowser({
  evaluate,
  waitFor,
  call,
  sessionId,
  output,
  writingUri,
}) {
  await call(
    "Emulation.setDeviceMetricsOverride",
    { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false },
    sessionId,
  );
  await evaluate('document.querySelector("[data-view=writing]").click()');
  await waitFor('!!document.querySelector("#writing-folder:not(:disabled)")');
  await evaluate(
    '(()=>{const n=document.querySelector("#writing-folder");n.value="*";n.dispatchEvent(new Event("change"));})()',
  );
  await waitFor(
    `!!document.querySelector('[data-resource-uri="${writingUri}"]:not(:disabled)')`,
  );
  await evaluate(
    `document.querySelector('[data-resource-uri="${writingUri}"]').click()`,
  );
  await waitFor('!!document.querySelector("#resource-content:not(:disabled)")');
  const original = await evaluate(
    '({title:document.querySelector("#resource-title").value,text:document.querySelector("#resource-content").value,revision:document.querySelector(".resource-editor .fine-print").textContent})',
  );
  await evaluate(
    "document.querySelector('[data-history-revision=\"1\"]').click()",
  );
  await waitFor(
    '!!document.querySelector(".revision-comparison") && !document.querySelector("#resource-content").disabled',
  );
  await evaluate('document.querySelector(".revision-comparison").open=true');
  await waitFor(
    'document.querySelector(".revision-comparison-body").textContent.includes("Saved revision 1")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    original.text,
  );
  // Native text insertion emits the real bubbling input event. A synthetic
  // non-bubbling event would bypass the shared editor's delegated listener.
  await evaluate(
    'document.querySelector("#resource-title").focus();document.querySelector("#resource-title").select()',
  );
  assert.deepEqual(
    await evaluate(
      '({active:document.activeElement.id,dialogs:Array.from(document.querySelectorAll("dialog[open]")).map(n=>n.id),inert:document.querySelector(".stage").inert})',
    ),
    { active: "resource-title", dialogs: [], inert: false },
  );
  await call("Page.bringToFront", {}, sessionId);
  await call("Input.insertText", { text: "Mira & the map" }, sessionId);
  await evaluate(
    'document.querySelector("#resource-content").focus();document.querySelector("#resource-content").select()',
  );
  await call(
    "Input.insertText",
    {
      text: "A cartographer who keeps a journal.\n<script>New scene</script>\n😀",
    },
    sessionId,
  );
  await evaluate(
    'document.querySelector("#resource-content").setSelectionRange(5,11)',
  );
  try {
    await waitFor(
      'document.querySelector(".revision-comparison-body").textContent.includes("2 added lines · 0 removed lines")',
    );
  } catch (error) {
    throw Error(
      error.message +
        " " +
        JSON.stringify(
          await evaluate(
            '({comparison:document.querySelector(".revision-comparison-body").textContent,saved:document.querySelector("#resource-history-content").textContent,editor:document.querySelector("#resource-content").value,open:document.querySelector(".revision-comparison").open})',
          ),
        ),
    );
  }
  assert.deepEqual(
    await evaluate(
      "({id:document.activeElement.id,start:document.activeElement.selectionStart,end:document.activeElement.selectionEnd})",
    ),
    { id: "resource-content", start: 5, end: 11 },
  );
  assert.equal(
    await evaluate(
      'document.querySelectorAll(".revision-comparison script").length',
    ),
    0,
  );
  assert.ok(
    await evaluate(
      'document.querySelector(".revision-comparison-body").textContent.includes("unsaved editor") && document.querySelector(".revision-title-change").textContent.includes("Mira & the map")',
    ),
  );
  await evaluate(
    'document.querySelector("[data-resource=copy-history]").click()',
  );
  await waitFor(
    'document.querySelector(".resource-workspace > [role=status]").textContent.includes("Save or discard")',
  );
  assert.equal(
    await evaluate('document.querySelector("#resource-title").value'),
    "Mira & the map",
  );
  // A controller repaint recreates the comparison closed. Reopen explicitly.
  await evaluate('document.querySelector(".revision-comparison").open=true');
  await waitFor(
    'document.querySelector(".revision-comparison-body").textContent.includes("2 added lines")',
  );
  for (const [theme, width, label] of [
    ["fyodor", 1280, "revision-comparison"],
    ["fyodor-dark", 1280, "revision-comparison-dark"],
    ["fyodor", 390, "revision-comparison-mobile"],
  ]) {
    await call(
      "Emulation.setDeviceMetricsOverride",
      { width, height: 1000, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );
    await evaluate(
      `window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.revision-comparison').scrollIntoView({block:'center'})`,
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
  }
  // Leave while an update is queued. The old mount must not repaint Explore,
  // and returning to Writing must retain the draft until explicit reload.
  await evaluate(
    '(()=>{const n=document.querySelector("#resource-content");n.value="Leaving before refresh.";n.dispatchEvent(new Event("input",{bubbles:true}));document.querySelector("[data-view=explore]").click();})()',
  );
  await waitFor(
    'document.body.dataset.view==="explore" && !!document.querySelector("#explore-namespace:not(:disabled)")',
  );
  await new Promise((resolve) => setTimeout(resolve, 250));
  assert.equal(
    await evaluate('!!document.querySelector(".writing-workspace")'),
    false,
  );
  await evaluate('document.querySelector("[data-view=writing]").click()');
  await waitFor('!!document.querySelector("#resource-content:not(:disabled)")');
  assert.equal(
    await evaluate('document.querySelector("#resource-content").value'),
    "Leaving before refresh.",
  );
  await evaluate('document.querySelector("[data-resource=reload]").click()');
  await waitFor(
    'document.querySelector(".resource-workspace > [role=status]").textContent.includes("Saved version loaded")',
  );
  assert.deepEqual(
    await evaluate(
      '({title:document.querySelector("#resource-title").value,text:document.querySelector("#resource-content").value,revision:document.querySelector(".resource-editor .fine-print").textContent})',
    ),
    original,
  );
  assert.equal(
    await evaluate('!!document.querySelector(".revision-comparison")'),
    false,
  );
}
