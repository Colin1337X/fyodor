// Optional Windows browser QA: private Edge profile, no user browser state.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {evaluationBrowser} from './evaluation-browser.mjs';
const profile=await mkdtemp(path.join(tmpdir(),'fyodor-theme-'));
const output=path.resolve('frontend/qa/themes');await mkdir(output,{recursive:true});
const browser=spawn(process.env.FYODOR_EDGE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  ['--headless=new','--no-first-run','--no-default-browser-check',`--user-data-dir=${profile}`,'--remote-debugging-port=0'],{windowsHide:true,stdio:'ignore'});
let socket;
try {
  let port;
  for(let i=0;i<100;++i){try{port=(await readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{await new Promise(r=>setTimeout(r,100));}}
  assert.ok(port,'Headless browser did not start');
  const endpoint=await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  socket=new WebSocket(endpoint.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  const pending=new Map();let next=1;
  socket.onmessage=({data})=>{const item=JSON.parse(data);if(pending.has(item.id)){const {resolve,reject,timer}=pending.get(item.id);clearTimeout(timer);pending.delete(item.id);item.error?reject(Error(JSON.stringify(item.error))):resolve(item.result);}};
  function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=next++;const timer=setTimeout(()=>{pending.delete(id);reject(Error(`Timeout ${method}`));},15000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,sessionId}));});}
  const {targetId}=await call('Target.createTarget',{url:process.env.FYODOR_PREVIEW || 'http://127.0.0.1:5178/'});
  const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
  await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:profile});
  await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false},sessionId);
  const evaluate=async expression=>{const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);assert.ok(!result.exceptionDetails,JSON.stringify(result.exceptionDetails));return result.result.value;};
  for(let i=0;i<50;++i){if(await evaluate('!!window.fyodorAppearance && !!document.querySelector(".app-shell")'))break;await new Promise(r=>setTimeout(r,100));}
  const reload=async()=>{
    const origin=await evaluate('performance.timeOrigin');
    await call('Page.reload',{},sessionId);
    for(let i=0;i<100;++i){
      try{if(await evaluate(`performance.timeOrigin!==${origin} && !!document.querySelector('.app-shell')`))return;}catch{}
      await new Promise(r=>setTimeout(r,50));
    }
    throw Error('Reload did not complete');
  };
  const results=[];
  const themes=await evaluate('window.fyodorThemes.map(t=>({id:t.id,background:t.roles.background,text:t.roles.text}))');
  for(const theme of themes){
    await evaluate(`window.fyodorAppearance.set({theme:${JSON.stringify(theme.id)}})`);
    await new Promise(r=>setTimeout(r,200));
    const actual=await evaluate('({theme:document.documentElement.dataset.theme,bg:getComputedStyle(document.body).backgroundColor,text:getComputedStyle(document.body).color,overflow:document.documentElement.scrollWidth>innerWidth})');
    const rgb=hex=>'rgb('+hex.slice(1).match(/../g).map(v=>parseInt(v,16)).join(', ')+')';
    assert.equal(actual.bg,rgb(theme.background));assert.equal(actual.text,rgb(theme.text));assert.equal(actual.overflow,false);
    const capture=await call('Page.captureScreenshot',{format:'png'},sessionId);
    await writeFile(path.join(output,`${theme.id}.png`),Buffer.from(capture.data,'base64'));results.push(actual);
  }
  assert.notEqual(await evaluate('getComputedStyle(document.querySelector(".panel")).boxShadow'),'none');
  await reload();await new Promise(r=>setTimeout(r,500));
  assert.equal(await evaluate('window.fyodorAppearance.get().theme'),'catppuccin-mocha');
  await evaluate('document.querySelector("#open-settings").click()');
  assert.equal(await evaluate('document.querySelectorAll("[data-theme-choice]").length'),14);
  await evaluate('document.querySelector("[data-theme-choice=strawberry]").click()');
  assert.equal(await evaluate('document.querySelector("[data-theme-choice=strawberry]").getAttribute("aria-pressed")'),'true');
  const settings=await call('Page.captureScreenshot',{format:'png'},sessionId);
  await writeFile(path.join(output,'settings.png'),Buffer.from(settings.data,'base64'));
  await evaluate('document.querySelector(".appearance-custom").open=true');
  const custom={accent:'#ffff00',fontSize:18,radius:0,font:'serif',mono:'courier',sidebarWidth:260,spacing:30,chatLayout:'wide',motion:'reduced',borders:false,shadows:false,translucency:true,codeColor:'#803040',syntaxKeyword:'#006600'};
  for(const [key,value] of Object.entries(custom))await evaluate(`(()=>{const n=document.querySelector('[data-appearance="${key}"]');${typeof value==='boolean'?'n.checked':'n.value'}=${JSON.stringify(value)};n.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const customized=await evaluate('({font:getComputedStyle(document.body).fontSize,radius:getComputedStyle(document.querySelector("#settings")).borderRadius,ink:getComputedStyle(document.documentElement).getPropertyValue("--on-accent"),shadow:getComputedStyle(document.querySelector("#settings")).boxShadow,rail:getComputedStyle(document.querySelector(".rail")).width})');
  assert.equal(customized.font,'18px');assert.equal(customized.radius,'0px');assert.equal(customized.ink,'#000000');assert.equal(customized.shadow,'none');assert.equal(customized.rail,'260px');
  const customCapture=await call('Page.captureScreenshot',{format:'png'},sessionId);
  await writeFile(path.join(output,'customization.png'),Buffer.from(customCapture.data,'base64'));
  await reload();await new Promise(r=>setTimeout(r,500));
  assert.equal(await evaluate('window.fyodorAppearance.get().fontSize'),18);
  await evaluate('document.querySelector("#open-settings").click();document.querySelector("#appearance-reset").click()');
  assert.equal(await evaluate('getComputedStyle(document.body).fontSize'),'14px');
  assert.equal(await evaluate('document.querySelector("#settings").open'),true);
  await evaluate(`(async()=>{const {messageBody}=await import('/src/message.js');const n=document.createElement('div');n.id='syntax-fixture';n.innerHTML=messageBody('~~~js\\nconst n = 42; // <script>x</script>\\n~~~'.replaceAll('~~~',String.fromCharCode(96).repeat(3)));document.body.append(n);window.fyodorAppearance.set({syntaxKeyword:'#006600'});})()`);
  assert.equal(await evaluate('document.querySelector("#syntax-fixture code").textContent'),'const n = 42; // <script>x</script>\n');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("#syntax-fixture .syntax-keyword")).color'),'rgb(0, 102, 0)');
  assert.equal(await evaluate('document.querySelector("#syntax-fixture script")'),null);
  if(process.env.FYODOR_RESOURCE_QA){
    const waitFor=async expression=>{for(let i=0;i<100;++i){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,50));}throw Error('Resource UI did not reach expected state: '+expression);};
    await evaluate('document.querySelector("#settings").close();document.querySelector("[data-view=resources]").click()');
    await waitFor('!!document.querySelector("[data-resource=new-document]:not(:disabled)")');
    await evaluate('document.querySelector("[data-resource=new-document]").click()');
    await evaluate(`(()=>{const title=document.querySelector('#resource-title'),content=document.querySelector('#resource-content');title.value='Browser document';content.value='<script>inert</script> saved from desktop';title.dispatchEvent(new Event('input'));content.dispatchEvent(new Event('input'));document.querySelector('[data-resource=save]').click();})()`);
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved to the local workspace")');
    const uri=await evaluate('document.querySelector("[data-resource-uri]").dataset.resourceUri');
    const resourceCapture=await call('Page.captureScreenshot',{format:'png'},sessionId);
    await writeFile(path.join(output,'resources.png'),Buffer.from(resourceCapture.data,'base64'));
    await reload();
    await waitFor('!!document.querySelector("[data-resource-uri]:not(:disabled)")');
    await evaluate('document.querySelector("[data-resource-uri]").click()');
    await waitFor('!!document.querySelector("#resource-content:not(:disabled)")');
    assert.equal(await evaluate('document.querySelector("#resource-content").value'),'<script>inert</script> saved from desktop');
    assert.equal(await evaluate('document.querySelector(".resource-editor script")'),null);
    await evaluate(`(()=>{const n=document.querySelector('#resource-content');n.value='second revision';n.dispatchEvent(new Event('input'));document.querySelector('[data-resource=save]').click();})()`);
    await waitFor('document.querySelectorAll("[data-history-revision]").length===2 && !document.querySelector("[data-resource=save]").disabled');
    await evaluate('[...document.querySelectorAll("[data-history-revision]")].find(n=>n.dataset.historyRevision==="1").click()');
    await waitFor('!!document.querySelector("#resource-history-content")');
    assert.equal(await evaluate('document.querySelector("#resource-history-content").textContent'),'<script>inert</script> saved from desktop');
    assert.equal(await evaluate('document.querySelector(".resource-history script")'),null);
    await evaluate('document.querySelector("[data-resource=copy-history]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("unsaved editor")');
    await evaluate('document.querySelector("[data-resource=save]").click()');
    await waitFor('document.querySelectorAll("[data-history-revision]").length===3 && !document.querySelector("[data-resource=save]").disabled');
    await evaluate('document.querySelector("[data-view=context]").click()');
    await waitFor('!!document.querySelector("[data-context-uri]:not(:disabled)")');
    await evaluate('document.querySelector("[data-context-uri]").click()');
    await evaluate(`(()=>{for(const [id,value] of [['context-prompt','a'],['context-budget','3'],['context-tokens','2']]){const n=document.querySelector('#'+id);n.value=value;n.dispatchEvent(new Event(id==='context-prompt'?'input':'change'));}document.querySelector('[data-context=generate]').click();})()`);
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("Context access denied")');
    await evaluate('document.querySelector("[data-context=grant]").click()');
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("set to read only")');
    await evaluate(`(()=>{const n=document.querySelector('[data-context-layer]');n.value='session';n.dispatchEvent(new Event('change'));})()`);
    await evaluate('document.querySelector("[data-context=generate]").click()');
    await waitFor('!!document.querySelector("#context-exact-prompt")');
    assert.equal(await evaluate('document.querySelector("#context-exact-prompt").textContent'),'<sc\n\na');
    assert.equal(await evaluate('JSON.parse(document.querySelector("#context-sources").textContent)[0].layer'),'session');
    assert.equal(await evaluate('document.querySelector(".context-source-details").textContent.includes("Prefix included")'),true);
    assert.equal(await evaluate('document.querySelector(".context-source-details pre").textContent'),'<sc');
    const receipt_id=await evaluate('document.querySelector("#context-receipt-id").value');
    assert.equal(await evaluate('document.querySelectorAll("[data-receipt-id]").length'),1);
    const principal=await evaluate('localStorage.getItem("fyodor-context-identity")');
    const contextCapture=await call('Page.captureScreenshot',{format:'png'},sessionId);
    await writeFile(path.join(output,'context.png'),Buffer.from(contextCapture.data,'base64'));
    for(const [theme,width,label] of [['fyodor',1280,'context-sources'],['fyodor-dark',1280,'context-sources-dark'],['fyodor',390,'context-sources-mobile']]){
      await evaluate(`window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.context-source-details').scrollIntoView({block:'center'})`);
      await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false},sessionId);await new Promise(r=>setTimeout(r,250));
      assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth'),false);
      await writeFile(path.join(output,label+'.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'},sessionId)).data,'base64'));
    }
    await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false},sessionId);
    await reload();
    await waitFor('!!document.querySelector("[data-context-uri]:not(:disabled)")');
    await waitFor('!!document.querySelector("[data-receipt-id]:not(:disabled)")');
    await evaluate('document.querySelector("[data-receipt-id]").click()');
    await waitFor('!!document.querySelector("#context-exact-prompt")');
    await evaluate('document.querySelector("[data-context-uri]").click();document.querySelector("[data-context=revoke]").click()');
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("revoked")');
    assert.equal(await evaluate('document.querySelectorAll("[data-receipt-id]").length'),0);
    await evaluate('document.querySelector("[data-context=receipt]").click()');
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("Context access denied")');
    assert.equal(await evaluate('!!document.querySelector("#context-exact-prompt")'),false);
    await evaluate('document.querySelector("[data-context=grant]").click()');
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("set to read only")');
    await evaluate(`(()=>{const n=document.querySelector('#context-query');n.value='<SCRIPT';n.dispatchEvent(new Event('input'));document.querySelector('[data-context=search]').click();})()`);
    await waitFor('!!document.querySelector("[data-context=search]:not(:disabled)")');
    assert.equal(await evaluate('document.querySelectorAll("[data-context-uri]").length'),0);
    await evaluate('document.querySelector("[data-context=grant-search]").click()');
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("set to read and search")');
    assert.equal(await evaluate('document.querySelectorAll("[data-context-uri]:checked").length'),1);
    await evaluate('document.querySelector("[data-context=revoke]").click()');
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("revoked")');
    assert.equal(await evaluate('document.querySelectorAll("[data-context-uri]").length'),0);
    await evaluate('document.querySelector("[data-context=grant-search]").click()');
    await waitFor('document.querySelector(".context-workspace [role=status]")?.textContent.includes("set to read and search")');
    await writeFile(process.env.FYODOR_RESOURCE_QA,JSON.stringify({uri,principal,receipt_id}));
    await evaluate('document.querySelector("[data-view=writing]").click()');
    await waitFor('!!document.querySelector("[data-resource=new-character]:not(:disabled)")');
    await evaluate('document.querySelector("[data-resource=new-character]").click()');
    await evaluate(`(()=>{const title=document.querySelector('#resource-title'),content=document.querySelector('#resource-content');title.value='Mira';content.value='A cartographer who keeps a journal.';title.dispatchEvent(new Event('input'));content.dispatchEvent(new Event('input'));document.querySelector('[data-resource=save]').click();})()`);
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved to the local workspace")');
    assert.equal(await evaluate('document.querySelector("[data-resource-uri*=characters]")!==null'),true);
    assert.equal(await evaluate('document.querySelector("#writing-count").textContent.startsWith("6 words")'),true);
    assert.equal(await evaluate('document.querySelector(".resource-editor").getBoundingClientRect().left < document.querySelector(".writing-sidebar").getBoundingClientRect().left'),true);
    await evaluate('document.querySelector("[data-writing-focus]").click()');
    assert.equal(await evaluate('getComputedStyle(document.querySelector(".writing-sidebar")).display'),'none');
    assert.equal(await evaluate('document.querySelector("#resource-content").value'),'A cartographer who keeps a journal.');
    await evaluate('document.querySelector("[data-resource=discard]").click()');
    assert.notEqual(await evaluate('getComputedStyle(document.querySelector(".writing-sidebar")).display'),'none');
    await evaluate('document.querySelector("[data-resource-uri*=characters]").click()');
    await waitFor('!!document.querySelector("[data-writing-focus]:not(:disabled)")');
    await evaluate('(()=>{document.querySelector("[data-writing-focus]").click();const n=document.querySelector("[data-writing-guidance=style]");n.value="Past tense.";n.dispatchEvent(new Event("input"));})()');
    // Guidance is retained through UI repaints but optional. Clear it before the
    // tiny model fixture's exact request/receipt assertions below.
    await evaluate('document.querySelector("[data-writing-context]").click()');
    assert.equal(await evaluate('document.querySelector("[data-writing-guidance=style]").value'),'Past tense.');
    await evaluate('(()=>{document.querySelector("[data-writing-context]").click();const n=document.querySelector("[data-writing-guidance=style]");n.value="";n.dispatchEvent(new Event("input"));})()');
    await evaluate(`(()=>{for(const [id,value] of [['writing-instructions','a'],['writing-budget','3'],['writing-tokens','2']]){const n=document.querySelector('#'+id);n.value=value;n.dispatchEvent(new Event(id==='writing-instructions'?'input':'change'));}document.querySelector('[data-writing-context]').click();document.querySelector('[data-writing-mode=continue]').click();})()`);
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Context access denied")');
    await evaluate('document.querySelector("[data-resource=writing-grant]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("grants set to read only")');
    for(const mode of ['generate','rewrite','continue']){
      await evaluate(`document.querySelector('[data-writing-mode=${mode}]').click()`);
      await waitFor('document.querySelector("[role=status]")?.textContent.includes("Preview generated and receipt saved")');
      assert.equal(await evaluate('document.querySelector("#resource-content").value'),'A cartographer who keeps a journal.');
      assert.ok((await evaluate('document.querySelector("#writing-exact-prompt").textContent')).includes(mode==='generate'?'Write:':mode==='rewrite'?'Rewrite:':'Continue:'));
    }
    const previewText=await evaluate('document.querySelector("#writing-output").textContent');
    await evaluate('document.querySelector("[data-resource=writing-apply]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("unsaved editor")');
    assert.equal(await evaluate('document.querySelector("#resource-content").value'),'A cartographer who keeps a journal.'+previewText);
    await evaluate('document.querySelector("[data-resource=save]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved to the local workspace")');
    assert.equal(await evaluate('document.querySelectorAll("[data-history-revision]").length'),2);
    const writing_uri=await evaluate('document.querySelector("[data-resource-uri*=characters]").dataset.resourceUri');
    const writing_receipt=await evaluate('document.querySelector(".writing-generate code").textContent');
    await writeFile(process.env.FYODOR_RESOURCE_QA,JSON.stringify({uri,principal,receipt_id,writing_uri,writing_receipt}));
    const writingCapture=await call('Page.captureScreenshot',{format:'png'},sessionId);
    await writeFile(path.join(output,'writing.png'),Buffer.from(writingCapture.data,'base64'));
    await reload();
    await waitFor('!!document.querySelector("[data-resource-uri*=characters]:not(:disabled)")');
    assert.equal(await evaluate('document.querySelector("h1").textContent'),'Writing');
    await evaluate('document.querySelector("[data-resource=new-project]").click()');
    await evaluate('(()=>{const n=document.querySelector("#resource-title");n.value="My atlas";n.dispatchEvent(new Event("input"));document.querySelector("[data-resource=save]").click();})()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved to the local workspace")');
    const project_uri=await evaluate('document.querySelector("[data-resource-uri*=projects]").dataset.resourceUri');
    await evaluate('document.querySelector("[data-resource-uri*=characters]").click()');
    await waitFor('!!document.querySelector("#writing-parent:not(:disabled)")');
    await evaluate(`document.querySelector('#writing-parent').value=${JSON.stringify(project_uri)};document.querySelector('[data-resource=move]').click()`);
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Resource moved")');
    assert.equal(await evaluate('document.querySelectorAll("[data-history-revision]").length'),3);
    await evaluate(`(()=>{const n=document.querySelector('#writing-folder');n.value=${JSON.stringify(project_uri)};n.dispatchEvent(new Event('change'));})()`);
    await waitFor('document.querySelectorAll("[data-resource-uri]").length===1 && !document.querySelector("#writing-folder").disabled');
    assert.equal(await evaluate('document.querySelector("[data-resource-uri]").dataset.resourceUri'),writing_uri);
    await reload();
    await waitFor('!!document.querySelector("[data-resource-uri*=characters]:not(:disabled)")');
    await evaluate('document.querySelector("[data-resource-uri*=characters]").click()');
    await waitFor('!!document.querySelector("#writing-parent:not(:disabled)")');
    assert.equal(await evaluate('document.querySelector("#writing-parent").value'),project_uri);
    // Grant the project once, then unselect it: subsequent inclusion must come
    // from the native folder provider, not a stale browser selection.
    await evaluate(`document.querySelector('[data-writing-context="'+${JSON.stringify(project_uri)}+'"]').click();document.querySelector('[data-resource=writing-grant]').click()`);
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("grants set to read only")');
    await evaluate(`document.querySelector('[data-writing-context="'+${JSON.stringify(project_uri)}+'"]').click();const tokens=document.querySelector('#writing-tokens');tokens.value='2';tokens.dispatchEvent(new Event('change'));document.querySelector('[data-writing-mode=continue]').click()`);
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Preview generated")');
    const autoSources=JSON.parse(await evaluate('document.querySelector("#writing-exact-prompt").nextElementSibling.textContent'));
    assert.deepEqual(autoSources.map(s=>[s.uri,s.layer]),[[writing_uri,'explicit'],[project_uri,'workspace']]);

    // Author the Writing lore fixture through Explore's real controls. No mock
    // gameplay state or API is involved; conflict injection is a second writer.
    await evaluate('document.querySelector("[data-view=explore]").click()');
    await waitFor('!!document.querySelector("[data-explore=new-world]:not(:disabled)")');
    await evaluate('document.querySelector("[data-explore=new-world]").click()');
    const editExplore=async(title,content)=>evaluate(`(()=>{for(const [key,value] of Object.entries(${JSON.stringify({title,content})})){const n=document.querySelector('#explore-'+key);n.value=value;n.dispatchEvent(new Event('input'));}})()`);
    await editExplore('Atlas','A world of sea glass.');
    await evaluate('document.querySelector("[data-explore=save]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved to your world library")');
    const world_uri=await evaluate('document.querySelector("[data-explore-uri]").dataset.exploreUri');
    await editExplore('Atlas','<script>inert</script> A world of sea glass.');
    await evaluate('document.querySelector("[data-explore=save]").click()');
    await waitFor('document.querySelectorAll("[data-explore-revision]").length===2 && !document.querySelector("#explore-title").disabled');
    await evaluate('document.querySelector(\'[data-explore-revision="1"]\').click()');
    await waitFor('!!document.querySelector("#explore-history-content") && !document.querySelector("#explore-title").disabled');
    await evaluate('document.querySelector("[data-explore=restore-text]").click();');
    await waitFor('document.querySelector(".fine-print")?.textContent.includes("Unsaved changes")');
    assert.equal(await evaluate('document.querySelector("#explore-content").value'),'A world of sea glass.');
    await evaluate('document.querySelector("[data-explore=lore]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Save or discard")');
    await evaluate('document.querySelector("[data-explore=reload]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved version loaded")');
    assert.equal(await evaluate('document.querySelector("#explore-content").value'),'<script>inert</script> A world of sea glass.');
    assert.equal(await evaluate('document.querySelector(".explore-workspace script")'),null);
    await evaluate(`(async()=>{const {backend}=await import('/src/api.js');await backend.updateResource('workspace',{uri:${JSON.stringify(world_uri)},title:'Atlas',content:'An independently edited world.'},'2');})()`);
    await editExplore('Atlas','A stale edit.');
    await evaluate('document.querySelector("[data-explore=save]").click()');
    await waitFor('!!document.querySelector(".danger-action[role=status]")?.textContent && !document.querySelector("#explore-title").disabled');
    assert.equal(await evaluate('document.querySelector("#explore-content").value'),'A stale edit.');
    await evaluate('document.querySelector("[data-explore=reload]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved version loaded")');
    assert.equal(await evaluate('document.querySelector("#explore-content").value'),'An independently edited world.');
    await evaluate('window.fyodorAppearance.set({theme:"fyodor"});document.querySelector(".stage>main").scrollTop=0');
    await new Promise(r=>setTimeout(r,250));
    await writeFile(path.join(output,'explore-world.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'},sessionId)).data,'base64'));
    await evaluate('document.querySelector("[data-explore=lore]").click()');
    await waitFor('!!document.querySelector("[data-explore=new-lore]:not(:disabled)")');
    await evaluate('document.querySelector("[data-explore=new-lore]").click()');
    await editExplore('The glass harbor','A harbor built from sea glass.');
    await evaluate('document.querySelector("[data-explore=save]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved to your world library")');
    const lore_uri=await evaluate('document.querySelector("[data-explore-uri]").dataset.exploreUri');
    assert.equal(lore_uri.startsWith(world_uri+'/lore/'),true);
    for(const [theme,width,label] of [['fyodor',1280,'explore-lore'],['fyodor-dark',1280,'explore-lore-dark'],['fyodor',390,'explore-lore-mobile']]){
      await evaluate(`window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.stage>main').scrollTop=0`);
      await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false},sessionId);
      await new Promise(r=>setTimeout(r,250));
      assert.equal(await evaluate('getComputedStyle(document.querySelector("#explore-content")).color===getComputedStyle(document.body).color'),true);
      assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth'),false);
      await writeFile(path.join(output,label+'.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'},sessionId)).data,'base64'));
    }
    await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false},sessionId);
    await reload();
    await waitFor('!!document.querySelector("[data-explore-uri]:not(:disabled)")');
    await evaluate('document.querySelector("[data-explore-uri]").click()');
    await waitFor('!!document.querySelector("[data-explore=lore]:not(:disabled)")');
    await evaluate('document.querySelector("[data-explore=lore]").click()');
    await waitFor('!!document.querySelector("[data-explore-uri]:not(:disabled)")');
    assert.equal(await evaluate('document.querySelector("[data-explore-uri]").dataset.exploreUri'),lore_uri);
    await evaluate('document.querySelector("[data-view=writing]").click()');
    await waitFor('!!document.querySelector("[data-resource-uri*=characters]:not(:disabled)")');
    await evaluate('document.querySelector("[data-resource-uri*=characters]").click()');
    await waitFor('!!document.querySelector("[data-lore-action=browse]:not(:disabled)")');
    await evaluate('const tokens=document.querySelector("#writing-tokens");tokens.value="2";tokens.dispatchEvent(new Event("change"));');
    await evaluate('document.querySelector(".writing-lore").open=true;document.querySelector("[data-lore-action=browse]").click()');
    await waitFor('!!document.querySelector("[data-lore-action=add]:not(:disabled)")');
    await evaluate('document.querySelector("[data-lore-action=add]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Lore linked")');
    assert.equal(await evaluate('document.querySelectorAll("[data-history-revision]").length'),4);
    await evaluate('document.querySelector("[data-lore-action=grant]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Model access allowed")');
    await evaluate('document.querySelector("[data-writing-mode=continue]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Preview generated")');
    const loreSources=JSON.parse(await evaluate('document.querySelector("#writing-exact-prompt").nextElementSibling.textContent'));
    assert.deepEqual(loreSources.map(s=>s.uri),[writing_uri,lore_uri,project_uri]);
    await reload();
    await waitFor('!!document.querySelector("[data-resource-uri*=characters]:not(:disabled)")');
    await evaluate('document.querySelector("[data-resource-uri*=characters]").click()');
    await waitFor('!!document.querySelector("[data-lore-action=remove]:not(:disabled)")');
    assert.equal(await evaluate('document.querySelector("[data-lore-action=remove]").dataset.loreUri'),lore_uri);
    assert.equal(await evaluate('document.querySelector(".writing-lore").textContent.includes("The glass harbor")'),true);
    await evaluate('window.fyodorAppearance.set({theme:"fyodor"});document.querySelector(".writing-lore").open=true;document.querySelector(".writing-lore").scrollIntoView({block:"center"})');
    await new Promise(r=>setTimeout(r,250));
    const loreCapture=await call('Page.captureScreenshot',{format:'png'},sessionId);
    await writeFile(path.join(output,'writing-lore.png'),Buffer.from(loreCapture.data,'base64'));

    await writeFile(process.env.FYODOR_RESOURCE_QA,JSON.stringify({uri,principal,receipt_id,writing_uri,writing_receipt,project_uri,lore_uri,world_uri}));
    await evaluate('document.querySelector(".stage>main").scrollTop=0');
    const foldersCapture=await call('Page.captureScreenshot',{format:'png'},sessionId);
    await writeFile(path.join(output,'writing-folders.png'),Buffer.from(foldersCapture.data,'base64'));
    for(const [theme,width,label] of [['fyodor',1280,'clay-writing'],['fyodor-dark',1280,'clay-writing-dark'],['fyodor',390,'clay-writing-mobile']]){
      await evaluate(`window.fyodorAppearance.set({theme:${JSON.stringify(theme)}})`);
      await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false},sessionId);
      await new Promise(r=>setTimeout(r,250));
      assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth'),false);
      const capture=await call('Page.captureScreenshot',{format:'png'},sessionId);
      await writeFile(path.join(output,label+'.png'),Buffer.from(capture.data,'base64'));
    }

    // Markdown/text import creates an unsaved document in the selected folder.
    // Downloads prove the actual draft bytes, independently of DOM display's
    // normalized textarea line endings. Native packages stay a separate format.
    await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false},sessionId);
    await evaluate(`(()=>{const picker=document.querySelector('#writing-folder');picker.value=${JSON.stringify(project_uri)};picker.dispatchEvent(new Event('change'));})()`);
    await waitFor('!!document.querySelector("#writing-text-import:not(:disabled)")');
    await evaluate('document.querySelector(".writing-library-actions").open=true');
    assert.equal(await evaluate('document.querySelector("#writing-text-import").parentElement.tabIndex'),0);
    assert.equal(await evaluate('(()=>{const input=document.querySelector("#writing-text-import"),control=input.parentElement;let activated=false;const original=input.click;input.click=()=>{activated=true;};control.focus();control.dispatchEvent(new KeyboardEvent("keydown",{key:"Enter",bubbles:true,cancelable:true}));input.click=original;return activated&&document.activeElement===control;})()'),true);
    const importedText='# Sea notes\r\n\r\n한글 👋 <script>inert</script>\n';
    const importText=async(content,name)=>evaluate(`(()=>{const transfer=new DataTransfer();transfer.items.add(new File([${JSON.stringify(content)}],${JSON.stringify(name)},{type:'text/plain'}));const input=document.querySelector('#writing-text-import');input.files=transfer.files;input.dispatchEvent(new Event('change'));})()`);
    await importText(importedText,'Sea notes.md');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Text imported into a new draft")');
    assert.equal(await evaluate('document.querySelector("#resource-title").value'),'Sea notes');
    assert.equal(await evaluate('document.querySelector("#resource-content").value'),importedText.replaceAll('\r\n','\n'));
    assert.equal(await evaluate('document.querySelector(".writing-workspace script")'),null);
    await importText('replacement','replacement.txt');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Save or discard")');
    assert.equal(await evaluate('document.querySelector("#resource-title").value'),'Sea notes');
    for(const [action,extension] of [['export-markdown','md'],['export-text','txt']]){
      await evaluate(`document.querySelector('[data-resource=${action}]').click()`);
      await waitFor('document.querySelector("[role=status]")?.textContent.includes("Current draft exported")');
      let downloaded;
      for(let i=0;i<100;i++){try{downloaded=await readFile(path.join(profile,'Sea notes.'+extension),'utf8');break;}catch{}await new Promise(r=>setTimeout(r,50));}
      assert.equal(downloaded,importedText);
    }
    await evaluate('document.querySelector("#resource-content").dispatchEvent(new KeyboardEvent("keydown",{key:"s",ctrlKey:true,bubbles:true,cancelable:true}))');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved to the local workspace")');
    const imported_uri=await evaluate('Array.from(document.querySelectorAll("[data-resource-uri]")).find(n=>n.querySelector("b").textContent==="Sea notes").dataset.resourceUri');
    await evaluate(`(async()=>{const {backend}=await import('/src/api.js');await backend.updateResource('workspace',{uri:${JSON.stringify(imported_uri)},title:'Sea notes',content:'Independent edit.'},'1');})()`);
    await evaluate('(()=>{const text=document.querySelector("#resource-content");text.value="My unsaved edit.";text.dispatchEvent(new Event("input"));document.querySelector("[data-resource=save]").click();})()');
    await waitFor('!!document.querySelector(".danger-action[role=status]")?.textContent && !document.querySelector("#resource-content").disabled');
    assert.equal(await evaluate('document.querySelector("#resource-content").value'),'My unsaved edit.');
    await evaluate('document.querySelector("[data-resource=reload]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved version loaded")');
    assert.equal(await evaluate('document.querySelector("#resource-content").value'),'Independent edit.');
    assert.equal(await evaluate('document.querySelector(".resource-editor .fine-print").textContent'),'Revision 2');
    await writeFile(process.env.FYODOR_RESOURCE_QA,JSON.stringify({uri,principal,receipt_id,writing_uri,writing_receipt,project_uri,lore_uri,world_uri,imported_uri}));
    await evaluate('document.querySelector("[data-view=datasets]").click()');
    await waitFor('!!document.querySelector("[data-dataset=new]:not(:disabled)")');
    await evaluate('document.querySelector("[data-dataset=new]").click()');
    await waitFor('!!document.querySelector("#dataset-mode:not(:disabled)")');
    await evaluate('(()=>{const mode=document.querySelector("#dataset-mode");mode.value="sft";mode.dispatchEvent(new Event("change"));})()');
    const datasetContent='aa\tb\nbb\tc\ncc\ta\naa\tb';
    await evaluate(`(()=>{for(const [key,value] of Object.entries(${JSON.stringify({title:'Supervised examples',content:datasetContent})})){const n=document.querySelector('#dataset-'+key);n.value=value;n.dispatchEvent(new Event('input'));}document.querySelector('[data-dataset=save]').click();})()`);
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Dataset saved")');
    const dataset_uri=await evaluate('document.querySelector("[data-dataset-uri]").dataset.datasetUri');
    assert.equal(await evaluate('document.querySelector(".dataset-stats").textContent.includes("1 exact duplicates")'),true);
    await evaluate('document.querySelector("[data-dataset-transform=deduplicate]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Derived draft prepared")');
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),'aa\tb\nbb\tc\ncc\ta');
    await evaluate('document.querySelector("[data-dataset=save]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Dataset saved")');
    const derived_uri=await evaluate('Array.from(document.querySelectorAll("[data-dataset-uri]")).find(n=>n.querySelector("b").textContent.endsWith("deduplicate")).dataset.datasetUri');
    assert.notEqual(derived_uri,dataset_uri);
    assert.equal(await evaluate(`(async()=>{const {backend}=await import('/src/api.js');return (await backend.resource('workspace',${JSON.stringify(dataset_uri)})).resource.content;})()`),datasetContent);
    await evaluate('document.querySelector("[data-dataset-transform=split]").click()');
    await waitFor('document.querySelectorAll("[data-dataset-part]").length===2 && !document.querySelector("[data-dataset-part]").disabled');
    await evaluate('document.querySelector(\'[data-dataset-part="0"]\').click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("training copy saved")');
    await evaluate('document.querySelector(\'[data-dataset-part="1"]\').click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("validation copy saved")');
    assert.equal(await evaluate('document.querySelectorAll("[data-dataset-part]:disabled").length'),2);
    await evaluate('document.querySelector("[data-dataset=export]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("training data exported")');
    let datasetDownload;
    for(let i=0;i<100;i++){try{datasetDownload=await readFile(path.join(profile,'Supervised examples · deduplicate.tsv'),'utf8');break;}catch{}await new Promise(r=>setTimeout(r,50));}
    assert.equal(datasetDownload,'aa\tb\nbb\tc\ncc\ta');
    await writeFile(process.env.FYODOR_RESOURCE_QA+'.dataset.tsv',datasetDownload);
    for(const [theme,width,label] of [['fyodor',1280,'datasets'],['fyodor-dark',1280,'datasets-dark'],['fyodor',390,'datasets-mobile']]){
      await evaluate(`window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.stage>main').scrollTop=0`);
      await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false},sessionId);await new Promise(r=>setTimeout(r,250));
      assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth'),false);
      await writeFile(path.join(output,label+'.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'},sessionId)).data,'base64'));
    }
    await reload();await waitFor('!!document.querySelector("[data-dataset-uri]:not(:disabled)")');
    await evaluate(`document.querySelector('[data-dataset-uri="'+${JSON.stringify(derived_uri)}+'"]').click()`);
    await waitFor('!!document.querySelector("#dataset-mode:not(:disabled)")');
    assert.equal(await evaluate('document.querySelector("#dataset-mode").value'),'sft');
    await evaluate('(()=>{const n=document.querySelector("#dataset-content");n.value="q\\t";n.dispatchEvent(new Event("input"));document.querySelector("[data-dataset=export]").click();})()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Fix format issues")');
    await evaluate('document.querySelector("[data-dataset=reload]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Saved version loaded")');
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),datasetDownload);
    // A format handoff must not accidentally reuse paths from an earlier run.
    await evaluate('document.querySelector("[data-view=training]").click()');
    await waitFor('!!document.querySelector("#train-data")');
    await evaluate('(()=>{for(const [id,value] of [["data","old-corpus.txt"],["evalData","old-validation.txt"],["mode","pretrain"]]){const n=document.querySelector("#train-"+id);n.value=value;n.dispatchEvent(new Event("change",{bubbles:true}));}})()');
    await evaluate('document.querySelector("[data-view=datasets]").click()');
    await waitFor('!!document.querySelector("[data-dataset=training]:not(:disabled)")');
    await evaluate('document.querySelector("[data-dataset=training]").click()');
    await waitFor('!!document.querySelector("#train-mode")');
    assert.equal(await evaluate('document.querySelector("#train-mode").value'),'sft');
    assert.equal(await evaluate('document.querySelector("#train-data").value'),'');
    assert.equal(await evaluate('document.querySelector("#train-evalData").value'),'');
    // Ingest saved revisions across libraries. An independent writer advances
    // the Writing head after selection; the source read must still use rev 2.
    await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false},sessionId);
    await evaluate('document.querySelector("[data-view=datasets]").click()');
    await waitFor('!!document.querySelector("[data-source=browse]:not(:disabled)")');
    await evaluate('document.querySelector(".dataset-source-picker").open=true;document.querySelector("[data-source=browse]").click()');
    await waitFor(`!!document.querySelector('[data-source-key="workspace:'+${JSON.stringify(imported_uri)}+'"]:not(:disabled)')`);
    await evaluate(`document.querySelector('[data-source-key="workspace:'+${JSON.stringify(imported_uri)}+'"]').click()`);
    await waitFor('document.querySelectorAll(".dataset-source-selection li").length===1');
    assert.equal(await evaluate('document.activeElement?.dataset.sourceKey'),'workspace:'+imported_uri);
    await evaluate(`(async()=>{const {backend}=await import('/src/api.js');await backend.updateResource('workspace',{uri:${JSON.stringify(imported_uri)},title:'Sea notes',content:'Newer saved text.'},'2');})()`);
    await evaluate('(()=>{const n=document.querySelector("#dataset-source-kind");n.value="explore";n.dispatchEvent(new Event("change"));})()');
    await waitFor(`!!document.querySelector('[data-source-key="workspace:'+${JSON.stringify(lore_uri)}+'"]:not(:disabled)')`);
    await evaluate(`document.querySelector('[data-source-key="workspace:'+${JSON.stringify(lore_uri)}+'"]').click()`);
    await waitFor('document.querySelectorAll(".dataset-source-selection li").length===2');
    await waitFor('!document.querySelector("#toast").classList.contains("show")');
    for(const [theme,width,label] of [['fyodor',1280,'datasets-sources'],['fyodor-dark',1280,'datasets-sources-dark'],['fyodor',390,'datasets-sources-mobile']]){
      await evaluate(`window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.stage>main').scrollTop=0`);
      await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false},sessionId);await new Promise(r=>setTimeout(r,250));
      assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth'),false);
      await writeFile(path.join(output,label+'.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'},sessionId)).data,'base64'));
    }
    await evaluate('document.querySelector("[data-source=build]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Source dataset draft prepared")');
    const corpus='Independent edit.\n\nA harbor built from sea glass.';
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),corpus);
    await evaluate('document.querySelector("[data-source=build]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Save or discard")');
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),corpus);
    await evaluate('document.querySelector("[data-dataset=save]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Dataset saved")');
    const corpus_uri=await evaluate('Array.from(document.querySelectorAll("[data-dataset-uri]")).find(n=>n.querySelector("b").textContent==="Selected sources · dataset").dataset.datasetUri');
    await evaluate('document.querySelector("[data-dataset=export]").click()');
    let corpusDownload;
    for(let i=0;i<100;i++){try{corpusDownload=await readFile(path.join(profile,'Selected sources · dataset.txt'),'utf8');break;}catch{}await new Promise(r=>setTimeout(r,50));}
    assert.equal(corpusDownload,corpus);await writeFile(process.env.FYODOR_RESOURCE_QA+'.source.txt',corpusDownload);

    // Import a real local chat through the public JSON control. Only the first
    // adjacent user/assistant pair is SFT; a tool breaks the later pair.
    const chatFixture={title:'Chat training examples',created:123,messages:[{role:'user',text:'a\nb'},{role:'assistant',text:'c'},{role:'user',text:'b'},{role:'tool',text:'tool output'},{role:'assistant',text:'a'}]};
    await evaluate(`(()=>{const transfer=new DataTransfer();transfer.items.add(new File([${JSON.stringify(JSON.stringify(chatFixture))}],'chat.json',{type:'application/json'}));const n=document.querySelector('#import-file');n.files=transfer.files;n.dispatchEvent(new Event('change',{bubbles:true}));})()`);
    await waitFor('document.querySelector("#chat-title")?.value==="Chat training examples"');
    await evaluate('document.querySelector("[data-view=datasets]").click()');
    await waitFor('!!document.querySelector("[data-source=clear]:not(:disabled)")');
    await evaluate('document.querySelector("[data-source=clear]").click()');
    await waitFor('document.querySelectorAll(".dataset-source-selection li").length===0 && !document.querySelector("#dataset-source-kind").disabled');
    await evaluate('(()=>{const n=document.querySelector("#dataset-source-kind");n.value="chats";n.dispatchEvent(new Event("change"));})()');
    await waitFor('!!document.querySelector("[data-source-key^=chat]:not(:disabled)")');
    await evaluate('Array.from(document.querySelectorAll(".dataset-source-list li")).find(n=>n.textContent.includes("Chat training examples")).querySelector("input").click()');
    await waitFor('document.querySelectorAll(".dataset-source-selection li").length===1');
    await evaluate('(()=>{const n=document.querySelector("#dataset-source-mode");n.value="sft";n.dispatchEvent(new Event("change"));})()');
    await evaluate('document.querySelector("[data-source=build]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("replacement policy")');
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),corpus);
    await evaluate('(()=>{const n=document.querySelector("#dataset-source-policy");n.value="spaces";n.dispatchEvent(new Event("change"));document.querySelector("[data-source=build]").click();})()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Source dataset draft prepared")');
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),'a b\tc');
    await evaluate('document.querySelector("[data-dataset=save]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Dataset saved")');
    const chat_dataset_uri=await evaluate('Array.from(document.querySelectorAll("[data-dataset-uri]")).find(n=>n.querySelector("b").textContent==="Chat training examples · dataset").dataset.datasetUri');
    assert.deepEqual(await evaluate('JSON.parse(localStorage.getItem("fyodor-state")).chats.find(c=>c.title==="Chat training examples").messages.map(({role,text})=>({role,text}))'),chatFixture.messages);
    await evaluate('document.querySelector("[data-dataset=export]").click()');
    let chatDownload;
    for(let i=0;i<100;i++){try{chatDownload=await readFile(path.join(profile,'Chat training examples · dataset.tsv'),'utf8');break;}catch{}await new Promise(r=>setTimeout(r,50));}
    assert.equal(chatDownload,'a b\tc');await writeFile(process.env.FYODOR_RESOURCE_QA+'.chat.tsv',chatDownload);
    // Context ingestion reads the included receipt slice through fresh native
    // authorization; a revoked source blocks creation even after UI selection.
    await evaluate('document.querySelector("[data-source=clear]").click()');
    await waitFor('document.querySelectorAll(".dataset-source-selection li").length===0 && !document.querySelector("#dataset-source-kind").disabled');
    await evaluate('(()=>{const n=document.querySelector("#dataset-source-kind");n.value="context";n.dispatchEvent(new Event("change"));})()');
    await waitFor(`!!document.querySelector('[data-source-receipt="'+${JSON.stringify(receipt_id)}+'"]:not(:disabled)')`);
    assert.equal(await evaluate('document.querySelector("#dataset-source-principal").value'),principal);
    await evaluate(`document.querySelector('[data-source-receipt="'+${JSON.stringify(receipt_id)}+'"]').click()`);
    await waitFor('!!document.querySelector("[data-source-key^=context]:not(:disabled)")');
    await evaluate('document.querySelector("[data-source-key^=context]").click()');
    await waitFor('document.querySelectorAll(".dataset-source-selection li").length===1');
    await evaluate('(()=>{const n=document.querySelector("#dataset-source-mode");n.value="cpt";n.dispatchEvent(new Event("change"));})()');
    // Editing the ID field must not relabel source rows from the loaded receipt.
    await evaluate(`(()=>{const n=document.querySelector('#dataset-source-receipt');n.value=${JSON.stringify(principal)};n.dispatchEvent(new Event('input'));})()`);
    assert.equal(await evaluate('document.querySelector(".dataset-source-selection").textContent.includes('+JSON.stringify(receipt_id)+')'),true);
    await evaluate(`(async()=>{const {backend}=await import('/src/api.js');await backend.contextPermissions('workspace',${JSON.stringify(principal)},${JSON.stringify(uri)},0);})()`);
    await evaluate('document.querySelector("[data-source=build]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Context access denied")');
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),'a b\tc');
    await evaluate(`(async()=>{const {backend}=await import('/src/api.js');await backend.contextPermissions('workspace',${JSON.stringify(principal)},${JSON.stringify(uri)},5);})()`);
    for(const [theme,width,label] of [['fyodor',1280,'datasets-context'],['fyodor-dark',1280,'datasets-context-dark'],['fyodor',390,'datasets-context-mobile']]){
      await evaluate(`window.fyodorAppearance.set({theme:${JSON.stringify(theme)}});document.querySelector('.stage>main').scrollTop=0`);
      await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false},sessionId);await new Promise(r=>setTimeout(r,250));
      assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth || document.querySelector(".stage>main").scrollWidth>document.querySelector(".stage>main").clientWidth'),false);
      await writeFile(path.join(output,label+'.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'},sessionId)).data,'base64'));
    }
    await evaluate('document.querySelector("[data-source=build]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Source dataset draft prepared")');
    assert.equal(await evaluate('document.querySelector("#dataset-content").value'),'<sc');
    await evaluate('document.querySelector("[data-dataset=save]").click()');
    await waitFor('document.querySelector("[role=status]")?.textContent.includes("Dataset saved")');
    const context_dataset_uri=await evaluate('Array.from(document.querySelectorAll("[data-dataset-uri]")).find(n=>n.querySelector("b").textContent.startsWith("Context · ")).dataset.datasetUri');
    await evaluate('document.querySelector("[data-dataset=export]").click()');
    // A long typed URI is sanitized/shortened by the shared download-name helper.
    const contextFilename=await evaluate(`(async()=>{const {writingDownloadName}=await import('/src/writing-files.js');return writingDownloadName(document.querySelector('#dataset-title').value,'txt');})()`);
    let contextDownload;
    for(let i=0;i<100;i++){try{contextDownload=await readFile(path.join(profile,contextFilename),'utf8');break;}catch{}await new Promise(r=>setTimeout(r,50));}
    assert.equal(contextDownload,'<sc');await writeFile(process.env.FYODOR_RESOURCE_QA+'.context.txt',contextDownload);
    const evaluations=await evaluationBrowser({evaluate,waitFor,reload,call,sessionId,output,profile});
    await writeFile(process.env.FYODOR_RESOURCE_QA,JSON.stringify({uri,principal,receipt_id,writing_uri,writing_receipt,project_uri,lore_uri,world_uri,imported_uri,dataset_uri,derived_uri,corpus_uri,chat_dataset_uri,context_dataset_uri,...evaluations}));



  }
  await writeFile(path.join(output,'results.json'),JSON.stringify(results,null,2));
  console.log(`${themes.length} browser palettes, computed colors, overflow and reload persistence passed`);
  await call('Browser.close');
} finally {
  socket?.close();
  if(browser.exitCode===null){browser.kill();await new Promise(resolve=>{browser.once('exit',resolve);setTimeout(resolve,3000);});}
  assert.ok(path.resolve(profile).startsWith(path.resolve(tmpdir())+path.sep+'fyodor-theme-'));
  await rm(profile,{recursive:true,force:true,maxRetries:5,retryDelay:100}).catch(()=>{});
}
