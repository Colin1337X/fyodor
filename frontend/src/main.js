import"./styles.css";
import{backend}from"./api.js";
import {formatConversation, normalizeChat, generationOptions} from "./session.js";
import {workspaceMarkup, apiMarkup} from "./workspace.js";
import{chooseCheckpoint,chooseDataset,chooseModel,chooseOutput,getBackendConnection,invokeDesktop}from"./platform.js";

// Storage and imported chats are untrusted. A corrupt value must not blank the app.
let persisted={};
try { persisted=JSON.parse(localStorage.getItem("fyodor-state")||"{}")||{}; } catch {}
if(!["workspace","chat","playground","training","logs","api"].includes(persisted.view))persisted.view="workspace";
const restored=[];
for(const entry of Array.isArray(persisted.chats)?persisted.chats.slice(0,300):[]) {
  try { const c=normalizeChat(entry,crypto.randomUUID());if(entry.id===persisted.active)persisted.active=c.id;restored.push(c); } catch {}
}
persisted.chats=restored;
const defaults={temperature:.8,top_p:.95,top_k:40,max_tokens:256,seed:"",system:"",frequency_penalty:0,presence_penalty:0,stop:"",draft_model_id:"",speculative_tokens:4};
const trainingDefaults={mode:"pretrain",data:"",output:"",base:"",checkpoint:"",resume:"",steps:100,learningRate:.001,rank:0,beta:.1,context:256,dimension:64,feedForward:128,layers:2,heads:4,kvHeads:2,seed:42,memoryMib:256};
const s={
  view:persisted.view||"workspace",health:null,runtime:null,baseUrl:"",actionBusy:false,models:[],selected:persisted.selected||null,
  chats:persisted.chats||[],active:persisted.active||null,params:{...defaults,...persisted.params},
  provider:{type:"local",endpoint:"https://api.openai.com/v1",apiKey:"",model:"",...persisted.provider},
  agent:{enabled:false,maxTurns:6,...persisted.agent},training:{...trainingDefaults,...persisted.training},
  trainStatus:{running:false,exitCode:null},logs:[],appLogs:[],busy:false,query:"",lastPrompt:"",
};
const meta={api:["API access","local REST interfaces"],workspace:["Workspace","models and runtime"],chat:["Chat","conversation"],playground:["Playground","generation controls"],training:["Training","native C trainer"],logs:["Logs","engine and trainer"]};
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const id=()=>crypto.randomUUID?.()||Date.now()+Math.random()+"";
const mid=m=>m.id??m.model_id;
const mname=m=>m.name||m.path?.split(/[\\/]/).pop()||`model ${mid(m)}`;
const root=document.querySelector("#app");
function save(){try{localStorage.setItem("fyodor-state",JSON.stringify({view:s.view,selected:s.selected,chats:s.chats,active:s.active,params:s.params,provider:{...s.provider,apiKey:s.provider.remember?s.provider.apiKey:""},agent:s.agent,training:s.training}))}catch{toast("Session storage is full; export chats to keep a copy.","error")}}
function log(message,source="ui"){s.appLogs.push(`[${new Date().toLocaleTimeString()}] [${source}] ${message}`);if(s.appLogs.length>400)s.appLogs.shift()}
function toast(text,kind=""){const node=document.querySelector("#toast");node.textContent=text;node.className="show "+kind;clearTimeout(toast.timer);toast.timer=setTimeout(()=>node.className="",2600)}
function newChat(open=true){const c={id:id(),title:"New chat",created:Date.now(),messages:[],draft:""};s.chats.unshift(c);s.active=c.id;if(open)s.view="chat";save();return c}
if(!s.chats.length)newChat(false);
if(!s.chats.some(c=>c.id===s.active))s.active=s.chats[0].id;
const chat=()=>s.chats.find(c=>c.id===s.active)||s.chats[0];

root.innerHTML=`<div class="app-shell"><aside class="rail"><button class="wordmark" data-view="workspace"><span>F</span><b>fyodor</b></button><nav>
<button data-view="workspace"><i>▦</i>Workspace</button><button data-view="chat"><i>◌</i>Chat</button><button data-view="playground"><i>⌁</i>Playground</button><button data-view="training"><i>△</i>Training</button><button data-view="api"><i>⌘</i>API access</button><button data-view="logs"><i>≡</i>Logs</button>
</nav><div class="rail-status"><i></i><span id="engine-state">starting</span></div></aside>
<aside class="history-panel"><div class="history-head"><b>Chats</b><button id="new-chat" aria-label="New conversation">+</button></div><label class="search">⌕<input id="chat-search" placeholder="Search chats"></label><div id="history"></div><div class="history-tools"><button id="import-chat">import</button><button id="export-chat">export</button><input type="file" id="import-file" accept=".json" hidden></div></aside>
<section class="stage"><header><button class="mobile-menu" aria-label="Open navigation">☰</button><div><b id="view-title"></b><span id="view-meta"></span></div><span class="provider-pill" id="provider-pill"></span><select id="global-model" aria-label="Active model"></select><button class="load-model">Load model</button><button class="settings-button" id="open-settings" aria-label="Settings">⚙</button></header><main id="workspace"></main><footer><span><i class="dot"></i><b id="footer-engine">backend</b></span><span id="footer-stat">0 models</span></footer></section></div><div class="scrim"></div><dialog id="settings"></dialog><div id="toast" role="status" aria-live="polite"></div>`;
const viewRoot=document.querySelector("#workspace");

function history(){const q=s.query.toLowerCase(),list=s.chats.filter(c=>!q||c.title.toLowerCase().includes(q));document.querySelector("#history").innerHTML=list.map(c=>`<button class="history-item ${c.id===s.active?"active":""}" data-chat="${c.id}"><span>${esc(c.title)}</span><small>${c.messages.length} messages</small></button>`).join("")||'<p class="no-results">No matching chats.</p>'}
function modelOptions(){if(s.provider.type==="openai")return `<option value="remote">${esc(s.provider.model||"Set endpoint model")}</option>`;return '<option value="">No model</option>'+s.models.filter(m=>m.generation_supported).map(m=>`<option value="${mid(m)}" ${mid(m)===s.selected?"selected":""}>${esc(mname(m))}</option>`).join("")}
function render(){
  document.body.dataset.view=s.view;
  document.querySelectorAll("[data-view]").forEach(n=>n.classList.toggle("active",n.dataset.view===s.view));
  document.querySelector("#view-title").textContent=meta[s.view][0];document.querySelector("#view-meta").textContent=meta[s.view][1];
  document.querySelector("#global-model").innerHTML=modelOptions();
  document.querySelector("#provider-pill").textContent=s.provider.type==="local"?"LOCAL":"OPENAI";
  document.querySelector(".stage>header .load-model").hidden=s.provider.type!=="local";
  history();({workspace,chat:chatView,playground,training:trainingView,logs:logsView,api:()=>viewRoot.innerHTML=apiMarkup(s)}[s.view]||workspace)();save();paintRanges();
}

function workspace(){viewRoot.innerHTML=workspaceMarkup(s);}

function messageMarkup(m,index){return `<article class="message ${m.role}"><i>${m.role==="assistant"?"F":m.role==="tool"?"⚒":"YOU"}</i><div><header><b>${m.role==="assistant"?"fyodor":m.role}</b><span>${m.tokens?m.tokens+" tok":""}</span></header><p>${esc(m.text)}</p><footer><button data-copy="${index}">□ copy</button>${m.role==="assistant"?`<button data-regen="${index}">↻ regenerate</button>`:""}</footer></div></article>`}
function chatView(){
  const c=chat(),messages=c.messages.map(messageMarkup).join("")||'<div class="chat-empty"><b>F</b><h2>Start typing.</h2><p>The selected provider will respond here.</p><div><button data-prompt="Explain this clearly: ">explain</button><button data-prompt="Help me write: ">write</button><button data-prompt="Review this code: ">review code</button></div></div>';
  viewRoot.innerHTML=`<div class="chat-view"><header class="chat-toolbar"><div><label>CHAT</label><input id="chat-title" value="${esc(c.title)}"></div><div class="toolbar-actions"><button id="agent-toggle" class="${s.agent.enabled?"active":""}">◎ agent ${s.agent.enabled?"on":"off"}</button><button id="clear-active">× clear</button><button id="download-active">↓ json</button></div></header><section class="message-list">${messages}<div id="typing"></div></section>
  <form class="composer"><div><textarea id="prompt" name="prompt" aria-label="Message" rows="2" placeholder="${s.agent.enabled?"Give the agent a goal…":"Message fyodor…"}" autofocus ${s.busy?"disabled":""}>${esc(c.draft||"")}</textarea><button class="send" aria-label="Send message" ${s.busy?"disabled":""}>↗</button></div><footer><button type="button" id="system-toggle">system prompt</button><span>${s.provider.type==="openai"?esc(s.provider.model||"configure model"):(s.selected?esc(mname(s.models.find(m=>mid(m)===s.selected)||{})):"select a model")} · enter to send</span></footer><textarea id="system-prompt" placeholder="System prompt…" ${s.params.system?"":"hidden"}>${esc(s.params.system)}</textarea></form></div>`;
  requestAnimationFrame(()=>{const n=viewRoot.querySelector(".message-list");n.scrollTop=n.scrollHeight});
}

function range(id,label,min,max,step,value=s.params[id]){const pct=(value-min)/(max-min)*100;return `<label class="range"><span>${label}<output id="${id}-out">${value}</output></span><input id="${id}" data-param="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" style="--pct:${pct}%"></label>`}
function playground(){viewRoot.innerHTML=`<div class="playground-view"><section class="prompt-lab"><header><div><label>PLAYGROUND</label><h1>Prompt lab</h1></div><button id="reset-params">reset</button></header><textarea id="lab-prompt" placeholder="Write a prompt to test against the selected model…">${esc(s.lastPrompt)}</textarea><div class="lab-actions"><span id="lab-count">${s.lastPrompt.length} chars</span><button id="run-lab" class="orange-action">run inference ↗</button></div><section id="lab-output"><p>Output appears here.</p></section></section><aside class="controls"><header><div><label>GENERATION</label><h2>Controls</h2></div></header>
  ${range("temperature","temperature",0,2,.01)}${range("top_p","top p",.01,1,.01)}${range("top_k","top k",0,200,1)}${range("max_tokens","max tokens",1,4096,1)}${s.provider.type==="openai"?range("frequency_penalty","frequency penalty",-2,2,.05)+range("presence_penalty","presence penalty",-2,2,.05):""}
  <label class="field">seed<input id="seed" data-param="seed" type="number" value="${esc(s.params.seed)}" placeholder="random"></label>${s.provider.type==="openai"?`<label class="field">stop sequences<input id="stop" data-param="stop" value="${esc(s.params.stop)}" placeholder="comma separated"></label>`:`<label class="field">Draft model<select data-param="draft_model_id"><option value="">No draft</option>${s.models.filter(m=>mid(m)!==s.selected&&(m.generation_supported||m.draft_supported)).map(m=>`<option value="${mid(m)}" ${String(mid(m))===String(s.params.draft_model_id)?"selected":""}>${esc(mname(m))}</option>`).join("")}</select></label>${range("speculative_tokens","draft window",1,32,1)}`}<label class="field">system prompt<textarea id="lab-system" data-param="system" rows="5">${esc(s.params.system)}</textarea></label><div class="presets"><label>PRESETS</label><button data-preset="precise">precise</button><button data-preset="balanced">balanced</button><button data-preset="creative">creative</button></div></aside></div>`}

function trainingInput(id,label,type="text",extra=""){return `<label class="field">${label}<div class="path-field"><input id="train-${id}" data-train="${id}" type="${type}" value="${esc(s.training[id])}" ${extra}><button type="button" data-pick="${id}">browse</button></div></label>`}
function trainingView(){
  const needsBase=s.training.mode!=="pretrain";
  viewRoot.innerHTML=`<div class="training-view"><section class="training-head orange-white"><div><label>NATIVE TRAINING</label><h1>Built, one step at a time.</h1><p>Pretrain, continue, supervise, or preference-tune with the bundled C trainer.</p></div><span class="train-state ${s.trainStatus.running?"running":""}">${s.trainStatus.running?"RUNNING":s.trainStatus.exitCode==null?"IDLE":"EXIT "+s.trainStatus.exitCode}</span></section>
  <form id="training-form"><section class="panel training-files"><header><div><label>RUN</label><h2>Inputs</h2></div></header><label class="field">mode<select id="train-mode" data-train="mode"><option value="pretrain">pretrain · random weights</option><option value="cpt">CPT · continued pretraining</option><option value="sft">SFT · supervised</option><option value="dpo">DPO · preference</option></select></label>
  ${trainingInput("data","dataset")}${needsBase?trainingInput("base","base model"):""}${trainingInput("output","output GGUF")}${trainingInput("checkpoint","checkpoint (optional)")}${trainingInput("resume","resume checkpoint (optional)")}
  <div class="run-actions"><button class="blue-action" id="start-training" type="submit" ${s.trainStatus.running?"disabled":""}>start training ↗</button><button class="danger-action" id="stop-training" type="button" ${s.trainStatus.running?"":"disabled"}>stop</button></div></section>
  <section class="panel training-config"><header><div><label>CONFIGURATION</label><h2>Optimizer + graph</h2></div></header><div class="config-grid">
  ${trainingNumber("steps","steps",1)}${trainingNumber("learningRate","learning rate",.000001,"step=\"0.0001\"")}${trainingNumber("rank","LoRA rank",0)}${trainingNumber("beta","DPO beta",.0001,"step=\"0.01\"")}${trainingNumber("context","context",1)}${trainingNumber("memoryMib","memory MiB",1)}
  ${trainingNumber("dimension","dimension",1)}${trainingNumber("feedForward","feed-forward",1)}${trainingNumber("layers","layers",1)}${trainingNumber("heads","heads",1)}${trainingNumber("kvHeads","KV heads",1)}${trainingNumber("seed","seed",0)}</div>
  <div class="training-note"><b>Native limitations</b><p>CPU backward. Dense LLaMA and Gemma 4 imports. MoE, MTP training, encoders, mixed precision, and distributed runs are not implemented.</p></div></section></form></div>`;
  document.querySelector("#train-mode").value=s.training.mode;
}
function trainingNumber(id,label,min,extra=""){return `<label>${label}<input data-train="${id}" type="number" min="${min}" value="${esc(s.training[id])}" ${extra}></label>`}

function logsView(){
  const lines=[...s.appLogs,...s.logs],filter=(s.logFilter||"all");
  const shown=filter==="all"?lines:lines.filter(line=>line.includes(`[${filter}]`));
  viewRoot.innerHTML=`<div class="logs-view"><header><div><label>LOGS</label><h1>Runtime stream</h1></div><div><select id="log-filter"><option value="all">all sources</option><option value="engine">engine</option><option value="trainer">trainer</option><option value="ui">ui</option></select><button id="refresh-logs">refresh</button><button id="copy-logs">copy</button><button id="clear-logs">clear</button></div></header><pre id="log-output">${esc(shown.join("\n")||"No log entries.")}</pre></div>`;
  document.querySelector("#log-filter").value=filter;requestAnimationFrame(()=>{const n=document.querySelector("#log-output");n.scrollTop=n.scrollHeight});
}

function settingsView(){
  const d=document.querySelector("#settings");d.innerHTML=`<form method="dialog" class="settings-card"><header><div><label>CONFIGURATION</label><h2>Provider + agent</h2></div><button value="close">×</button></header><div class="settings-grid"><label class="field">provider<select id="provider-type"><option value="local">local C runtime</option><option value="openai">OpenAI-compatible endpoint</option></select></label><label class="field remote-setting">base URL<input id="provider-endpoint" value="${esc(s.provider.endpoint)}" placeholder="https://api.openai.com/v1"></label><label class="field remote-setting">API key<input id="provider-key" type="password" value="${esc(s.provider.apiKey)}" autocomplete="off"></label><label class="field remote-setting">model ID<div class="path-field"><input id="provider-model" value="${esc(s.provider.model)}" placeholder="gpt-5"><button type="button" id="test-provider">test</button></div></label><label class="check"><input id="remember-key" type="checkbox" ${s.provider.remember?"checked":""}> remember API key on this device</label><hr><label class="check"><input id="agent-enabled" type="checkbox" ${s.agent.enabled?"checked":""}> enable Agent mode</label><label class="field">maximum agent turns<input id="agent-turns" type="number" min="1" max="24" value="${s.agent.maxTurns}"></label><p class="settings-note">Agent tools are intentionally read-only: runtime status, loaded models, and local time.</p></div><footer><button value="close">cancel</button><button type="button" id="save-settings" class="blue-action">save configuration</button></footer></form>`;d.querySelector("#provider-type").value=s.provider.type;toggleRemoteSettings();d.showModal();
}
function toggleRemoteSettings(){const remote=document.querySelector("#provider-type")?.value==="openai";document.querySelectorAll(".remote-setting").forEach(n=>n.hidden=!remote)}

function paintRanges(){document.querySelectorAll('input[type="range"]').forEach(n=>{n.style.setProperty("--pct",(n.value-n.min)/(n.max-n.min)*100+"%")})}
// Runtime data is read from the engine, including compiled capabilities and
// the actual provider attached to each model. Never label an offline model CPU.
async function refresh(){
  try {
    const [h,m,r,c]=await Promise.all([backend.health(),backend.models(),backend.runtime(),getBackendConnection()]);
    s.health=h;s.models=m.models||[];s.runtime=r;s.baseUrl=c.baseUrl;
    if(!s.models.some(m=>mid(m)===s.selected&&m.generation_supported))s.selected=s.models.find(m=>m.generation_supported)?.id||null;
    document.body.classList.add("connected");document.querySelector("#engine-state").textContent="engine online";
    document.querySelector("#footer-engine").textContent="C runtime";document.querySelector("#footer-stat").textContent=s.models.length+" models";
  } catch(error) {
    s.health=null;s.runtime=null;s.models=[];s.selected=null;document.body.classList.remove("connected");
    document.querySelector("#engine-state").textContent="engine offline";document.querySelector("#footer-engine").textContent="disconnected";log(error.message,"engine");
  }
  render();
}
async function load(){
  if(s.actionBusy||s.busy)return;
  const path=window.__TAURI_INTERNALS__?await chooseModel():window.prompt("Local model path (for example D:/Models/model.gguf)");
  if(!path)return;
  s.actionBusy=true;
  try {const response=await backend.loadModel(path);if(response.model?.generation_supported)s.selected=response.model.id;log(`loaded ${path}`,"engine");}
  catch(error){toast(error.message,"error");}
  finally{s.actionBusy=false;await refresh();}
}
async function setCompute(modelId,compute){
  if(s.actionBusy||s.busy)return;s.actionBusy=true;render();
  try{await backend.setCompute(modelId,compute);log(`model ${modelId}: ${compute.toUpperCase()}`,"engine");toast(`${compute.toUpperCase()} active`,"success");}
  catch(error){toast(error.message,"error");}
  finally{s.actionBusy=false;await refresh();}
}

function download(data){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));a.download=(data.title||"fyodor-chat").replace(/\W+/g,"-").toLowerCase()+".json";a.click();URL.revokeObjectURL(a.href)}

async function externalCompletion(messages,withTools=false){
  if(!s.provider.model)throw new Error("Configure an endpoint model first");
  const body={model:s.provider.model,messages,max_tokens:+s.params.max_tokens,temperature:+s.params.temperature,top_p:+s.params.top_p,stream:false};
  if(+s.params.frequency_penalty)body.frequency_penalty=+s.params.frequency_penalty;
  if(+s.params.presence_penalty)body.presence_penalty=+s.params.presence_penalty;
  if(s.params.stop.trim())body.stop=s.params.stop.split(",").map(x=>x.trim()).filter(Boolean);
  if(withTools)body.tools=agentTools;
  return backend.openaiChat(s.provider,body);
}
const agentTools=[
  {type:"function",function:{name:"runtime_status",description:"Read the current Fyodor runtime status",parameters:{type:"object",properties:{}}}},
  {type:"function",function:{name:"list_loaded_models",description:"List models loaded into the local C runtime",parameters:{type:"object",properties:{}}}},
  {type:"function",function:{name:"local_time",description:"Read the user's current local date and time",parameters:{type:"object",properties:{}}}},
];
async function runTool(name){if(name==="runtime_status")return JSON.stringify(s.health||{online:false});if(name==="list_loaded_models")return JSON.stringify(s.models.map(m=>({id:mid(m),name:mname(m),architecture:m.architecture})));if(name==="local_time")return new Date().toString();return JSON.stringify({error:"Unknown tool"})}
async function agentCompletion(messages){
  if(s.provider.type!=="openai"){const system=(s.params.system?s.params.system+"\n":"")+"You are in Agent mode. Work through the goal carefully and return the completed result.";return localCompletion(messages.at(-1).content,system)}
  const work=[...messages];
  for(let turn=0;turn<s.agent.maxTurns;turn++){
    const result=await externalCompletion(work,true),choice=result.choices?.[0]?.message;if(!choice)throw new Error("Endpoint returned no assistant message");
    work.push(choice);if(!choice.tool_calls?.length)return{text:choice.content||"",tokens:result.usage?.completion_tokens||0};
    for(const call of choice.tool_calls){const output=await runTool(call.function?.name);log(`${call.function?.name} → ${output}`,"agent");work.push({role:"tool",tool_call_id:call.id,content:output})}
  }
  throw new Error("Agent reached its turn limit");
}
async function localCompletion(prompt,system=s.params.system){
  if(!s.models.find(m=>mid(m)===s.selected)?.generation_supported)throw new Error("Select a local text model first");
  return backend.generate({model_id:s.selected,prompt,...generationOptions(s.params),...(s.params.draft_model_id?{draft_model_id:Number(s.params.draft_model_id),speculative_tokens:Number(s.params.speculative_tokens)}:{})});
}
async function complete(messages,raw=false){
  if(s.agent.enabled&&s.provider.type==="openai")return agentCompletion(messages);
  if(s.provider.type==="openai"){const result=await externalCompletion(messages),choice=result.choices?.[0]?.message;if(!choice)throw new Error("Endpoint returned no assistant message");return{text:choice.content||"",tokens:result.usage?.completion_tokens||0};}
  return localCompletion(raw?messages.at(-1).content:formatConversation(messages,s.models.find(m=>mid(m)===s.selected)?.architecture));
}

async function generate(prompt,source="chat"){
  if(s.busy||s.actionBusy)return;s.busy=true;s.lastPrompt=prompt;let target;
  if(source==="chat"){target=chat();target.draft="";target.messages.push({role:"user",text:prompt});if(target.title==="New chat")target.title=prompt.slice(0,42);render();document.querySelector("#typing").innerHTML=`<p class="thinking">${s.agent.enabled?"agent is working...":"fyodor is generating..."}</p>`}
  else document.querySelector("#lab-output").innerHTML='<p class="thinking">running inference...</p>';
  try{const started=performance.now(),messages=source==="chat"?[{role:"system",content:s.params.system||"You are Fyodor."},...target.messages.map(m=>({role:m.role==="tool"?"user":m.role,content:m.text}))]:[{role:"system",content:s.params.system||"You are Fyodor."},{role:"user",content:prompt}];const result=await complete(messages,source!=="chat"),answer={role:"assistant",text:result.text||"",tokens:result.generated_tokens||result.tokens||0,seconds:(performance.now()-started)/1000,seed:result.seed};
    if(source==="chat"){target.messages.push(answer);render()}else{const n=document.querySelector("#lab-output");n.dataset.text=answer.text;n.innerHTML=`<header><span>${answer.tokens} tokens · ${answer.seconds.toFixed(2)}s · seed ${answer.seed??"—"}</span><button data-copy-lab>□ copy</button></header><pre>${esc(answer.text)}</pre>`}log(`completion finished in ${answer.seconds.toFixed(2)}s`,"ui");save()}
  catch(error){toast(error.message,"error");log(error.message,"ui");if(source==="chat"){target.messages.pop();target.draft=prompt;render()}}finally{s.busy=false;if(source==="chat")render()}
}

async function refreshTraining(){if(!window.__TAURI_INTERNALS__)return;try{const previous=JSON.stringify(s.trainStatus);s.trainStatus=await invokeDesktop("training_status");if(s.view==="training"&&previous!==JSON.stringify(s.trainStatus))trainingView();if(s.view==="logs")await refreshLogs()}catch(error){log(error.message,"trainer")}}
async function refreshLogs(){if(window.__TAURI_INTERNALS__)try{s.logs=await invokeDesktop("runtime_logs")}catch(error){log(error.message,"ui")}if(s.view==="logs")logsView()}
function collectTraining(){document.querySelectorAll("[data-train]").forEach(n=>{s.training[n.dataset.train]=n.type==="number"?+n.value:n.value});save()}

document.addEventListener("click",async event=>{
  if(event.target.closest("[data-refresh]")){await refresh();return;}
  const compute=event.target.closest("[data-compute]");if(compute&&!compute.disabled){await setCompute(+compute.dataset.id,compute.dataset.compute);return;}
  const url=event.target.closest("[data-copy-url]");if(url){await navigator.clipboard.writeText(s.baseUrl+url.dataset.copyUrl);toast("URL copied");return;}
  if(event.target.closest("[data-copy-token]")){const c=await getBackendConnection();if(c.token){await navigator.clipboard.writeText(c.token);toast("Local token copied");}return;}
  const view=event.target.closest("[data-view]")?.dataset.view;if(view){s.view=view;document.body.classList.remove("menu-open");render();if(view==="logs")refreshLogs();return}
  const selectedChat=event.target.closest("[data-chat]")?.dataset.chat;if(selectedChat){s.active=selectedChat;s.view="chat";render();return}
  if(event.target.closest("#new-chat,[data-new-chat]")){newChat();render()}
  if(event.target.closest(".load-model"))load();
  const model=event.target.closest("[data-model]");if(model){s.selected=+model.dataset.model;render()}
  const unload=event.target.closest("[data-unload]");if(unload)try{await backend.unloadModel(+unload.dataset.unload);await refresh()}catch(error){toast(error.message,"error")}
  if(event.target.closest("#clear-chats")&&!s.busy){s.chats=[];newChat(false);render()}
  if(event.target.closest("#clear-active")&&!s.busy){chat().messages=[];render()}
  if(event.target.closest("#download-active,#export-chat"))download(chat());
  if(event.target.closest("#import-chat"))document.querySelector("#import-file").click();
  const prompt=event.target.closest("[data-prompt]")?.dataset.prompt;if(prompt){document.querySelector("#prompt").value=prompt;document.querySelector("#prompt").focus()}
  const copy=event.target.closest("[data-copy]");if(copy){await navigator.clipboard.writeText(chat().messages[+copy.dataset.copy].text);toast("copied","success")}
  const regenerate=event.target.closest("[data-regen]");if(regenerate&&!s.busy){const i=+regenerate.dataset.regen,p=chat().messages.slice(0,i).reverse().find(x=>x.role==="user");chat().messages.splice(i-1);if(p)generate(p.text)}
  if(event.target.closest("#system-toggle"))document.querySelector("#system-prompt").toggleAttribute("hidden");
  if(event.target.closest("#agent-toggle")){s.agent.enabled=!s.agent.enabled;render()}
  if(event.target.closest("#reset-params")){s.params={...defaults};render()}
  const preset=event.target.closest("[data-preset]")?.dataset.preset;if(preset){Object.assign(s.params,{precise:{temperature:.2,top_p:.8,top_k:20},balanced:{temperature:.8,top_p:.95,top_k:40},creative:{temperature:1.25,top_p:1,top_k:80}}[preset]);render()}
  if(event.target.closest("#run-lab")){const p=document.querySelector("#lab-prompt").value.trim();if(p)generate(p,"lab")}
  if(event.target.closest("[data-copy-lab]"))navigator.clipboard.writeText(document.querySelector("#lab-output").dataset.text||"");
  if(event.target.closest(".mobile-menu"))document.body.classList.toggle("menu-open");if(event.target.closest(".scrim"))document.body.classList.remove("menu-open");
  if(event.target.closest("#open-settings"))settingsView();if(event.target.closest("#provider-type"))toggleRemoteSettings();
  if(event.target.closest("#save-settings")){s.provider.type=document.querySelector("#provider-type").value;s.provider.endpoint=document.querySelector("#provider-endpoint").value.trim();s.provider.apiKey=document.querySelector("#provider-key").value.trim();s.provider.model=document.querySelector("#provider-model").value.trim();s.provider.remember=document.querySelector("#remember-key").checked;s.agent.enabled=document.querySelector("#agent-enabled").checked;s.agent.maxTurns=Math.max(1,Math.min(24,+document.querySelector("#agent-turns").value||6));save();document.querySelector("#settings").close();render()}
  if(event.target.closest("#test-provider"))try{const p={type:"openai",endpoint:document.querySelector("#provider-endpoint").value,apiKey:document.querySelector("#provider-key").value};const result=await backend.openaiModels(p);toast(`connected · ${result.data?.length||0} models`,"success")}catch(error){toast(error.message,"error")}
  const pick=event.target.closest("[data-pick]")?.dataset.pick;if(pick){let path;if(pick==="data")path=await chooseDataset();else if(pick==="output")path=await chooseOutput();else if(pick==="checkpoint"||pick==="resume")path=await chooseCheckpoint(pick==="checkpoint");else path=await chooseModel();if(path){s.training[pick]=path;trainingView()}}
  if(event.target.closest("#stop-training"))try{await invokeDesktop("stop_training");await refreshTraining()}catch(error){toast(error.message,"error")}
  if(event.target.closest("#refresh-logs"))refreshLogs();if(event.target.closest("#copy-logs")){await navigator.clipboard.writeText([...s.appLogs,...s.logs].join("\n"));toast("logs copied","success")}
  if(event.target.closest("#clear-logs")){s.appLogs=[];if(window.__TAURI_INTERNALS__)await invokeDesktop("clear_runtime_logs");s.logs=[];logsView()}
});
document.addEventListener("input",event=>{
  if(event.target.id==="prompt"){chat().draft=event.target.value;save()}
  if(event.target.id==="chat-search"){s.query=event.target.value;history()}
  if(event.target.id==="lab-prompt"){s.lastPrompt=event.target.value;document.querySelector("#lab-count").textContent=event.target.value.length+" chars";}
  if(event.target.matches("[data-param]")){s.params[event.target.dataset.param]=event.target.type==="range"?+event.target.value:event.target.value;const out=document.querySelector("#"+event.target.dataset.param+"-out");if(out)out.value=event.target.value;paintRanges();save()}
  if(event.target.matches("[data-train]")){collectTraining();if(event.target.id==="train-mode")trainingView()}
  if(event.target.id==="log-filter"){s.logFilter=event.target.value;logsView()}
  if(event.target.id==="provider-type")toggleRemoteSettings();
});
document.addEventListener("change",event=>{
  if(event.target.id==="global-model"&&s.provider.type==="local"){s.selected=+event.target.value||null;render()}
  if(event.target.id==="chat-title"){chat().title=event.target.value.trim()||"New chat";render()}
  if(event.target.id==="system-prompt"){s.params.system=event.target.value;save()}
  if(event.target.id==="import-file"&&event.target.files[0]){if(event.target.files[0].size>4*1024*1024){toast("Chat imports are limited to 4 MiB.","error");return;}const reader=new FileReader();reader.onload=()=>{try{const c=normalizeChat(JSON.parse(reader.result),id());s.chats.unshift(c);s.active=c.id;s.view="chat";render()}catch{toast("invalid chat file","error")}};reader.readAsText(event.target.files[0])}
});
document.addEventListener("keydown",event=>{if(event.target.id==="prompt"&&event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();event.target.form.requestSubmit()}});
document.addEventListener("submit",async event=>{
  if(event.target.matches(".composer")){event.preventDefault();const p=event.target.prompt.value.trim();if(p)generate(p);return}
  if(event.target.id==="training-form"){event.preventDefault();collectTraining();try{await invokeDesktop("start_training",{args:{...s.training,...(s.training.mode==="pretrain"?{rank:0,base:""}:{})}});log("training run launched","trainer");toast("training started","success");await refreshTraining()}catch(error){toast(error.message,"error");log(error.message,"trainer")}}
});

render();refresh();setInterval(()=>{if(s.view==="training")refreshTraining();if(s.view==="logs")refreshLogs()},1500);
