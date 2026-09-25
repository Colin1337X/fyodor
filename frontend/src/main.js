import"./styles.css";
import {icon, navigationMarkup} from './icons.js';
import {messageBody} from './message.js';
import {benchmarkMarkup, parseBenchmark} from './benchmark.js';
import {parseTrainingMetric} from './training-metrics.js';
import{backend}from"./api.js";
import {formatConversation, normalizeChat, generationOptions} from "./session.js";
import {workspaceMarkup, apiMarkup} from "./workspace.js";
import{chooseCheckpoint,chooseDataset,chooseModel,chooseOutput,getBackendConnection,invokeDesktop}from"./platform.js";

// Storage and imported chats are untrusted. A corrupt value must not blank the app.
let persisted={};
try { persisted=JSON.parse(localStorage.getItem("fyodor-state")||"{}")||{}; } catch {}
if(typeof persisted!=="object"||Array.isArray(persisted))persisted={};
if(!["workspace","models","chat","playground","training","logs","api","benchmark"].includes(persisted.view))persisted.view="workspace";
const restored=[];
for(const entry of Array.isArray(persisted.chats)?persisted.chats.slice(0,300):[]) {
  try { const c=normalizeChat(entry,crypto.randomUUID());if(entry.id===persisted.active)persisted.active=c.id;restored.push(c); } catch {}
}
persisted.chats=restored;
const defaults={temperature:.8,top_p:.95,top_k:40,max_tokens:256,seed:"",system:"",frequency_penalty:0,presence_penalty:0,stop:"",draft_model_id:"",speculative_tokens:4};
const trainingDefaults={mode:"pretrain",data:"",output:"",base:"",checkpoint:"",resume:"",steps:100,accumulate:1,threads:0,learningRate:.001,rank:0,beta:.1,context:256,dimension:64,feedForward:128,layers:2,heads:4,kvHeads:2,seed:42,memoryMib:256};
// Restore only known primitive settings. Imported storage cannot introduce
// arbitrary attribute strings where a numeric control is expected.
function restoreSettings(defaults, value){
  const result={...defaults};
  if(!value||typeof value!=="object"||Array.isArray(value))return result;
  for(const [key,base] of Object.entries(defaults)){
    const v=value[key];
    if(typeof base==="number"){
      const n=typeof v==="string"&&v.trim()!==""?Number(v):v;
      if(typeof n==="number"&&Number.isFinite(n))result[key]=n;
    }else if(typeof v===typeof base)result[key]=v;
  }
  return result;
}
const s={
  view:persisted.view||"workspace",health:null,runtime:null,baseUrl:"",actionBusy:false,models:[],selected:persisted.selected||null,
  chats:persisted.chats||[],active:persisted.active||null,params:restoreSettings(defaults,persisted.params),
  provider:restoreSettings({type:"local",endpoint:"https://api.openai.com/v1",apiKey:"",model:"",remember:false,stream:true},persisted.provider),
  agent:restoreSettings({enabled:false,maxTurns:6},persisted.agent),training:restoreSettings(trainingDefaults,persisted.training),
  trainStatus:{running:false,exitCode:null},logs:[],appLogs:[],busy:false,query:"",lastPrompt:"",labAnswer:null,
};
if(!["local","openai"].includes(s.provider.type))s.provider.type="local";
s.agent.maxTurns=Math.max(1,Math.min(24,Math.trunc(s.agent.maxTurns)));
// Window-only state never enters session storage or backend payloads.
const ui={benchmarks:[],logPaused:false,error:'',messageLimit:80};
const meta={models:["Models","local registry and compute"],benchmark:["Benchmarks","measured runtime results"],api:["API access","local REST interfaces"],workspace:["Overview","models and activity"],chat:["Chat","conversation"],playground:["Playground","generation controls"],training:["Training","native C trainer"],logs:["Logs","engine and trainer"]};
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const id=()=>crypto.randomUUID?.()||Date.now()+Math.random()+"";
const mid=m=>m.id??m.model_id;
const mname=m=>m.name||m.path?.split(/[\\/]/).pop()||`model ${mid(m)}`;
const root=document.querySelector("#app");
function save(){try{localStorage.setItem("fyodor-state",JSON.stringify({view:s.view,selected:s.selected,chats:s.chats,active:s.active,params:s.params,provider:{...s.provider,apiKey:s.provider.remember?s.provider.apiKey:""},agent:s.agent,training:s.training}))}catch{toast("Session storage is full; export chats to keep a copy.","error")}}
function log(message,source="ui"){s.appLogs.push(`[${new Date().toLocaleTimeString()}] [${source}] ${message}`);if(s.appLogs.length>400)s.appLogs.shift()}
function toast(text,kind=""){const node=document.querySelector("#toast");if(!node)return;node.textContent=text;node.className="show "+kind;clearTimeout(toast.timer);toast.timer=setTimeout(()=>node.className="",2600)}
function newChat(open=true){const c={id:id(),title:"New chat",created:Date.now(),messages:[],draft:""};s.chats.unshift(c);s.active=c.id;if(open)s.view="chat";save();return c}
if(!s.chats.length)newChat(false);
if(!s.chats.some(c=>c.id===s.active))s.active=s.chats[0].id;
const chat=()=>s.chats.find(c=>c.id===s.active)||s.chats[0];

root.innerHTML=`<div class="app-shell"><aside class="rail"><button class="wordmark" data-view="workspace" aria-label="Fyodor overview"><span>F</span><b>fyodor</b></button><nav aria-label="Workspaces">${navigationMarkup()}</nav><div class="rail-bottom"><button id="collapse-sidebar" aria-label="Toggle sidebar" title="Toggle sidebar">${icon('sidebar')}<span>Collapse sidebar</span></button><div class="rail-status"><i></i><span id="engine-state">starting</span></div></div></aside>
<aside class="history-panel"><div class="history-head"><b>Chats</b><button id="new-chat" aria-label="New conversation">+</button></div><label class="search">⌕<input id="chat-search" placeholder="Search chats"></label><div id="history"></div><div class="history-tools"><button id="import-chat">import</button><button id="export-chat">export</button><input type="file" id="import-file" accept=".json" hidden></div></aside>
<section class="stage"><header><button class="mobile-menu" aria-label="Open navigation">${icon('menu')}</button><div><b id="view-title"></b><span id="view-meta"></span></div><span class="provider-pill" id="provider-pill"></span><select id="global-model" aria-label="Active model"></select><button class="load-model">Load model</button><button class="settings-button" id="open-settings" aria-label="Settings">${icon('settings')}</button></header><main id="workspace" aria-label="Workspace"></main><footer><span><i class="dot"></i><b id="footer-engine">backend</b></span><span id="footer-stat">0 models</span></footer></section></div><div class="scrim"></div><dialog id="settings" aria-label="Settings"></dialog><div id="toast" role="status" aria-live="polite"></div>`;
const viewRoot=document.querySelector("#workspace");
const narrowNavigation=matchMedia('(max-width:940px)');
function syncNavigation(){
  const open=narrowNavigation.matches&&document.body.classList.contains('menu-open');
  document.querySelector('.rail').inert=narrowNavigation.matches&&!open;
  document.querySelector('.history-panel').inert=narrowNavigation.matches&&!open;
  document.querySelector('.stage').inert=open;
  document.querySelector('.mobile-menu').setAttribute('aria-expanded',String(open));
}
new MutationObserver(syncNavigation).observe(document.body,{attributes:true,attributeFilter:['class']});
narrowNavigation.addEventListener('change',syncNavigation);syncNavigation();

function history(){const q=s.query.toLowerCase(),list=s.chats.filter(c=>!q||c.title.toLowerCase().includes(q));document.querySelector("#history").innerHTML=list.map(c=>`<button class="history-item ${c.id===s.active?"active":""}" data-chat="${c.id}"><span>${esc(c.title)}</span><small>${c.messages.length} messages</small></button>`).join("")||'<p class="no-results">No matching chats.</p>'}
function modelOptions(){if(s.provider.type==="openai")return `<option value="remote">${esc(s.provider.model||"Set endpoint model")}</option>`;return '<option value="">No model</option>'+s.models.filter(m=>m.generation_supported).map(m=>`<option value="${mid(m)}" ${mid(m)===s.selected?"selected":""}>${esc(mname(m))}</option>`).join("")}
function render(){
  document.body.dataset.view=s.view;
  document.querySelectorAll("[data-view]").forEach(n=>n.classList.toggle("active",n.dataset.view===s.view));
  document.querySelector("#view-title").textContent=meta[s.view][0];document.querySelector("#view-meta").textContent=meta[s.view][1];
  document.querySelector("#global-model").innerHTML=modelOptions();
  document.querySelector("#provider-pill").textContent=s.provider.type==="local"?"LOCAL":"OPENAI";
  document.querySelector(".stage>header .load-model").hidden=s.provider.type!=="local";
  document.querySelectorAll('.rail [data-view]').forEach(n=>n.setAttribute('aria-current',n.dataset.view===s.view?'page':'false'));
  const activeRuntime=s.runtime?.models?.find(m=>m.model_id===s.selected);
  document.querySelector('#footer-stat').textContent=`${s.models.length} models · ${activeRuntime?.compute?.toUpperCase()||'no active compute'}${s.busy?' · generating':s.actionBusy?' · loading':s.trainStatus.running?' · training':''}`;
  document.querySelector('#global-model').disabled=s.busy||s.actionBusy;
  document.querySelectorAll('.stage>header .load-model').forEach(n=>n.disabled=s.actionBusy||s.busy);
  history();({workspace,models:()=>{viewRoot.innerHTML=workspaceMarkup(s,true)},benchmark:()=>{viewRoot.innerHTML=benchmarkMarkup(ui.benchmarks)},chat:chatView,playground,training:trainingView,logs:logsView,api:()=>viewRoot.innerHTML=apiMarkup(s)}[s.view]||workspace)();save();paintRanges();
}

function workspace(){viewRoot.innerHTML=workspaceMarkup(s);}

function messageMarkup(m,index){return `<article class="message ${m.role}"><i>${m.role==="assistant"?"F":m.role==="tool"?"T":"YOU"}</i><div><header><b>${m.role==="assistant"?"Fyodor":m.role}</b><span>${m.tokens?m.tokens+" tokens":""}${m.seconds?' · '+m.seconds.toFixed(2)+'s':''}</span></header>${messageBody(m.text)}<footer><button data-copy="${index}">Copy</button>${m.role==="assistant"?`<button data-regen="${index}" ${s.busy?'disabled':''}>Regenerate</button>`:`<button data-edit="${index}" ${s.busy?'disabled':''}>Edit & resend</button>`}</footer></div></article>`}
function chatView(){
  const c=chat(),offset=Math.max(0,c.messages.length-ui.messageLimit),messages=(offset?'<button id="older-messages">Show earlier messages</button>':'')+c.messages.slice(offset).map((m,i)=>messageMarkup(m,i+offset)).join("")||'<div class="chat-empty"><b>F</b><h2>Start a conversation</h2><p>Choose a model above, then send a message.</p><div><button data-prompt="Explain this clearly: ">Explain</button><button data-prompt="Help me write: ">Write</button><button data-prompt="Review this code: ">Review code</button></div></div>';
  viewRoot.innerHTML=`<div class="chat-view"><header class="chat-toolbar"><div><label>CHAT</label><input id="chat-title" value="${esc(c.title)}"></div><div class="toolbar-actions"><button id="agent-toggle" class="${s.agent.enabled?"active":""}">◎ agent ${s.agent.enabled?"on":"off"}</button><button id="clear-active">× clear</button><button id="download-active">↓ json</button></div></header><section class="message-list">${messages}<div id="typing"></div></section>
  <form class="composer"><div><textarea id="prompt" name="prompt" aria-label="Message" rows="2" placeholder="${s.agent.enabled?"Give the agent a goal…":"Message fyodor…"}" autofocus ${s.busy?"disabled":""}>${esc(c.draft||"")}</textarea><button class="send" aria-label="Send message" ${s.busy?"disabled":""}>↗</button></div><footer><button type="button" id="system-toggle">system prompt</button><span>${s.provider.type==="openai"?esc(s.provider.model||"configure model"):(s.selected?esc(mname(s.models.find(m=>mid(m)===s.selected)||{})):"select a model")} · enter to send</span></footer><textarea id="system-prompt" placeholder="System prompt…" ${s.params.system?"":"hidden"}>${esc(s.params.system)}</textarea></form></div>`;
  if(ui.error){const n=document.createElement('div');n.className='error-banner';n.textContent=ui.error;viewRoot.querySelector('.message-list').append(n)}
  const composer=viewRoot.querySelector('.composer');
  composer.insertAdjacentHTML('beforeend',`<details class="generation-controls"><summary>Generation settings</summary><div class="config-grid">${[['temperature','Temperature',0,5,.01],['top_p','Top P',.01,1,.01],['top_k','Top K',0,1000000,1],['max_tokens','Max tokens',1,4096,1]].map(([key,label,min,max,step])=>`<label>${label}<input data-param="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${esc(s.params[key])}"></label>`).join('')}</div></details>`);
  if(s.busy&&ui.request?.chatId===c.id){viewRoot.querySelector('#typing').textContent=ui.request.partial||'Generating response…';if(ui.request.controller)composer.querySelector('footer').insertAdjacentHTML('afterbegin','<button type="button" id="stop-generation">Stop</button>');}
  requestAnimationFrame(()=>{const n=viewRoot.querySelector(".message-list");if(n)n.scrollTop=n.scrollHeight});
}

function range(id,label,min,max,step,value=s.params[id]){const pct=(value-min)/(max-min)*100;return `<label class="range"><span>${label}<output id="${id}-out">${value}</output></span><input id="${id}" data-param="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" style="--pct:${pct}%"></label>`}
function labOutput(){if(s.busy&&ui.request&&!ui.request.chatId)return `<p class="thinking">${esc(ui.request.partial||'Generating response…')}</p>${ui.request.controller?'<button id="stop-generation">Stop</button>':''}`;const a=s.labAnswer;return a?`<header><span>${esc(a.tokens)} tokens · ${a.seconds.toFixed(2)}s · seed ${esc(a.seed??"—")}</span><button data-copy-lab>Copy</button></header><pre>${esc(a.text)}</pre>`:`<p>${esc(ui.error||'Output appears here.')}</p>`}
function playground(){viewRoot.innerHTML=`<div class="playground-view"><section class="prompt-lab"><header><div><label>PLAYGROUND</label><h1>Prompt lab</h1></div><button id="reset-params">reset</button></header><textarea id="lab-prompt" placeholder="Write a prompt to test against the selected model…">${esc(s.lastPrompt)}</textarea><div class="lab-actions"><span id="lab-count">${s.lastPrompt.length} chars</span><button id="run-lab" class="orange-action">run inference ↗</button></div><section id="lab-output">${labOutput()}</section></section><aside class="controls"><header><div><label>GENERATION</label><h2>Controls</h2></div></header>
  ${range("temperature","temperature",0,2,.01)}${range("top_p","top p",.01,1,.01)}${range("top_k","top k",0,200,1)}${range("max_tokens","max tokens",1,4096,1)}${s.provider.type==="openai"?range("frequency_penalty","frequency penalty",-2,2,.05)+range("presence_penalty","presence penalty",-2,2,.05):""}
  <label class="field">seed<input id="seed" data-param="seed" type="number" value="${esc(s.params.seed)}" placeholder="random"></label>${s.provider.type==="openai"?`<label class="field">stop sequences<input id="stop" data-param="stop" value="${esc(s.params.stop)}" placeholder="comma separated"></label>`:`<label class="field">Draft model<select data-param="draft_model_id"><option value="">No draft</option>${s.models.filter(m=>mid(m)!==s.selected&&(m.generation_supported||m.draft_supported)).map(m=>`<option value="${mid(m)}" ${String(mid(m))===String(s.params.draft_model_id)?"selected":""}>${esc(mname(m))}</option>`).join("")}</select></label>${range("speculative_tokens","draft window",1,32,1)}`}<label class="field">system prompt<textarea id="lab-system" data-param="system" rows="5">${esc(s.params.system)}</textarea></label><div class="presets"><label>PRESETS</label><button data-preset="precise">precise</button><button data-preset="balanced">balanced</button><button data-preset="creative">creative</button></div></aside></div>`}

function trainingInput(id,label,type="text",extra=""){return `<label class="field">${label}<div class="path-field"><input id="train-${id}" data-train="${id}" type="${type}" value="${esc(s.training[id])}" ${extra}><button type="button" data-pick="${id}">browse</button></div></label>`}
function trainingView(){
  const needsBase=s.training.mode!=="pretrain";
  viewRoot.innerHTML=`<div class="training-view"><section class="training-head orange-white"><div><label>NATIVE TRAINING</label><h1>Training workspace</h1><p>Pretrain, continue, supervise, or preference-tune with the bundled C trainer.</p></div><span class="train-state ${s.trainStatus.running?"running":""}">${s.trainStatus.running?(s.trainStatus.stopping?"SAVING / STOPPING":"RUNNING"):s.trainStatus.exitCode==null?"IDLE":"EXIT "+s.trainStatus.exitCode}</span></section>
  <form id="training-form"><section class="panel training-files"><header><div><label>RUN</label><h2>Inputs</h2></div></header><label class="field">mode<select id="train-mode" data-train="mode"><option value="pretrain">pretrain · random weights</option><option value="cpt">CPT · continued pretraining</option><option value="sft">SFT · supervised</option><option value="dpo">DPO · preference</option></select></label>
  ${trainingInput("data","dataset")}${needsBase?trainingInput("base","base model"):""}${trainingInput("output","output GGUF")}${trainingInput("checkpoint","checkpoint (optional)")}${trainingInput("resume","resume checkpoint (optional)")}
  <div class="run-actions"><button class="blue-action" id="start-training" type="submit" ${s.trainStatus.running?"disabled":""}>start training ↗</button><button class="danger-action" id="stop-training" type="button" ${s.trainStatus.running&&!s.trainStatus.stopping?"":"disabled"}>stop &amp; save</button></div><p>Stop finishes the current update and saves the output model and optional checkpoint. Closing the app waits for saving.</p></section>
  <section class="panel training-config"><header><div><label>CONFIGURATION</label><h2>Optimizer + graph</h2></div></header><div class="config-grid">
  ${trainingNumber("steps","optimizer updates",1)}${trainingNumber("threads","CPU threads (0 = auto)",0,'max="64" step="1"')}${trainingNumber("accumulate","sequences / pairs per update",1,'max="1024" step="1"')}${trainingNumber("learningRate","learning rate",.000001,"step=\"0.0001\"")}${trainingNumber("rank","LoRA rank",0)}${trainingNumber("beta","DPO beta",.0001,"step=\"0.01\"")}${trainingNumber("context","context",1)}${trainingNumber("memoryMib","memory MiB",1)}
  ${trainingNumber("dimension","dimension",1)}${trainingNumber("feedForward","feed-forward",1)}${trainingNumber("layers","layers",1)}${trainingNumber("heads","heads",1)}${trainingNumber("kvHeads","KV heads",1)}${trainingNumber("seed","seed",0)}</div>
  <div class="training-note"><b>Gradient accumulation</b><p>Each update processes the selected number of sequences, or preference pairs for DPO, one at a time. Loss averages supervised tokens (DPO: pairs). Memory holds one sequence graph, or one pair.</p><b>Native limitations</b><p>CPU backward. Dense LLaMA and Gemma 4 imports. MoE, MTP training, encoders, mixed precision, and distributed runs are not implemented.</p></div></section></form></div>`;
  document.querySelector("#train-mode").value=s.training.mode;
  viewRoot.querySelector('.training-view').insertAdjacentHTML('beforeend','<section class="panel training-live" id="training-live" aria-label="Training telemetry"></section>');updateTrainingLive();
  if(!window.__TAURI_INTERNALS__){document.querySelector('#start-training').disabled=true;document.querySelector('#start-training').title='Training launches from the desktop app';}
}
function trainingNumber(id,label,min,extra=""){return `<label>${label}<input data-train="${id}" type="number" min="${min}" value="${esc(s.training[id])}" ${extra}></label>`}

function logsView(){
  const lines=[...s.appLogs,...s.logs],filter=(s.logFilter||"all");
  const shown=filter==="all"?lines:lines.filter(line=>line.includes(`[${filter}]`));
  viewRoot.innerHTML=`<div class="logs-view"><header><div><label>LOGS</label><h1>Runtime stream</h1></div><div><select id="log-filter"><option value="all">all sources</option><option value="engine">engine</option><option value="trainer">trainer</option><option value="ui">ui</option></select><button id="refresh-logs">refresh</button><button id="copy-logs">copy</button><button id="clear-logs">clear</button></div></header><pre id="log-output">${esc(shown.join("\n")||"No log entries.")}</pre></div>`;
  document.querySelector("#log-filter").value=filter;
  document.querySelector('#refresh-logs').insertAdjacentHTML('beforebegin',`<button id="pause-logs" aria-pressed="${ui.logPaused}">${ui.logPaused?'Resume scrolling':'Pause scrolling'}</button>`);
  updateLogOutput();
}

function updateLogOutput(){
  const n=document.querySelector('#log-output');if(!n)return;
  const scroll=n.scrollTop,filter=s.logFilter||'all';
  const lines=[...s.appLogs,...s.logs].slice(-2000).filter(line=>filter==='all'||line.includes(`[${filter}]`));
  const text=lines.join('\n')||'No log entries.';
  if(n.textContent!==text)n.textContent=text;
  n.scrollTop=ui.logPaused?scroll:n.scrollHeight;
}

function settingsView(){
  const d=document.querySelector("#settings");d.innerHTML=`<form method="dialog" class="settings-card"><header><div><label>CONFIGURATION</label><h2>Settings</h2></div><button value="close">×</button></header><div class="settings-grid"><label class="field">provider<select id="provider-type"><option value="local">local C runtime</option><option value="openai">OpenAI-compatible endpoint</option></select></label><label class="field remote-setting">base URL<input id="provider-endpoint" value="${esc(s.provider.endpoint)}" placeholder="https://api.openai.com/v1"></label><label class="field remote-setting">API key<input id="provider-key" type="password" value="${esc(s.provider.apiKey)}" autocomplete="off"></label><label class="field remote-setting">model ID<div class="path-field"><input id="provider-model" value="${esc(s.provider.model)}" placeholder="gpt-5"><button type="button" id="test-provider">test</button></div></label><label class="check"><input id="remember-key" type="checkbox" ${s.provider.remember?"checked":""}> remember API key on this device</label><hr><label class="check"><input id="agent-enabled" type="checkbox" ${s.agent.enabled?"checked":""}> enable Agent mode</label><label class="field">maximum agent turns<input id="agent-turns" type="number" min="1" max="24" value="${s.agent.maxTurns}"></label><p class="settings-note">Agent tools are intentionally read-only: runtime status, loaded models, and local time.</p></div><footer><button value="close">cancel</button><button type="button" id="save-settings" class="blue-action">save configuration</button></footer></form>`;d.querySelector("#provider-type").value=s.provider.type;toggleRemoteSettings();d.showModal();
}
function toggleRemoteSettings(){const remote=document.querySelector("#provider-type")?.value==="openai";document.querySelectorAll(".remote-setting").forEach(n=>n.hidden=!remote)}

function appearanceSettings(){
  const a=window.fyodorAppearance.get(),d=document.querySelector('#settings');
  d.querySelector('.settings-card>header').insertAdjacentHTML('afterend',`<section class="settings-section"><h3>Appearance</h3><div class="theme-grid">${window.fyodorAppearance.themes.map(t=>`<button type="button" data-theme-choice="${t}" aria-pressed="${a.theme===t}">${t.replaceAll('-',' ')}</button>`).join('')}</div><label class="check"><input id="compact-density" type="checkbox" ${a.compact?'checked':''}>Compact spacing</label><p class="fine-print">Appearance changes apply immediately and persist on this device.</p></section>`);
  d.querySelector('.settings-grid').insertAdjacentHTML('afterbegin',`<h3>Connection</h3><label class="check remote-setting"><input id="remote-stream" type="checkbox" ${s.provider.stream?'checked':''}>Stream responses from remote endpoints</label><p class="fine-print">The local C HTTP API currently returns completed responses. Remote streaming supports Stop; Agent mode uses complete tool-call responses.</p>`);
  toggleRemoteSettings();d.querySelector('#save-settings').disabled=s.busy;
}

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
  render();
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

async function externalCompletion(messages,withTools=false,onUpdate){
  if(!s.provider.model)throw new Error("Configure an endpoint model first");
  const body={model:s.provider.model,messages,max_tokens:+s.params.max_tokens,temperature:+s.params.temperature,top_p:+s.params.top_p,stream:false};
  if(+s.params.frequency_penalty)body.frequency_penalty=+s.params.frequency_penalty;
  if(+s.params.presence_penalty)body.presence_penalty=+s.params.presence_penalty;
  if(s.params.stop.trim())body.stop=s.params.stop.split(",").map(x=>x.trim()).filter(Boolean);
  if(withTools)body.tools=agentTools;
  if(onUpdate&&!withTools&&s.provider.stream){body.stream=true;body.stream_options={include_usage:true};return backend.openaiChat(s.provider,body,{onUpdate,signal:ui.request?.controller?.signal});}
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
async function complete(messages,raw=false,onUpdate){
  if(s.agent.enabled&&s.provider.type==="openai")return agentCompletion(messages);
  if(s.provider.type==="openai"){const result=await externalCompletion(messages,false,onUpdate);if(typeof result.text==='string')return result;const choice=result.choices?.[0]?.message;if(!choice)throw new Error("Endpoint returned no assistant message");return{text:choice.content||"",tokens:result.usage?.completion_tokens||0};}
  return localCompletion(raw?messages.at(-1).content:formatConversation(messages,s.models.find(m=>mid(m)===s.selected)?.architecture));
}

async function generate(prompt,source="chat"){
  if(s.busy||s.actionBusy)return;
  s.busy=true;s.lastPrompt=prompt;ui.error='';
  const target=source==='chat'?chat():null,started=performance.now();
  const request={chatId:target?.id,partial:'',frame:0,controller:s.provider.type==='openai'&&s.provider.stream&&!s.agent.enabled?new AbortController():null};
  ui.request=request;
  if(target){target.draft='';target.messages.push({role:'user',text:prompt});if(target.title==='New chat')target.title=prompt.slice(0,42)}
  render();
  const update=text=>{
    request.partial=text;
    if(request.frame)return;
    // Streaming updates touch only one text node and preserve manual scrolling.
    // Navigation and theme changes do not restart or redirect the request.
    request.frame=requestAnimationFrame(()=>{
      request.frame=0;
      if(ui.request!==request)return;
      const n=target&&s.view==='chat'&&chat().id===target.id?document.querySelector('#typing'):!target&&s.view==='playground'?document.querySelector('#lab-output .thinking'):null;
      if(!n)return;
      const list=n.closest('.message-list'),follow=list&&list.scrollHeight-list.scrollTop-list.clientHeight<80;
      n.textContent=text;
      if(follow)list.scrollTop=list.scrollHeight;
    });
  };
  try{
    const messages=target?[{role:'system',content:s.params.system||'You are Fyodor.'},...target.messages.map(m=>({role:m.role==='tool'?'user':m.role,content:m.text}))]:[{role:'system',content:s.params.system||'You are Fyodor.'},{role:'user',content:prompt}];
    const result=await complete(messages,!target,update);
    const answer={role:'assistant',text:result.text||'',tokens:result.generated_tokens||result.tokens||0,seconds:(performance.now()-started)/1000,seed:Number.isSafeInteger(result.seed)?result.seed:undefined};
    if(target)target.messages.push(answer);else s.labAnswer=answer;
    log(`Completion finished in ${answer.seconds.toFixed(2)}s`);
  }catch(error){
    const stopped=error.name==='AbortError';
    ui.error=stopped?'Generation stopped.':error.message;
    log(ui.error);toast(ui.error,stopped?'':'error');
    if(request.partial){const answer={role:'assistant',text:request.partial,seconds:(performance.now()-started)/1000,tokens:0};if(target)target.messages.push(answer);else s.labAnswer=answer;}
    else if(target){target.messages.pop();target.draft=prompt;}
  }finally{
    cancelAnimationFrame(request.frame);ui.request=null;s.busy=false;save();render();
  }
}

async function refreshTraining(){if(!window.__TAURI_INTERNALS__)return;try{s.trainStatus=await invokeDesktop("training_status");if(s.view==="training"){const n=document.querySelector('.train-state');if(n){n.textContent=s.trainStatus.running?(s.trainStatus.stopping?'SAVING / STOPPING':'RUNNING'):s.trainStatus.exitCode==null?'IDLE':'EXIT '+s.trainStatus.exitCode;n.classList.toggle('running',s.trainStatus.running)}const start=document.querySelector('#start-training'),stop=document.querySelector('#stop-training');if(start)start.disabled=s.trainStatus.running;if(stop)stop.disabled=!s.trainStatus.running||s.trainStatus.stopping;await refreshLogs();updateTrainingLive()}}catch(error){log(error.message,"trainer")}}
async function refreshLogs(){if(window.__TAURI_INTERNALS__)try{s.logs=(await invokeDesktop("runtime_logs")).slice(-2000)}catch(error){log(error.message,"ui")}if(s.view==="logs")updateLogOutput()}
function updateTrainingLive(){
  const n=document.querySelector('#training-live');if(!n)return;
  const lines=s.logs.filter(x=>x.includes('[trainer]')).slice(-200),points=[];
  for(const line of lines){const metric=parseTrainingMetric(line);if(metric)points.push(metric)}
  const latest=points.at(-1);let chart='';
  if(points.length>1){const lo=Math.min(...points.map(p=>p.loss)),hi=Math.max(...points.map(p=>p.loss));const coords=points.map((p,i)=>`${(i/(points.length-1)*580+10).toFixed(2)},${(110-(p.loss-lo)/(hi-lo||1)*100).toFixed(2)}`).join(' ');chart=`<svg class="loss-chart" viewBox="0 0 600 120" role="img" aria-label="Training loss over recent logged steps"><polyline points="${coords}" fill="none" stroke="currentColor" stroke-width="2"/></svg>`}
  const throughput=latest?.tokensPerSecond==null?'':` · ${latest.tokensPerSecond.toFixed(1)} tokens/s`;
  const threads=latest?.cpuThreads==null?'':` · ${latest.cpuThreads} CPU threads`;
  const memory=latest?.graphBytes==null?'':` · graph ${(latest.graphBytes/1048576).toFixed(1)} MiB`;
  n.innerHTML=`<h2>${latest?`Step ${latest.step} · loss ${latest.loss.toFixed(6)}${throughput}${memory}${threads}`:'Run telemetry'}</h2>${chart}<pre>${esc(lines.join('\n')||'Trainer output appears here during a desktop run.')}</pre>`;
}
function collectTraining(){document.querySelectorAll("[data-train]").forEach(n=>{s.training[n.dataset.train]=n.type==="number"?+n.value:n.value});save()}

document.addEventListener("click",async event=>{
  if(event.target.closest('#stop-generation')){ui.request?.controller?.abort();return;}
  if(event.target.closest('#collapse-sidebar')){const a=window.fyodorAppearance.get();window.fyodorAppearance.set({collapsed:!a.collapsed});return;}
  const theme=event.target.closest('[data-theme-choice]');if(theme){window.fyodorAppearance.set({theme:theme.dataset.themeChoice});document.querySelectorAll('[data-theme-choice]').forEach(n=>n.setAttribute('aria-pressed',n===theme));return;}
  if(event.target.closest('#pause-logs')){ui.logPaused=!ui.logPaused;const n=event.target.closest('#pause-logs');n.textContent=ui.logPaused?'Resume scrolling':'Pause scrolling';n.setAttribute('aria-pressed',ui.logPaused);updateLogOutput();return;}
  if(event.target.closest('#older-messages')){ui.messageLimit+=80;const list=document.querySelector('.message-list'),height=list.scrollHeight;chatView();requestAnimationFrame(()=>{const n=document.querySelector('.message-list');if(n)n.scrollTop=n.scrollHeight-height});return;}
  const code=event.target.closest('[data-copy-code]');if(code){await navigator.clipboard.writeText(code.closest('.code-block').querySelector('code').textContent);toast('Code copied');return;}
  const edit=event.target.closest('[data-edit]');if(edit&&!s.busy){const index=+edit.dataset.edit;chat().draft=chat().messages[index].text;ui.editIndex=index;ui.editChatId=chat().id;chatView();document.querySelector('#prompt').focus();return;}
  if(event.target.closest("[data-refresh]")){await refresh();return;}
  const compute=event.target.closest("[data-compute]");if(compute&&!compute.disabled){await setCompute(+compute.dataset.id,compute.dataset.compute);return;}
  const url=event.target.closest("[data-copy-url]");if(url){await navigator.clipboard.writeText(s.baseUrl+url.dataset.copyUrl);toast("URL copied");return;}
  if(event.target.closest("[data-copy-token]")){const c=await getBackendConnection();if(c.token){await navigator.clipboard.writeText(c.token);toast("Local token copied");}return;}
  // The body also has data-view for CSS. Match navigation buttons only, or
  // every click would find the body, rerender, and discard its actual action.
  const view=event.target.closest("button[data-view]")?.dataset.view;if(view){s.view=view;document.body.classList.remove("menu-open");render();if(view==="logs")refreshLogs();return}
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
  const regenerate=event.target.closest("[data-regen]");if(regenerate&&!s.busy){const i=+regenerate.dataset.regen;let index=i-1;while(index>=0&&chat().messages[index].role!=='user')index--;if(index>=0){const prompt=chat().messages[index].text;chat().messages.splice(index);generate(prompt)}}
  if(event.target.closest("#system-toggle"))document.querySelector("#system-prompt").toggleAttribute("hidden");
  if(event.target.closest("#agent-toggle")){if(s.provider.type!=="openai"){toast("Agent tools require an OpenAI-compatible endpoint.");return}s.agent.enabled=!s.agent.enabled;render()}
  if(event.target.closest("#reset-params")){s.params={...defaults};render()}
  const preset=event.target.closest("[data-preset]")?.dataset.preset;if(preset){Object.assign(s.params,{precise:{temperature:.2,top_p:.8,top_k:20},balanced:{temperature:.8,top_p:.95,top_k:40},creative:{temperature:1.25,top_p:1,top_k:80}}[preset]);render()}
  if(event.target.closest("#run-lab")){const p=document.querySelector("#lab-prompt").value.trim();if(p)generate(p,"lab")}
  if(event.target.closest("[data-copy-lab]"))navigator.clipboard.writeText(s.labAnswer?.text||"");
  if(event.target.closest(".mobile-menu")){document.body.classList.toggle("menu-open");syncNavigation();if(document.body.classList.contains('menu-open'))document.querySelector('.rail nav button').focus();}if(event.target.closest(".scrim")){document.body.classList.remove("menu-open");syncNavigation();document.querySelector('.mobile-menu').focus();}
  if(event.target.closest("#open-settings")){settingsView();appearanceSettings()}if(event.target.closest("#provider-type"))toggleRemoteSettings();
  if(event.target.closest("#save-settings")&&!s.busy){s.provider.type=document.querySelector("#provider-type").value;s.provider.endpoint=document.querySelector("#provider-endpoint").value.trim();s.provider.apiKey=document.querySelector("#provider-key").value.trim();s.provider.model=document.querySelector("#provider-model").value.trim();s.provider.remember=document.querySelector("#remember-key").checked;s.provider.stream=document.querySelector('#remote-stream').checked;s.agent.enabled=document.querySelector("#agent-enabled").checked;s.agent.maxTurns=Math.max(1,Math.min(24,+document.querySelector("#agent-turns").value||6));save();document.querySelector("#settings").close();render()}
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
  if(event.target.id==='compact-density')window.fyodorAppearance.set({compact:event.target.checked});
  if(event.target.id==='benchmark-file'&&event.target.files[0]){const file=event.target.files[0];if(file.size>1024*1024){toast('Benchmark files are limited to 1 MiB.','error');return}file.text().then(text=>{ui.benchmarks.unshift(parseBenchmark(text));ui.benchmarks=ui.benchmarks.slice(0,12);if(s.view==='benchmark')render()}).catch(error=>toast(error.message,'error'));}
  if(event.target.id==="global-model"&&s.provider.type==="local"){s.selected=+event.target.value||null;render()}
  if(event.target.id==="chat-title"){chat().title=event.target.value.trim()||"New chat";render()}
  if(event.target.id==="system-prompt"){s.params.system=event.target.value;save()}
  if(event.target.id==="import-file"&&event.target.files[0]){if(event.target.files[0].size>4*1024*1024){toast("Chat imports are limited to 4 MiB.","error");return;}const reader=new FileReader();reader.onload=()=>{try{const c=normalizeChat(JSON.parse(reader.result),id());s.chats.unshift(c);s.active=c.id;s.view="chat";render()}catch{toast("invalid chat file","error")}};reader.readAsText(event.target.files[0])}
});
document.addEventListener("keydown",event=>{if(event.key==='Escape'){document.body.classList.remove('menu-open');syncNavigation();if(narrowNavigation.matches)document.querySelector('.mobile-menu').focus();}if(event.key==='Tab'&&narrowNavigation.matches&&document.body.classList.contains('menu-open')){const nodes=[...document.querySelectorAll('.rail button,.history-panel button,.history-panel input')].filter(n=>!n.disabled&&n.offsetParent!==null),first=nodes[0],last=nodes.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus()}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus()}}if((event.ctrlKey||event.metaKey)&&event.key==='\\'){event.preventDefault();document.querySelector('#collapse-sidebar').click()}if(event.target.id==="prompt"&&event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();event.target.form.requestSubmit()}});
document.addEventListener("submit",async event=>{
  if(event.target.matches(".composer")){event.preventDefault();const p=event.target.prompt.value.trim();if(p&&!s.busy){if(Number.isInteger(ui.editIndex)&&ui.editChatId===chat().id)chat().messages.splice(ui.editIndex);delete ui.editIndex;delete ui.editChatId;generate(p)}return}
  if(event.target.id==="training-form"){event.preventDefault();collectTraining();try{await invokeDesktop("start_training",{args:{...s.training,...(s.training.mode==="pretrain"?{rank:0,base:""}:{})}});log("training run launched","trainer");toast("training started","success");await refreshTraining()}catch(error){toast(error.message,"error");log(error.message,"trainer")}}
});

render();refresh();setInterval(()=>{if(s.view==="training")refreshTraining();if(s.view==="logs")refreshLogs()},1500);
