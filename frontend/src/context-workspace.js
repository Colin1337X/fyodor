// The local identity labels explicit grants; it is not a remote credential.
// Layer order, token reservation and receipt visibility are enforced in C.
import {backend} from './api.js';
import {escapeHtml as esc} from './message.js';
import {contextReceiptSources} from './context-receipt-view.js';
const key='fyodor-context-identity';
const layers=['explicit','workspace','session','retrieved','global'];
const sourceLayers=new Map();
let principal;
try{principal=localStorage.getItem(key);}catch{}
if(!/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(principal||'') || principal==='00000000-0000-0000-0000-000000000000')principal=crypto.randomUUID();
try{localStorage.setItem(key,principal);}catch{}
const state={principal,namespace:'workspace',items:[],next:'',selected:new Set(),prompt:'',budget:4096,tokens:32,busy:false,error:'',notice:'',receiptId:'',receipt:null};
let mount=0,repaint=()=>{};
let receiptPage=[],receiptNext='';
let searchQuery='',searchMode=false;
export function contextWorkspace(root,modelId){
  const generation=++mount;
  const current=()=>generation===mount && document.body.dataset.view==='context';
  async function history(after=''){const result=await backend.contextReceipts(state.namespace,state.principal,after);receiptPage=result.receipts;receiptNext=result.next;}
  async function load(after=''){
    const result=searchMode?await backend.contextSearch(state.namespace,state.principal,searchQuery):await backend.resources(state.namespace,after);
    state.items=result.resources;state.next=result.next;await history();
  }
  async function run(work){if(state.busy)return;state.busy=true;state.error='';state.notice='';paint();try{await work();}catch(error){state.error=error.message;}finally{state.busy=false;repaint();}}
  function paint(){
    if(!current())return;
    root.innerHTML=`<section class="resource-workspace context-workspace"><header><p class="eyebrow">LOCAL CONTEXT</p><h1>Generate with resources</h1><p class="muted">Choose saved context, grant access explicitly, then inspect the executed prompt and source revisions.</p></header>
      <details><summary>Workspace and identity</summary><label>Namespace<input id="context-namespace" maxlength="64" value="${esc(state.namespace)}"></label><label>Context identity<input id="context-principal" maxlength="36" value="${esc(state.principal)}"></label></details>
      <p role="status" class="${state.error?'danger-action':'muted'}">${esc(state.error||state.notice||(state.busy?'Working…':'Select a model to begin.'))}</p>
      <div class="resource-columns"><section aria-label="Context selection"><h2>Saved resources</h2><div class="resource-toolbar"><button data-context="refresh">First page</button><button data-context="next" ${state.next?'':'disabled'}>Next page</button><button data-context="clear">Clear selection</button></div>
      <ul class="resource-list">${state.items.map(item=>`<li><label class="check"><input type="checkbox" data-context-uri="${esc(item.uri)}" ${state.selected.has(item.uri)?'checked':''}><span>${esc(item.title||'Untitled')}<small>${esc(item.uri)}</small></span></label><label>Context layer<select data-context-layer="${esc(item.uri)}">${layers.map(layer=>`<option value="${layer}" ${layer===(sourceLayers.get(item.uri)||'explicit')?'selected':''}>${layer}</option>`).join('')}</select></label></li>`).join('')||'<li>No resources. Create documents or notes in Resources.</li>'}</ul>
      <p id="context-selected">${state.selected.size} / 64 selected.</p><div class="resource-toolbar"><button data-context="grant">Grant read only</button><button data-context="revoke">Revoke all grants</button></div><p class="fine-print">Allow or remove access to the selected resources.</p></section>
      <section class="panel"><label>Prompt<textarea id="context-prompt" rows="6">${esc(state.prompt)}</textarea></label><div class="resource-toolbar"><label>Context byte budget<input id="context-budget" type="number" min="0" max="1048576" value="${state.budget}"></label><label>Maximum new tokens<input id="context-tokens" type="number" min="1" max="256" value="${state.tokens}"></label><button class="primary" data-context="generate" ${modelId?'':'disabled'}>Generate and save receipt</button></div></section></div>
      <section class="panel context-receipt"><h2>Executed request receipt</h2><div class="resource-toolbar"><label>Receipt ID<input id="context-receipt-id" value="${esc(state.receiptId)}"></label><button data-context="receipt">Read saved receipt</button></div>${state.receipt?`<h3>Output</h3><pre id="context-output">${esc(state.receipt.output)}</pre><h3>Context used</h3>${contextReceiptSources(state.receipt)}<details open><summary>Exact prompt and sources</summary><h4>Prompt</h4><pre id="context-exact-prompt">${esc(state.receipt.prompt)}</pre><details><summary>Source revisions and byte ranges</summary><pre id="context-sources">${esc(state.receipt.sources_json)}</pre></details><h4>Model and sampling</h4><pre>${esc(state.receipt.metadata_json)}</pre></details>`:'<p class="muted">Your saved requests will appear here.</p>'}</section></section>`;
    if(state.busy)root.querySelectorAll('button,input,textarea,select').forEach(n=>n.disabled=true);
    root.querySelector('[aria-label="Context selection"] h2').insertAdjacentHTML('afterend',`<div class="resource-toolbar"><label>Search permitted context<input id="context-query" value="${esc(searchQuery)}" maxlength="128"></label><button data-context="search">Search</button><button data-context="browse">Browse all local resources</button></div><p class="fine-print">${searchMode?'Matching resources you can use.':'All saved resources. Grant access before generating.'}</p><button data-context="grant-search">Grant read and search</button>`);
    root.querySelector('#context-query').addEventListener('input',event=>searchQuery=event.target.value);
    root.querySelector('.context-receipt h2').insertAdjacentHTML('afterend',`<div class="resource-toolbar"><button data-context="history">Refresh history</button><button data-context="history-next" ${receiptNext?'':'disabled'}>Next receipts</button></div><ul class="receipt-history">${receiptPage.map(item=>`<li><button data-receipt-id="${esc(item.id)}">${esc(item.id)} · ${esc(new Date(Number(item.created_ms)).toLocaleString())}</button></li>`).join('')||'<li class="muted">No accessible receipts on this page.</li>'}</ul>`);
    root.querySelectorAll('[data-receipt-id]').forEach(n=>n.addEventListener('click',()=>{state.receiptId=n.dataset.receiptId;void action('receipt');}));
    if(state.busy)root.querySelectorAll('button,input,textarea,select').forEach(n=>n.disabled=true);
    for(const [id,keyName] of [['context-prompt','prompt'],['context-receipt-id','receiptId']])root.querySelector('#'+id).addEventListener('input',event=>state[keyName]=event.target.value);
    for(const [id,keyName,max,min] of [['context-budget','budget',1048576,0],['context-tokens','tokens',256,1]])root.querySelector('#'+id).addEventListener('change',event=>{const value=Number(event.target.value);if(Number.isInteger(value)&&value>=min&&value<=max)state[keyName]=value;else event.target.value=state[keyName];});
    root.querySelector('#context-namespace').addEventListener('change',event=>{if(!/^[A-Za-z0-9_-]{1,64}$/.test(event.target.value)){event.target.value=state.namespace;return;}state.namespace=event.target.value;state.selected.clear();sourceLayers.clear();state.receipt=null;state.receiptId='';state.items=[];state.next='';receiptPage=[];receiptNext='';void run(()=>load());});
    root.querySelector('#context-principal').addEventListener('change',event=>{
      const value=event.target.value;
      if(!/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(value)||value==='00000000-0000-0000-0000-000000000000'){event.target.value=state.principal;return;}
      state.principal=value;try{localStorage.setItem(key,value);}catch{}state.receipt=null;state.receiptId='';state.selected.clear();sourceLayers.clear();state.items=[];state.next='';receiptPage=[];receiptNext='';void run(()=>load());
    });
    root.querySelectorAll('[data-context-layer]').forEach(n=>n.addEventListener('change',()=>{if(layers.includes(n.value))sourceLayers.set(n.dataset.contextLayer,n.value);}));
    root.querySelectorAll('[data-context-uri]').forEach(n=>n.addEventListener('change',()=>{
      if(n.checked && state.selected.size<64){state.selected.add(n.dataset.contextUri);if(!sourceLayers.has(n.dataset.contextUri))sourceLayers.set(n.dataset.contextUri,searchMode?'retrieved':'explicit');paint();}else {state.selected.delete(n.dataset.contextUri);n.checked=false;}
      root.querySelector('#context-selected').textContent=state.selected.size+' / 64 selected.';
    }));
    root.querySelectorAll('[data-context]').forEach(n=>n.addEventListener('click',()=>void action(n.dataset.context)));
  }
  async function action(action){
    await run(async()=>{
      if(action==='refresh')await load();
      else if(action==='search'){if(!searchQuery.trim())throw Error('Enter a search query.');searchMode=true;state.items=[];state.next='';await load();}
      else if(action==='browse'){searchMode=false;await load();}
      else if(action==='history')await history();
      else if(action==='history-next')await history(receiptNext);
      else if(action==='next')await load(state.next);
      else if(action==='clear'){state.selected.clear();sourceLayers.clear();}
      else if(action==='grant'||action==='grant-search'||action==='revoke'){
        state.receipt=null;
        if(!state.selected.size)throw Error('Select at least one resource.');
        let changed=0;
        try{for(const uri of state.selected){await backend.contextPermissions(state.namespace,state.principal,uri,action==='grant'?1:action==='grant-search'?5:0);++changed;}}
        catch(error){throw Error(`${changed} grants changed before the operation stopped: ${error.message}`);}
        state.notice=`${changed} resource grants ${action==='grant'?'set to read only':action==='grant-search'?'set to read and search':'revoked'}.`;
        if(searchMode)await load();else await history();
      } else if(action==='generate'){
        state.receipt=null;state.receiptId='';
        const result=await backend.contextGenerate({namespace:state.namespace,principal:state.principal,model_id:modelId,resources:[...state.selected],layers:[...state.selected].map(uri=>sourceLayers.get(uri)||'explicit'),prompt:state.prompt,byte_budget:state.budget,max_tokens:state.tokens});
        state.receiptId=result.receipt_id;
        state.receipt=await backend.contextReceipt(state.namespace,state.principal,state.receiptId);
        await history();
        state.notice=`Saved receipt · ${result.prompt_tokens} prompt tokens · ${result.generated_tokens} generated tokens · ${result.context_bytes} context bytes.`;
      } else if(action==='receipt'){
        state.receipt=null;
        state.receipt=await backend.contextReceipt(state.namespace,state.principal,state.receiptId);
      }
    });
  }
  repaint=paint;paint();if(!state.busy)void run(()=>load());
}

export function contextIdentity(){return state.principal;}
