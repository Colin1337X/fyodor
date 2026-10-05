// The picker owns selection/mapping only; its parent owns busy state, draft
// replacement and requests. Shared state survives view repaints, not reloads.
import {escapeHtml as esc} from './message.js';
import {sourceLimit, sourcePage, resourceSource, chatSource, datasetFromSources} from './dataset-sources.js';

const state={namespace:null,kind:'writing',mode:'cpt',delimiterPolicy:'reject',prompt:'',items:[],next:'',selected:new Map(),open:false};
export function resetDatasetSources(namespace) {
  Object.assign(state,{namespace,items:[],next:'',selected:new Map()});
}
export function datasetSourcePicker(root, {namespace,busy,run,repaint,api,getChats=()=>[],onDraft}) {
  if (state.namespace!==namespace) resetDatasetSources(namespace);
  const chatItems=state.kind==='chats'?getChats():[];
  root.innerHTML=`<details class="panel dataset-source-picker" ${state.open?'open':''}><summary>Create from Fyodor sources · ${state.selected.size} selected</summary>
    <div class="resource-toolbar"><label>Source library<select id="dataset-source-kind"><option value="writing">Writing</option><option value="explore">Explore</option><option value="chats">Local chats</option></select></label><button data-source="browse">Browse sources</button><button data-source="next" ${state.next&&state.kind!=='chats'?'':'disabled'}>Next page</button><button data-source="clear">Clear selection</button></div>
    <ul class="resource-list dataset-source-list">${(state.kind==='chats'?chatItems:state.items).map(item=>{
      const key=state.kind==='chats'?'chat:'+item.id:namespace+':'+item.uri;
      return `<li><label class="check"><input type="checkbox" data-source-key="${esc(key)}" ${state.selected.has(key)?'checked':''}><span>${esc(item.title||'Untitled')}<small>${state.kind==='chats'?item.messages.length+' messages':'Revision '+esc(item.revision)}</small></span></label></li>`;
    }).join('')||'<li class="muted">Browse a library to choose saved sources.</li>'}</ul>
    ${state.selected.size?`<h3>Selected sources · ${state.selected.size} / ${sourceLimit}</h3><ul class="dataset-source-selection">${[...state.selected].map(([key,source])=>`<li><span>${esc(source.title)}<small>${source.kind==='local-chat'?'Local chat snapshot':'Revision '+esc(source.revision)+' · '+esc(source.kind)}</small></span><button data-source-remove="${esc(key)}" aria-label="Remove ${esc(source.title)}">Remove</button></li>`).join('')}</ul>`:''}
    <div class="resource-toolbar"><label>Output format<select id="dataset-source-mode"><option value="pretrain">Pretraining corpus</option><option value="cpt">CPT corpus</option><option value="sft">SFT records</option></select></label>${state.mode==='sft'?`<label>Tabs and line breaks<select id="dataset-source-policy"><option value="reject">Reject inside fields</option><option value="spaces">Replace with spaces</option></select></label>`:''}</div>
    ${state.mode==='sft'?`<label>Prompt for Writing / Explore records<textarea id="dataset-source-prompt" rows="2">${esc(state.prompt)}</textarea></label><p class="muted">Saved resource text becomes the completion. Chats use adjacent user → assistant messages only; tool messages break a pair.</p>`:'<p class="muted">Saved resource text is copied in selection order. Chats include role labels and saved message text.</p>'}
    <button class="primary" data-source="build" ${state.selected.size?'':'disabled'}>Create dataset draft</button><p class="fine-print">Sources stay unchanged. Review the new draft before saving.</p></details>`;
  root.querySelector('details').addEventListener('toggle',event=>state.open=event.target.open);
  const kind=root.querySelector('#dataset-source-kind');kind.value=state.kind;
  kind.addEventListener('change',()=>withFocus(run(async()=>{state.kind=kind.value;state.items=[];state.next='';await browse();}),'#dataset-source-kind'));
  const mode=root.querySelector('#dataset-source-mode');mode.value=state.mode;
  mode.addEventListener('change',()=>{state.mode=mode.value;repaint();focus('#dataset-source-mode');});
  const policy=root.querySelector('#dataset-source-policy');if(policy){policy.value=state.delimiterPolicy;policy.addEventListener('change',()=>state.delimiterPolicy=policy.value);}
  root.querySelector('#dataset-source-prompt')?.addEventListener('input',event=>state.prompt=event.target.value);
  root.querySelectorAll('[data-source-key]').forEach(control=>control.addEventListener('change',()=>withFocus(run(async()=>{
    const key=control.dataset.sourceKey;
    if(!control.checked){state.selected.delete(key);return;}
    if(state.selected.size>=sourceLimit)throw Error('Select at most 32 sources.');
    const source=state.kind==='chats'?chatSource(chatItems.find(item=>'chat:'+item.id===key)):resourceSource(state.items.find(item=>namespace+':'+item.uri===key),namespace);
    state.selected.set(key,source);
  }),`[data-source-key="${CSS.escape(control.dataset.sourceKey)}"]`)));
  root.querySelectorAll('[data-source-remove]').forEach(control=>control.addEventListener('click',()=>{state.selected.delete(control.dataset.sourceRemove);repaint();focus('[data-source=clear]');}));
  root.querySelectorAll('[data-source]').forEach(control=>control.addEventListener('click',()=>withFocus(run(async()=>{
    if(control.dataset.source==='clear')state.selected.clear();
    else if(control.dataset.source==='browse')await browse();
    else if(control.dataset.source==='next'&&state.next)await browse(state.next);
    else if(control.dataset.source==='build')await onDraft(()=>datasetFromSources(api,[...state.selected.values()],state));
  }),`[data-source="${control.dataset.source}"]`)));
  if(busy)root.querySelectorAll('button,input,textarea,select').forEach(control=>control.disabled=true);
  // Inject the existing resource transport through the parent for tests and to
  // avoid coupling this renderer to native API implementation details.
  function focus(selector) {
    // Parent repaint replaces this root. Look up its current replacement and
    // restore the initiating control, but never steal focus from another view.
    if(document.body.dataset.view==='datasets'&&state.namespace===namespace)
      document.querySelector('#dataset-source-picker '+selector)?.focus();
  }
  function withFocus(work, selector) {void work.then(()=>focus(selector));}
  async function browse(after='') {
    if(state.kind==='chats'){state.items=[];state.next='';return;}
    const page=await sourcePage(api,namespace,state.kind,after);state.items=page.resources;state.next=page.next;
  }
}
