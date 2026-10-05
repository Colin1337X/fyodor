import {backend} from './api.js';
import {writingLorePanel, writingLoreUris} from './writing-lore.js';
import {arrangeWritingStudio, writingStatsLabel, writingGuidanceKey, writingPrompt} from './writing-studio.js';
import {decodeWritingFile, downloadWritingText, writingFileLimit} from './writing-files.js';
import {contextIdentity} from './context-workspace.js';
import {escapeHtml as esc} from './message.js';
// Resource edits use CAS in the native store. Parent links and accepted receipts
// are preserved there; never round-trip arbitrary metadata through JS numbers.
// Context grants are explicit and do not inherit from folder membership.
// The C Writing provider adds readable ancestors in one authorized snapshot;
// client selections must not enumerate or duplicate that hierarchy.
const newState=()=>({guidance:new Map(),focus:false,items:[],projects:[],loreItems:[],loreNext:'',loreTitles:new Map(),loreOpen:false,folder:null,namespace:'workspace',next:'',draft:null,revision:'0',dirty:false,busy:false,error:'',notice:'',history:[],historyNext:'0',preview:null,instructions:'',budget:4096,tokens:32,context:new Set(),generated:null,receipt:null,accepted:null});
const states={resources:newState(),writing:newState()};
window.addEventListener('beforeunload',event=>{
  if(Object.values(states).some(state=>state.dirty)){event.preventDefault();event.returnValue='';}
});
let mount=0;
let repaint=()=>{};
export function resourceWorkspace(root,view='resources',modelId=null){
  const state=states[view];const writing=view==='writing';
  const generation=++mount;
  const current=()=>generation===mount && document.body.dataset.view===view;
  function paint(){
    if(!current())return;
    const d=state.draft;
    const folderOptions=selected=>`<option value="">Unfiled</option>${state.projects.filter(p=>p.uri!==d?.uri).map(p=>`<option value="${esc(p.uri)}" ${p.uri===selected?'selected':''}>${esc(p.title||'Untitled project')}</option>`).join('')}`;
    root.innerHTML=`<section class="resource-workspace"><header class="view-heading"><div><p class="eyebrow">LOCAL LIBRARY</p><h1>${writing?'Writing':'Resources'}</h1><p class="muted">${writing?'Markdown and plain text, notes and reusable characters.':'Documents and notes saved in your local workspace.'}</p></div></header>
      <div class="resource-toolbar"><label>Namespace<input id="resource-namespace" value="${esc(state.namespace)}" maxlength="64"></label><button data-resource="refresh">Refresh</button><button data-resource="new-document">New document</button><button data-resource="new-note">New note</button>${writing?'<button data-resource="new-character">New character</button><button data-resource="new-project">New project / folder</button>':''}<label class="file-button">Import package<input id="resource-import" type="file" accept=".json" hidden></label></div>
      <p role="status" class="${state.error?'danger-action':'muted'}">${esc(state.error||state.notice||(state.busy?'Working…':''))}</p>
      <div class="resource-columns"><section aria-label="Resource list"><ul class="resource-list">${state.items.map(r=>`<li><button data-resource-uri="${esc(r.uri)}"><b>${esc(r.title||'Untitled')}</b><small>${esc(r.uri.split('/')[3]?.replace(/s$/,'')||'Resource')}</small><small>Revision ${esc(r.revision)}</small></button>${writing&&r.uri!==d?.uri?`<label class="check"><input type="checkbox" data-writing-context="${esc(r.uri)}" ${state.context.has(r.uri)?'checked':''}>Use as additional context</label>`:''}</li>`).join('')||'<li class="muted">No resources on this page.</li>'}</ul><button data-resource="next" ${state.next?'':'disabled'}>Next page</button></section>
      <section class="panel resource-editor" aria-label="Resource editor">${d?`<p class="fine-print">${state.revision==='0'?'New resource':'Revision '+esc(state.revision)}${state.dirty?' · Unsaved changes':''}</p><label>Title<input id="resource-title" maxlength="1024" value="${esc(d.title)}"></label><label>Content<textarea id="resource-content" rows="16">${esc(d.content)}</textarea></label><div class="resource-toolbar"><button class="primary" data-resource="save">Save</button><button data-resource="discard">Discard editor</button><button data-resource="export">Export package</button>${state.revision!=='0'?'<button class="danger-action" data-resource="delete">Delete resource</button>':''}</div>`:'<h2>Open a resource</h2><p class="muted">Select a saved resource or create a document or note.</p>'}</section></div>${d&&state.revision!=='0'?`<section class="panel resource-history"><h2>Revision history</h2><div class="resource-toolbar"><button data-resource="history">Latest revisions</button><button data-resource="history-next" ${state.historyNext!=='0'?'':'disabled'}>Older revisions</button></div><ul>${state.history.map(item=>`<li><button data-history-revision="${esc(item.revision)}">Revision ${esc(item.revision)}${item.deleted?' · Deleted':''}</button> <small>${esc(new Date(Number(item.modified_ms)).toLocaleString())}</small></li>`).join('')||'<li>No revisions loaded.</li>'}</ul>${state.preview?`<h3>Revision ${esc(state.preview.revision)}${state.preview.deleted?' · Deleted':''}</h3><h4>${esc(state.preview.resource.title)}</h4><pre id="resource-history-content">${esc(state.preview.resource.content)}</pre><button data-resource="copy-history">Copy title and text to editor</button>`:''}</section>`:''}</section>`;
    if(writing){
      root.querySelector('.resource-toolbar').insertAdjacentHTML('beforeend',`<label>Browse folder<select id="writing-folder"><option value="*">All Writing</option><option value="" ${state.folder===''?'selected':''}>Unfiled</option>${state.projects.map(p=>`<option value="${esc(p.uri)}" ${p.uri===state.folder?'selected':''}>${esc(p.title||'Untitled project')}</option>`).join('')}</select></label>`);
      if(d&&state.revision!=='0')root.querySelector('.resource-editor').insertAdjacentHTML('beforeend',`<section><h2>Project / folder</h2><div class="resource-toolbar"><label>Place in<select id="writing-parent">${folderOptions(d.metadata?.writing_parent||'')}</select></label><button data-resource="move">Move resource</button></div></section>`);
      root.querySelector('#writing-folder').addEventListener('change',event=>{state.folder=event.target.value==='*'?null:event.target.value;void run(()=>load());});
    }
    if(writing&&d&&state.revision!=='0')root.querySelector('.resource-editor').insertAdjacentHTML('beforeend',`<section class="writing-generate"><h2>Write with a model</h2><p class="fine-print">${state.context.size} additional sources selected.</p><label>Instructions<textarea id="writing-instructions" rows="3">${esc(state.instructions)}</textarea></label><div class="resource-toolbar"><label>Context byte budget<input id="writing-budget" type="number" min="0" max="1048576" value="${state.budget}"></label><label>New tokens<input id="writing-tokens" type="number" min="1" max="256" value="${state.tokens}"></label><button data-resource="writing-grant">Set read-only grants</button></div><p class="fine-print">Allow the model to read this resource and the selected context.</p><div class="resource-toolbar">${['generate','rewrite','continue'].map(mode=>`<button data-writing-mode="${mode}" ${modelId?'':'disabled'}>${mode[0].toUpperCase()+mode.slice(1)}</button>`).join('')}</div>${state.generated?`<h3>Generated preview</h3><pre id="writing-output">${esc(state.generated.text)}</pre><p>Receipt: <code>${esc(state.generated.receipt_id)}</code></p><button data-resource="writing-apply">${state.generated.mode==='continue'?'Append preview to editor':'Replace editor text with preview'}</button>${state.receipt?`<details><summary>Executed prompt and sources</summary><pre id="writing-exact-prompt">${esc(state.receipt.prompt)}</pre><pre>${esc(state.receipt.sources_json)}</pre></details>`:''}`:''}</section>`);
    if(writing&&d&&state.revision!=='0'){
      root.querySelector('.resource-editor').insertAdjacentHTML('beforeend',writingLorePanel(d,state.loreItems,state.loreNext,state.loreTitles));
      const lore=root.querySelector('.writing-lore');lore.open=state.loreOpen;
      lore.addEventListener('toggle',()=>{if(lore.isConnected)state.loreOpen=lore.open;});
      lore.querySelectorAll('[data-lore-action]').forEach(button=>button.addEventListener('click',()=>void action('lore-'+button.dataset.loreAction,button.dataset.loreUri)));
    }
    if(writing){
      arrangeWritingStudio(root,state);
      root.querySelector('[data-writing-focus]')?.addEventListener('click',()=>{
        state.focus=!state.focus;paint();root.querySelector('[data-writing-focus]')?.focus();
      });
    }
    root.querySelectorAll('[data-writing-context]').forEach(n=>n.addEventListener('change',()=>{if(n.checked&&state.context.size<63)state.context.add(n.dataset.writingContext);else{state.context.delete(n.dataset.writingContext);n.checked=false;}paint();}));
    root.querySelector('#writing-instructions')?.addEventListener('input',event=>state.instructions=event.target.value);
    root.querySelectorAll('[data-writing-guidance]').forEach(input=>input.addEventListener('input',()=>{
      const key=writingGuidanceKey(state.namespace,state.draft.uri);
      state.guidance.set(key,{...state.guidance.get(key),[input.dataset.writingGuidance]:input.value});
    }));
    for(const [id,key,max,min] of [['writing-budget','budget',1048576,0],['writing-tokens','tokens',256,1]])root.querySelector('#'+id)?.addEventListener('change',event=>{const value=Number(event.target.value);if(Number.isInteger(value)&&value>=min&&value<=max)state[key]=value;else event.target.value=state[key];});
    root.querySelectorAll('[data-writing-mode]').forEach(n=>n.addEventListener('click',()=>void action('writing-'+n.dataset.writingMode)));
    if(state.busy)root.querySelectorAll('button,input,textarea,select').forEach(n=>n.disabled=true);
    function markEdited(){state.dirty=true;root.querySelector('.resource-editor .fine-print').textContent=(state.revision==='0'?'New resource':'Revision '+state.revision)+' · Unsaved changes';}
    root.querySelector('#resource-title')?.addEventListener('input',event=>{state.draft.title=event.target.value;markEdited();});
    root.querySelector('#resource-content')?.addEventListener('input',event=>{state.draft.content=event.target.value;markEdited();const count=root.querySelector('#writing-count');if(count)count.textContent=writingStatsLabel(state.draft.content);});
    if(writing&&d){
      root.querySelector('[data-resource="save"]').title='Save (Ctrl/Cmd+S)';
      root.querySelector('.resource-editor').addEventListener('keydown',event=>{
        if((event.ctrlKey||event.metaKey)&&!event.altKey&&!event.shiftKey&&event.key.toLowerCase()==='s'){
          event.preventDefault();void action('save');
        }
      });
    }
    root.querySelector('#resource-namespace').addEventListener('change',event=>{
      if(state.dirty){state.error='Save or discard the current editor before changing namespaces.';paint();return;}
      if(!/^[A-Za-z0-9_-]{1,64}$/.test(event.target.value)){state.error='Use 1–64 letters, digits, underscores or hyphens.';paint();return;}
      state.namespace=event.target.value;state.loreItems=[];state.loreNext='';state.loreTitles.clear();state.folder=null;state.projects=[];state.draft=null;state.context.clear();state.generated=null;state.receipt=null;state.accepted=null;void run(()=>load());
    });
    root.querySelector('#resource-import').addEventListener('change',event=>{
      const file=event.target.files[0];if(!file)return;
      void run(async()=>{if(state.dirty)throw Error('Save or discard the current editor first.');if(file.size>8*1024*1024)throw Error('Package exceeds 8 MiB.');const text=await file.text();if(writing&&!/^fyodor:\/\/writing\//.test(JSON.parse(text)?.uri||''))throw Error('Choose a Writing resource package.');await backend.importResource(state.namespace,text);await load();state.notice='Package imported.';});
    });
    root.querySelector('#writing-text-import')?.addEventListener('change',event=>{
      const file=event.target.files[0];if(!file)return;
      void run(async()=>{
        if(state.dirty)throw Error('Save or discard the current editor before importing text.');
        if(file.size>writingFileLimit)throw Error('Writing files must be 1 MiB or smaller.');
        const text=decodeWritingFile(new Uint8Array(await file.arrayBuffer()),file.name);
        state.draft={schema:1,uri:'fyodor://writing/documents/'+crypto.randomUUID(),...text,metadata:state.folder?{writing_parent:state.folder}:{},provenance:{}};
        Object.assign(state,{revision:'0',dirty:true,generated:null,receipt:null,accepted:null,preview:null,history:[],historyNext:'0'});
        state.notice='Text imported into a new draft. Save to keep it.';
      });
    });
    root.querySelectorAll('.file-button').forEach(control=>{
      control.tabIndex=state.busy?-1:0;control.setAttribute('role','button');
      control.setAttribute('aria-disabled',String(state.busy));
      control.addEventListener('keydown',event=>{
        if(!state.busy&&(event.key==='Enter'||event.key===' ')){event.preventDefault();control.querySelector('input[type=file]')?.click();}
      });
    });
    root.querySelectorAll('[data-history-revision]').forEach(n=>n.addEventListener('click',()=>void run(async()=>{state.preview=await backend.resource(state.namespace,state.draft.uri,n.dataset.historyRevision);})));
    root.querySelectorAll('[data-resource-uri]').forEach(n=>n.addEventListener('click',()=>void run(async()=>{
      if(state.dirty)throw Error('Save or discard the current editor first.');
      const result=await backend.resource(state.namespace,n.dataset.resourceUri);state.draft=result.resource;state.revision=result.revision;state.preview=null;state.generated=null;state.receipt=null;state.accepted=null;state.context.delete(state.draft.uri);await history();
    })));
    root.querySelectorAll('[data-resource]').forEach(n=>n.addEventListener('click',()=>void action(n.dataset.resource)));
  }
  async function load(after=''){
    if(writing&&!after){
      const projects=[];let cursor='';
      do{const page=await backend.resources(state.namespace,cursor,'projects');projects.push(...page.resources);cursor=page.next;}while(cursor);
      state.projects=projects;
      if(state.folder&&!projects.some(p=>p.uri===state.folder))state.folder=null;
    }
    const data=await backend.resources(state.namespace,after,writing?'writing':'all',writing?state.folder:null);state.items=data.resources;state.next=data.next;
  }
  async function history(before='0'){
    const data=await backend.resourceHistory(state.namespace,state.draft.uri,before);
    state.history=data.revisions;state.historyNext=data.next;
    if(writing){
      const missing=writingLoreUris(state.draft).filter(uri=>!state.loreTitles.has(uri));
      const titles=await Promise.all(missing.map(async uri=>{
        try{return [uri,(await backend.resource(state.namespace,uri)).resource.title||'Untitled lore'];}
        catch{return [uri,'Unavailable lore'];}
      }));
      for(const [uri,title] of titles)state.loreTitles.set(uri,title);
    }
  }
  async function run(work){if(state.busy)return;state.busy=true;state.error='';state.notice='';paint();try{await work();}catch(error){state.error=error.message;}finally{state.busy=false;repaint();}}
  async function reloadDraft(){
    const updated=await backend.resource(state.namespace,state.draft.uri);
    state.draft=updated.resource;state.revision=updated.revision;
    state.dirty=false;
    state.generated=null;state.receipt=null;state.accepted=null;state.preview=null;
    await load();await history();
  }
  async function action(name,loreUri){
    if(state.busy)return;
    if(name==='discard'){state.draft=null;state.generated=null;state.receipt=null;state.accepted=null;state.preview=null;state.history=[];state.dirty=false;state.error='';paint();return;}
    if(name.startsWith('new-')){
      if(state.dirty){state.error='Save or discard the current editor first.';paint();return;}
      state.draft={schema:1,uri:'fyodor://writing/'+(name==='new-note'?'notes/':name==='new-character'?'characters/':name==='new-project'?'projects/':'documents/')+crypto.randomUUID(),title:'Untitled',content:'',metadata:writing&&state.folder?{writing_parent:state.folder}:{},provenance:{}};
      state.revision='0';state.generated=null;state.receipt=null;state.accepted=null;state.preview=null;state.history=[];state.historyNext='0';state.dirty=true;state.error='';paint();return;
    }
    const destination=name==='move'?root.querySelector('#writing-parent')?.value:null;
    await run(async()=>{
      if(name==='lore-browse'||name==='lore-next'){
        const page=await backend.resources(state.namespace,name==='lore-next'?state.loreNext:'','lore');
        state.loreItems=page.resources;state.loreNext=page.next;state.loreOpen=true;
      }
      else if(name==='lore-add'||name==='lore-remove'){
        if(state.dirty)throw Error('Save or discard edits before changing lore references.');
        const links=writingLoreUris(state.draft);
        const next=name==='lore-add'?[...links,loreUri]:links.filter(uri=>uri!==loreUri);
        await backend.setWritingLore(state.namespace,state.draft.uri,state.revision,next);
        await reloadDraft();state.loreOpen=true;state.notice=name==='lore-add'?'Lore linked.':'Lore unlinked.';
      }
      else if(name==='lore-grant'){
        let changed=0;
        try{for(const uri of [state.draft.uri,...writingLoreUris(state.draft)]){
          await backend.contextPermissions(state.namespace,contextIdentity(),uri,1);++changed;
        }}catch(error){throw Error(`${changed} grants changed before the operation stopped: ${error.message}`);}
        state.notice='Model access allowed for this resource and its linked lore.';
      }
      else if(name==='move'){
        if(state.dirty)throw Error('Save or discard edits before moving this resource.');
        await backend.moveWriting(state.namespace,state.draft.uri,state.revision,destination);
        await reloadDraft();state.notice='Resource moved.';
      }
      else if(name==='writing-grant'){
        let changed=0;
        try{for(const uri of [state.draft.uri,...state.context]){await backend.contextPermissions(state.namespace,contextIdentity(),uri,1);++changed;}}
        catch(error){throw Error(`${changed} grants changed before the operation stopped: ${error.message}`);}
        state.notice=`${changed} source grants set to read only.`;
      }
      else if(['writing-generate','writing-rewrite','writing-continue'].includes(name)){
        if(state.dirty)throw Error('Save edits before generating from this resource.');
        state.generated=null;state.receipt=null;state.accepted=null;
        const mode=name.slice(8),target=state.draft.uri,revision=state.revision,principal=contextIdentity();
        const result=await backend.writingGenerate({namespace:state.namespace,principal,model_id:modelId,target,expected_revision:revision,mode,
          resources:[...state.context],layers:[...state.context].map(()=>'workspace'),prompt:writingPrompt(state.instructions,state.guidance.get(writingGuidanceKey(state.namespace,target))),byte_budget:state.budget,max_tokens:state.tokens});
        state.generated={...result,mode,target,revision,principal};
        state.receipt=await backend.contextReceipt(state.namespace,principal,result.receipt_id);
        state.notice='Preview generated and receipt saved. Editor text is unchanged.';
      }
      else if(name==='writing-apply'&&state.generated){
        if(state.dirty||state.draft.uri!==state.generated.target||state.revision!==state.generated.revision)throw Error('The editor changed after generation. Review your edits before applying a preview.');
        state.draft.content=state.generated.mode==='continue'?state.draft.content+state.generated.text:state.generated.text;
        state.accepted={receipt_id:state.generated.receipt_id,principal:state.generated.principal};
        state.dirty=true;state.notice='Preview copied to the unsaved editor. Save to add a revision.';
      }
      else if(name==='history')await history();
      else if(name==='reload'&&state.draft){await reloadDraft();state.dirty=false;state.notice='Saved version loaded.';}
      else if((name==='export-markdown'||name==='export-text')&&state.draft){downloadWritingText(state.draft,name==='export-markdown'?'md':'txt');state.notice='Current draft exported.';}
      else if(name==='history-next')await history(state.historyNext);
      else if(name==='copy-history'&&state.preview){if(state.dirty)throw Error('Save or discard your edits before copying a revision.');state.draft.title=state.preview.resource.title;state.draft.content=state.preview.resource.content;state.dirty=true;state.notice='Historical title and text copied to the unsaved editor.';}
      else if(name==='refresh')await load();
      else if(name==='next')await load(state.next);
      else if(name==='save'&&state.draft){const result=await (state.revision==='0'?backend.saveResource:backend.updateResource)(state.namespace,state.draft,state.revision,state.accepted);state.revision=result.revision;state.accepted=null;state.dirty=false;await load();await history();state.notice='Saved to the local workspace.';}
      else if(name==='export'&&state.draft){if(state.dirty)throw Error('Save the current editor before exporting.');const text=await backend.exportResource(state.namespace,state.draft.uri);const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fyodor-resource.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
      else if(name==='delete'&&state.draft){if(state.dirty)throw Error('Save or discard unsaved edits before deleting.');await backend.deleteResource(state.namespace,state.draft.uri,state.revision);state.draft=null;state.dirty=false;await load();state.notice='Resource deleted; revision history is retained.';}
    });
  }
  repaint=paint;paint();if(!state.busy)void run(()=>load());
}
