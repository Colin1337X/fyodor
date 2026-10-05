import {backend} from './api.js';
import {escapeHtml as esc} from './message.js';
import {datasetModes,datasetLimit,isDataset,savedDatasetMode,inspectDataset,decodeDataset,transformDataset,newDataset,datasetPage} from './dataset-data.js';
import {writingDownloadName} from './writing-files.js';

const state = {namespace:'workspace',items:[],next:'',draft:null,revision:'0',mode:'pretrain',originalMode:null,dirty:false,busy:false,error:'',notice:'',query:'',seed:42,count:10,validationPercent:20,parts:[]};
let mount=0,repaint=()=>{};
window.addEventListener('beforeunload',event=>{if(state.dirty||state.parts.some(part=>!part.revision)){event.preventDefault();event.returnValue='';}});

export function datasetWorkspace(root, prepareTraining = null) {
  const generation=++mount;
  const current=()=>generation===mount&&document.body.dataset.view==='datasets';
  function paint(){
    if(!current())return;
    const d=state.draft,report=d?inspectDataset(d.content,state.mode):null;
    root.innerHTML=`<section class="resource-workspace dataset-workspace"><header class="view-heading"><div><p class="eyebrow">DATASET STUDIO</p><h1>Datasets</h1><p class="muted">Shape your training data and keep its source intact.</p></div></header>
      <div class="resource-toolbar"><label>Namespace<input id="dataset-namespace" maxlength="64" value="${esc(state.namespace)}"></label><button data-dataset="refresh">Refresh</button><button data-dataset="new">New dataset</button><button data-dataset="import">Import text / TSV</button><input id="dataset-import" type="file" accept=".txt,.tsv" hidden></div>
      <p role="status" aria-live="polite" class="${state.error?'danger-action':'muted'}">${esc(state.error||state.notice||(state.busy?'Working…':''))}</p>
      <div class="resource-columns"><section aria-label="Dataset library"><ul class="resource-list">${state.items.map(item=>`<li><button data-dataset-uri="${esc(item.uri)}" aria-pressed="${d?.uri===item.uri}"><b>${esc(item.title||'Untitled dataset')}</b><small>Revision ${esc(item.revision)}</small></button></li>`).join('')||'<li class="muted">No datasets on this page.</li>'}</ul><button data-dataset="next" ${state.next?'':'disabled'}>Next page</button></section>
      <section class="panel resource-editor" aria-label="Dataset editor">${d?`<p class="fine-print">${state.revision==='0'?'New dataset':'Revision '+esc(state.revision)}${state.dirty?' · Unsaved changes':''}</p><label>Title<input id="dataset-title" maxlength="1024" value="${esc(d.title)}"></label>
        <label>Training format<select id="dataset-mode">${Object.entries(datasetModes).map(([value,label])=>`<option value="${value}" ${state.mode===value?'selected':''}>${label}</option>`).join('')}</select></label>
        <div class="resource-toolbar"><button class="primary" data-dataset="save">Save</button><button data-dataset="copy">Create copy</button><button data-dataset="discard">Discard editor</button>${state.revision!=='0'?'<button data-dataset="reload">Reload saved version</button><button data-dataset="package">Export package</button>':''}<button data-dataset="export">Export training file</button>${prepareTraining?'<button data-dataset="training">Export & open Training</button>':''}</div>
        <label>${report.structured?'Tab-separated records':'Corpus text'}<textarea id="dataset-content" rows="14" spellcheck="false">${esc(d.content)}</textarea></label><div id="dataset-inspection"></div>`:'<h2>Build a dataset</h2><p class="muted">Create one or import your training text.</p>'}</section></div>
      ${d?`<section class="panel dataset-tools"><h2>Create a derived copy</h2><div class="resource-toolbar"><label>Filter text<input id="dataset-query" value="${esc(state.query)}"></label><label>Seed<input id="dataset-seed" type="number" min="0" max="4294967295" value="${state.seed}"></label><label>Sample records<input id="dataset-count" type="number" min="1" value="${state.count}"></label><label>Validation %<input id="dataset-percent" type="number" min="1" max="99" value="${state.validationPercent}"></label></div>
        <div class="resource-toolbar">${[['filter','Keep filtered records'],['sample','Sample'],['shuffle','Shuffle'],['deduplicate','Remove exact duplicates'],['split','Train / validation split']].map(([key,label])=>`<button data-dataset-transform="${key}">${label}</button>`).join('')}</div>
        <p class="muted">${report.structured?'Structured records':'Nonempty text lines'} are the units for these operations. Save the source first.</p>
        ${state.parts.length?`<h3>Train / validation copies</h3><ul class="dataset-parts">${state.parts.map((part,index)=>`<li><b>${esc(part.resource.title)}</b><span>${part.revision?'Saved · Revision '+esc(part.revision):'Not saved'}</span><button data-dataset-part="${index}" ${part.revision?'disabled':''}>Save ${part.label}</button></li>`).join('')}</ul>`:''}
      </section>`:''}</section>`;
    root.querySelectorAll('[data-dataset]').forEach(button=>button.addEventListener('click',()=>void action(button.dataset.dataset)));
    root.querySelectorAll('[data-dataset-uri]').forEach(button=>button.addEventListener('click',()=>void run(async()=>{requireClean();await open(button.dataset.datasetUri);})));
    root.querySelectorAll('[data-dataset-transform]').forEach(button=>button.addEventListener('click',()=>void derive(button.dataset.datasetTransform)));
    root.querySelectorAll('[data-dataset-part]').forEach(button=>button.addEventListener('click',()=>void run(async()=>{
      const part=state.parts[Number(button.dataset.datasetPart)];
      if(part.revision)return;
      const saved=await backend.saveResource(state.namespace,part.resource,'0');part.revision=saved.revision;await load();state.notice=`${part.label} copy saved.`;
    })));
    for(const key of ['title','content'])root.querySelector('#dataset-'+key)?.addEventListener('input',event=>{
      state.draft[key]=event.target.value;state.dirty=true;
      root.querySelector('.resource-editor .fine-print').textContent=(state.revision==='0'?'New dataset':'Revision '+state.revision)+' · Unsaved changes';
      inspect();
    });
    root.querySelector('#dataset-mode')?.addEventListener('change',event=>{state.mode=event.target.value;state.dirty=true;paint();});
    root.querySelector('#dataset-query')?.addEventListener('input',event=>{state.query=event.target.value;inspect();});
    for(const [id,key,min,max] of [['dataset-seed','seed',0,0xffffffff],['dataset-count','count',1,100000],['dataset-percent','validationPercent',1,99]])root.querySelector('#'+id)?.addEventListener('change',event=>{
      const value=Number(event.target.value);if(Number.isInteger(value)&&value>=min&&value<=max)state[key]=value;else event.target.value=state[key];
    });
    root.querySelector('#dataset-namespace').addEventListener('change',event=>{
      const namespace=event.target.value;void run(async()=>{requireClean();if(!/^[A-Za-z0-9_-]{1,64}$/.test(namespace))throw Error('Use 1–64 letters, digits, underscores or hyphens.');state.namespace=namespace;clear();await load();});
    });
    root.querySelector('#dataset-import').addEventListener('change',event=>{
      const file=event.target.files[0];if(!file)return;void run(async()=>{
        requireClean();if(file.size>datasetLimit)throw Error('Studio imports must be 1 MiB or smaller.');
        const content=decodeDataset(new Uint8Array(await file.arrayBuffer()),file.name);
        clear();state.draft=newDataset(state.mode,file.name.replace(/\.(txt|tsv)$/i,'').slice(0,250)||'Imported dataset',content,{dataset_studio:{version:1,operation:'file-import',filename:file.name}});
        state.dirty=true;state.notice='Imported into a new dataset draft. Review the format and save.';
      });
    });
    inspect();
    if(state.busy)root.querySelectorAll('button,input,textarea,select').forEach(control=>control.disabled=true);
  }

  function inspect(){
    const target=root.querySelector('#dataset-inspection');if(!target||!state.draft)return;
    const report=inspectDataset(state.draft.content,state.mode),matches=report.rows.filter(row=>row.raw.toLowerCase().includes(state.query.toLowerCase()));
    target.innerHTML=`<div class="dataset-stats"><span>${report.rows.length.toLocaleString()} ${report.structured?'records':'nonempty lines'}</span><span>${report.bytes.toLocaleString()} bytes</span><span>${report.duplicates.toLocaleString()} exact duplicates</span></div>
      <h3>Format check</h3>${report.issues.length?`<ul class="dataset-issues">${report.issues.slice(0,30).map(issue=>`<li>${issue.line?'Line '+issue.line+': ':''}${esc(issue.message)}</li>`).join('')}</ul>${report.issues.length>30?`<p>${report.issues.length-30} more issues.</p>`:''}`:'<p class="muted">Format checked. Model tokenization and context fit are checked when training.</p>'}
      <details class="dataset-records"><summary>Inspect records · ${matches.length} match${matches.length===1?'':'es'}</summary><div class="dataset-table"><table><thead><tr><th>Line</th>${(report.structured?(state.mode==='sft'?['Prompt','Completion']:['Prompt','Chosen','Rejected']):['Text']).map(label=>`<th>${label}</th>`).join('')}</tr></thead><tbody>${matches.slice(0,50).map(row=>`<tr><td>${row.line}</td>${row.fields.map(field=>`<td>${esc(field)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>${matches.length>50?'<p class="muted">Showing the first 50 matches.</p>':''}</details>`;
  }
  function requireClean(){if(state.dirty||state.parts.some(part=>!part.revision))throw Error('Save or discard the current draft and split copies first.');}
  function clear(){Object.assign(state,{draft:null,revision:'0',originalMode:null,dirty:false,parts:[]});}
  async function load(after=''){const page=await datasetPage(backend,state.namespace,after);state.items=page.resources;state.next=page.next;}
  async function open(uri){const result=await backend.resource(state.namespace,uri);if(!isDataset(result.resource.uri))throw Error('Choose a dataset resource.');clear();state.draft=result.resource;state.revision=result.revision;state.originalMode=savedDatasetMode(result.resource);state.mode=state.originalMode||'pretrain';}
  async function run(work){if(state.busy)return;state.busy=true;state.error='';state.notice='';paint();try{await work();}catch(error){state.error=error.message;}finally{state.busy=false;repaint();}}
  const source = () => ({uri:state.draft.uri,revision:state.revision,namespace:state.namespace});
  async function derive(operation){await run(async()=>{
    requireClean();if(state.revision==='0')throw Error('Save the source dataset first.');
    const parameters={query:state.query,seed:state.seed,count:state.count,validationPercent:state.validationPercent};
    const result=transformDataset(state.draft.content,state.mode,operation,parameters);
    const provenance={dataset_studio:{version:1,operation,source:source(),parameters,units:state.mode==='sft'||state.mode==='dpo'?'records':'nonempty-lines'}};
    if(operation==='split'){
      state.parts=['training','validation'].map(label=>({label,revision:null,resource:newDataset(state.mode,state.draft.title+' · '+label,result[label],provenance)}));
      state.notice='Split copies prepared. Save each copy to keep it.';
    }else{
      state.draft=newDataset(state.mode,state.draft.title+' · '+operation,result.content,provenance);state.revision='0';state.originalMode=null;state.dirty=true;
      state.notice='Derived draft prepared. The source is unchanged.';
    }
  });}
  async function action(name){
    if(name==='import'){if(!state.busy)root.querySelector('#dataset-import').click();return;}
    await run(async()=>{
      if(name==='new'){requireClean();clear();state.draft=newDataset(state.mode);state.dirty=true;}
      else if(name==='discard')clear();
      else if(name==='refresh')await load();
      else if(name==='next'&&state.next)await load(state.next);
      else if(name==='reload'){if(state.parts.some(part=>!part.revision))throw Error('Save or discard split copies before reloading.');await open(state.draft.uri);state.notice='Saved version loaded.';}
      else if(name==='copy'){
        if(state.parts.some(part=>!part.revision))throw Error('Save or discard split copies first.');
        const provenance=state.revision==='0'?state.draft.provenance:{dataset_studio:{version:1,operation:'copy',source:source(),edited:state.dirty}};
        state.draft=newDataset(state.mode,state.draft.title+' · copy',state.draft.content,provenance);state.revision='0';state.originalMode=null;state.dirty=true;
      }else if(name==='save'){
        if(state.revision!=='0'&&state.mode!==state.originalMode)throw Error('Create a copy to change the saved training format.');
        if(state.revision==='0')state.draft.metadata={dataset_studio:{version:1,mode:state.mode}};
        const saved=await (state.revision==='0'?backend.saveResource:backend.updateResource)(state.namespace,state.draft,state.revision);
        state.revision=saved.revision;state.originalMode=state.mode;state.dirty=false;await load();state.notice='Dataset saved.';
      }else if(name==='package'){
        if(state.dirty)throw Error('Save the dataset before exporting its package.');
        download(await backend.exportResource(state.namespace,state.draft.uri),'json','application/json');
      }else if(name==='export'||name==='training'){
        if(inspectDataset(state.draft.content,state.mode).issues.length)throw Error('Fix format issues before exporting training data.');
        download(state.draft.content,state.mode==='sft'||state.mode==='dpo'?'tsv':'txt','text/plain;charset=utf-8');state.notice='Current training data exported.';
        if(name==='training')prepareTraining({mode:state.mode,title:state.draft.title});
      }
    });
  }
  function download(text,extension,type){
    const url=URL.createObjectURL(new Blob([text],{type})),link=document.createElement('a');link.href=url;
    link.download=writingDownloadName(state.draft.title,'txt').replace(/\.txt$/,'.'+extension);link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  repaint=paint;paint();if(!state.busy)void run(()=>load());
}
