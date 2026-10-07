// End-to-end browser -> authenticated C HTTP -> SQLite -> separate CLI process.
import {spawn,execFileSync} from 'node:child_process';
import {mkdtemp,readFile,rm,copyFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const folder=await mkdtemp(path.join(tmpdir(),'fyodor-resource-browser-'));
const database=path.join(folder,'workspace.db');let vite,server,token,base;
try{
  const modelPath=path.join(folder,'pretraining.trained.gguf');
  await copyFile(path.resolve('build-cpu/backend/test_models/pretraining.trained.gguf'),modelPath);
  server=spawn(path.resolve('build-cpu/fyodor-backend.exe'),['--port','0'],{cwd:folder,env:{...process.env,FYODOR_STORE_PATH:database,NYA_COMPUTE:'cpu'},windowsHide:true,stdio:['ignore','pipe','ignore']});
  const ready=await new Promise((resolve,reject)=>{let text='';const timer=setTimeout(()=>reject(Error('Backend timeout')),10000);server.stdout.on('data',part=>{text+=part;if(text.includes('\n')){clearTimeout(timer);resolve(text.split('\n')[0].trim().split(' '));}});server.once('error',reject);});
  assert.equal(ready[0],'FYODOR_READY');base='http://127.0.0.1:'+ready[2];token=ready[3];
  const loaded=await fetch(base+'/api/v1/model/load',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({path:modelPath})});
  assert.ok(loaded.ok);assert.ok((await loaded.json()).model.generation_supported);
  const comparisonPath=path.join(folder,'comparison.trained.gguf');await copyFile(modelPath,comparisonPath);
  const comparison=await fetch(base+'/api/v1/model/load',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({path:comparisonPath})});
  assert.ok(comparison.ok);assert.ok((await comparison.json()).model.generation_supported);
  vite=spawn(process.execPath,[path.resolve('frontend/node_modules/vite/bin/vite.js'),'--host','127.0.0.1','--port','5179','--strictPort'],{cwd:path.resolve('frontend'),windowsHide:true,env:{...process.env,VITE_FYODOR_URL:base,VITE_FYODOR_TOKEN:token},stdio:'ignore'});
  for(let i=0;i<100;++i){try{if((await fetch('http://127.0.0.1:5179/')).ok)break;}catch{}await new Promise(r=>setTimeout(r,50));}
  const evidence=path.join(folder,'evidence.json');
  const child=spawn(process.execPath,['frontend/tests/theme-browser.mjs'],{windowsHide:true,env:{...process.env,FYODOR_PREVIEW:'http://localhost:5179/',FYODOR_RESOURCE_QA:evidence},stdio:'inherit'});
  assert.equal(await new Promise(resolve=>child.once('exit',resolve)),0);
  const {uri,principal,receipt_id,writing_uri,writing_receipt,project_uri,lore_uri,world_uri,imported_uri,dataset_uri,derived_uri,corpus_uri,chat_dataset_uri,context_dataset_uri,evaluation_definition_uri,evaluation_run_uri,evaluation_source_uri,evaluation_case_dataset_uri,evaluation_imported_dataset_uri,evaluation_mapped_definition_uri,evaluation_mapped_run_uri}=JSON.parse(await readFile(evidence,'utf8'));
  const record=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(record.title,'Browser document');assert.equal(record.content,'<script>inert</script> saved from desktop');
  const receipt=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','context','receipt',principal,receipt_id],{windowsHide:true,encoding:'utf8'}));
  assert.equal(receipt.prompt,'<sc\n\na');assert.equal(receipt.sources[0].uri,uri);
  const written=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',writing_uri],{windowsHide:true,encoding:'utf8'}));
  assert.deepEqual(written.provenance.writing_generations,[{receipt_id:writing_receipt,source_revision:'1',saved_revision:'2',mode:'continue',text_edited:false}]);
  assert.equal(written.metadata.writing_parent,project_uri);
  assert.deepEqual(written.metadata.writing_lore,[lore_uri]);
  for(const [uri,title,content] of [[world_uri,'Atlas','An independently edited world.'],[lore_uri,'The glass harbor','A harbor built from sea glass.']]){
    const resource=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',uri],{windowsHide:true,encoding:'utf8'}));
    assert.equal(resource.title,title);assert.equal(resource.content,content);
  }
  const imported=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',imported_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(imported.content,'Newer saved text.');assert.equal(imported.metadata.writing_parent,project_uri);
  const derived=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',derived_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(derived.metadata.dataset_studio.mode,'sft');assert.equal(derived.content,'aa\tb\nbb\tc\ncc\ta');
  assert.equal(derived.provenance.dataset_studio.source.uri,dataset_uri);assert.equal(derived.provenance.dataset_studio.source.revision,'1');
  execFileSync(path.resolve('build-cpu/fyodor-train.exe'),['--mode','sft','--base',modelPath,'--rank','0','--steps','1','--threads','1','--data',evidence+'.dataset.tsv','--output',path.join(folder,'dataset-studio-trained.gguf')],{windowsHide:true,encoding:'utf8',stdio:'pipe'});
  console.log('Dataset Studio export accepted by the native SFT trainer');
  const corpus=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',corpus_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(corpus.content,'Independent edit.\n\nA harbor built from sea glass.');
  assert.deepEqual(corpus.provenance.dataset_studio.sources.map(({uri,revision})=>({uri,revision})),[{uri:imported_uri,revision:'2'},{uri:lore_uri,revision:'1'}]);
  assert.equal(corpus.provenance.dataset_studio.output_sha256,createHash('sha256').update(corpus.content).digest('hex'));
  const chatData=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',chat_dataset_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(chatData.content,'a b\tc');assert.deepEqual(chatData.provenance.dataset_studio.sources[0].message_pairs,[[0,1]]);
  for(const [mode,file] of [['cpt',evidence+'.source.txt'],['sft',evidence+'.chat.tsv']]){
    execFileSync(path.resolve('build-cpu/fyodor-train.exe'),['--mode',mode,'--base',modelPath,'--rank','0','--steps','1','--threads','1','--data',file,'--output',path.join(folder,mode+'-source-trained.gguf')],{windowsHide:true,encoding:'utf8',stdio:'pipe'});
  }
  console.log('Writing/Explore CPT and local-chat SFT exports completed native training updates');
  const contextData=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',context_dataset_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(contextData.content,'<sc');
  const contextOrigin=contextData.provenance.dataset_studio.sources[0];
  assert.equal(contextOrigin.receipt_id,receipt_id);assert.equal(contextOrigin.principal,principal);assert.equal(contextOrigin.uri,uri);assert.equal(contextOrigin.revision,'3');
  assert.equal(contextOrigin.receipt_source.length,3);assert.ok(contextOrigin.receipt_source.original_length>3);
  execFileSync(path.resolve('build-cpu/fyodor-train.exe'),['--mode','cpt','--base',modelPath,'--rank','0','--steps','1','--threads','1','--data',evidence+'.context.txt','--output',path.join(folder,'context-source-trained.gguf')],{windowsHide:true,encoding:'utf8',stdio:'pipe'});
  console.log('Permission-checked Context receipt source export completed a native CPT update');
  const evaluation=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',evaluation_definition_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(JSON.parse(evaluation.content).cases.length,3);
  const result=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',evaluation_run_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(result.title,'Independent title');assert.equal(result.metadata.evaluation_studio.kind,'run');
  const run=JSON.parse(result.content);assert.equal(run.definition.uri,evaluation_definition_uri);assert.equal(run.definition.revision,'1');assert.equal(run.models.length,2);
  assert.deepEqual(run.results.map(row=>row.status),['pass','pass','error','pass','unreviewed','error']);
  assert.equal(run.results[1].review_note,'<script>inert</script> reviewed');
  assert.deepEqual(run.results.map(row=>row.generated_tokens),[2,2,null,2,2,null]);
  console.log('Evaluation definitions, real two-model outputs, errors and reviewed results persist through independent native CLI export');
  const exportResource=uri=>JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',uri],{windowsHide:true,encoding:'utf8'}));
  const trainingSource=exportResource(evaluation_source_uri);assert.equal(trainingSource.content,'a\tAFTER');
  const testDataset=exportResource(evaluation_case_dataset_uri),testCases=JSON.parse(testDataset.content).cases;
  assert.equal(testDataset.metadata.evaluation_dataset.version,1);assert.equal(testCases.length,40);assert.equal(testCases[20].label,'<script>inert</script> test case');
  assert.ok(testCases.every(item=>item.prompt==='a'&&item.check.type==='exact'&&item.check.expected==='bc'));
  const testOrigin=testDataset.provenance.evaluation_studio;
  assert.equal(testOrigin.source.uri,evaluation_source_uri);assert.equal(testOrigin.source.revision,'1');assert.equal(testOrigin.mapping,'prompt-completion-exact');
  assert.equal(testOrigin.source.content_sha256,createHash('sha256').update(Array.from({length:40},()=> 'a\tbc').join('\n')).digest('hex'));
  assert.deepEqual(testOrigin.source_records,Array.from({length:40},(_,i)=>({line:i+1})));
  const initialCases=structuredClone(testCases);initialCases[20].label='Record 21';assert.equal(testOrigin.mapped_cases_sha256,createHash('sha256').update(JSON.stringify(initialCases)).digest('hex'));
  const importedCases=exportResource(evaluation_imported_dataset_uri);assert.ok(JSON.parse(importedCases.content).cases.every(item=>item.prompt==='b'));
  assert.equal(importedCases.provenance.evaluation_studio.operation,'case-file-import');
  const mapped=exportResource(evaluation_mapped_definition_uri),definition=JSON.parse(mapped.content),mapping=mapped.provenance.evaluation_studio;
  assert.equal(mapping.source.uri,evaluation_imported_dataset_uri);assert.equal(mapping.source.revision,'1');assert.equal(mapping.source.content_sha256,createHash('sha256').update(JSON.stringify({schema:1,kind:'evaluation-dataset',cases:testCases})).digest('hex'));
  assert.deepEqual(mapping.selection,{method:'sample',seed:7,count:2,algorithm:'mulberry32-fisher-yates-v1'});assert.ok(definition.cases.every(item=>item.prompt==='a'&&item.check.expected==='bc'));
  assert.deepEqual(definition.cases,mapping.source_records.map(row=>testCases[row.case_index]));
  const mappedRun=JSON.parse(exportResource(evaluation_mapped_run_uri).content);assert.equal(mappedRun.definition.uri,evaluation_mapped_definition_uri);assert.equal(mappedRun.definition.revision,'1');
  assert.deepEqual(mappedRun.results.map(row=>({status:row.status,text:row.text,generated_tokens:row.generated_tokens})),Array.from({length:4},()=>({status:'pass',text:'bc',generated_tokens:2})));
  console.log('Historical SFT -> reusable test dataset -> file interchange -> sampled definition -> real evaluation results verified through native CLI persistence');
  console.log('Quick switcher keyboard/focus, retained drafts, title search, exact revision inspection and stale-write guards passed');
  console.log('Resources/Explore/Writing revision comparisons, live draft focus, dirty-copy guards and exact saved reload passed');
  console.log('Desktop resource create/save/reload and native CLI export passed');
}finally{
  if(server&&base&&token)await fetch(base+'/api/v1/shutdown',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:'{}'}).catch(()=>{});
  for(const child of [vite,server])if(child&&child.exitCode===null){child.kill();await new Promise(resolve=>{child.once('exit',resolve);setTimeout(resolve,2000);});}
  assert.ok(path.resolve(folder).startsWith(path.resolve(tmpdir())+path.sep+'fyodor-resource-browser-'));
  await rm(folder,{recursive:true,force:true,maxRetries:5,retryDelay:100});
}
