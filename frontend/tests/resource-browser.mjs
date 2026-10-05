// End-to-end browser -> authenticated C HTTP -> SQLite -> separate CLI process.
import {spawn,execFileSync} from 'node:child_process';
import {mkdtemp,readFile,rm,copyFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
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
  vite=spawn(process.execPath,[path.resolve('frontend/node_modules/vite/bin/vite.js'),'--host','127.0.0.1','--port','5179','--strictPort'],{cwd:path.resolve('frontend'),windowsHide:true,env:{...process.env,VITE_FYODOR_URL:base,VITE_FYODOR_TOKEN:token},stdio:'ignore'});
  for(let i=0;i<100;++i){try{if((await fetch('http://127.0.0.1:5179/')).ok)break;}catch{}await new Promise(r=>setTimeout(r,50));}
  const evidence=path.join(folder,'evidence.json');
  const child=spawn(process.execPath,['frontend/tests/theme-browser.mjs'],{windowsHide:true,env:{...process.env,FYODOR_PREVIEW:'http://localhost:5179/',FYODOR_RESOURCE_QA:evidence},stdio:'inherit'});
  assert.equal(await new Promise(resolve=>child.once('exit',resolve)),0);
  const {uri,principal,receipt_id,writing_uri,writing_receipt,project_uri,lore_uri,world_uri,imported_uri,dataset_uri,derived_uri}=JSON.parse(await readFile(evidence,'utf8'));
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
  assert.equal(imported.content,'Independent edit.');assert.equal(imported.metadata.writing_parent,project_uri);
  const derived=JSON.parse(execFileSync(path.resolve('build-cpu/fyodor.exe'),['--store',database,'--namespace','workspace','resource','export',derived_uri],{windowsHide:true,encoding:'utf8'}));
  assert.equal(derived.metadata.dataset_studio.mode,'sft');assert.equal(derived.content,'aa\tb\nbb\tc\ncc\ta');
  assert.equal(derived.provenance.dataset_studio.source.uri,dataset_uri);assert.equal(derived.provenance.dataset_studio.source.revision,'1');
  execFileSync(path.resolve('build-cpu/fyodor-train.exe'),['--mode','sft','--base',modelPath,'--rank','0','--steps','1','--threads','1','--data',evidence+'.dataset.tsv','--output',path.join(folder,'dataset-studio-trained.gguf')],{windowsHide:true,encoding:'utf8',stdio:'pipe'});
  console.log('Dataset Studio export accepted by the native SFT trainer');
  console.log('Desktop resource create/save/reload and native CLI export passed');
}finally{
  if(server&&base&&token)await fetch(base+'/api/v1/shutdown',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:'{}'}).catch(()=>{});
  for(const child of [vite,server])if(child&&child.exitCode===null){child.kill();await new Promise(resolve=>{child.once('exit',resolve);setTimeout(resolve,2000);});}
  assert.ok(path.resolve(folder).startsWith(path.resolve(tmpdir())+path.sep+'fyodor-resource-browser-'));
  await rm(folder,{recursive:true,force:true,maxRetries:5,retryDelay:100});
}
