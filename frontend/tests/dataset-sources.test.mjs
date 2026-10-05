import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {datasetFromSources,resourceSource,chatSource,sourceKind,sourcePage} from '../src/dataset-sources.js';
import {datasetLimit,inspectDataset} from '../src/dataset-data.js';

const writing='fyodor://writing/documents/11111111-2222-4333-8444-555555555555';
const world='fyodor://explore/worlds/11111111-2222-4333-8444-555555555555';
const lore=world+'/lore/66666666-7777-4888-8999-aaaaaaaaaaaa';
const source=(uri=writing,revision='9007199254740993')=>resourceSource({uri,revision,title:'Source'},'workspace');
const sha=text=>createHash('sha256').update(text).digest('hex');
function transport(texts) {
  const reads=[];
  return {reads,resource:async(namespace,uri,revision)=>{
    reads.push({namespace,uri,revision});
    return {revision,deleted:false,resource:{uri,content:texts[uri]}};
  }};
}
test('Source ingestion reads pinned revisions, preserves exact corpus bytes and records provenance',async()=>{
  const api=transport({[writing]:'한글\r\nOne',[lore]:'Lore\ttext'});
  const sources=[source(),source(lore,'2')], originals=structuredClone(sources);
  const result=await datasetFromSources(api,sources,{mode:'cpt'});
  assert.equal(result.content,'한글\r\nOne\n\nLore\ttext');
  assert.deepEqual(api.reads,[{namespace:'workspace',uri:writing,revision:'9007199254740993'},{namespace:'workspace',uri:lore,revision:'2'}]);
  const provenance=result.provenance.dataset_studio;
  assert.equal(provenance.output_sha256,sha(result.content));
  assert.equal(provenance.sources[0].content_sha256,sha('한글\r\nOne'));
  assert.equal(provenance.sources[1].byte_start,new TextEncoder().encode('한글\r\nOne\n\n').length);
  assert.deepEqual(sources,originals);assert.equal(result.metadata.dataset_studio.mode,'cpt');
});
test('SFT mappings require explicit delimiter replacement and never manufacture DPO preferences',async()=>{
  const api=transport({[writing]:'First\r\nsecond\tthird'});
  await assert.rejects(datasetFromSources(api,[source()],{mode:'sft',prompt:'q'}),/replacement policy/);
  const result=await datasetFromSources(api,[source()],{mode:'sft',prompt:'p\nq',delimiterPolicy:'spaces'});
  assert.equal(result.content,'p q\tFirst second third');assert.equal(inspectDataset(result.content,'sft').issues.length,0);
  assert.equal(result.provenance.dataset_studio.resource_prompt,'p\nq');
  await assert.rejects(datasetFromSources(api,[source()],{mode:'dpo'}),/chosen\/rejected/);
});
test('Chat snapshots keep exact roles/text and only adjacent user/assistant pairs become SFT',async()=>{
  const chat={id:'local-id',title:'Chat',created:123,messages:[
    {role:'user',text:'a'},{role:'assistant',text:'A'},
    {role:'user',text:'b'},{role:'tool',text:'tool output'},{role:'assistant',text:'B'},
    {role:'user',text:'c\nline'},{role:'assistant',text:'C'}]};
  const selected=chatSource(chat);chat.messages[0].text='changed';
  const result=await datasetFromSources({},[selected],{mode:'sft',delimiterPolicy:'spaces'});
  assert.equal(result.content,'a\tA\nc line\tC');
  const origin=result.provenance.dataset_studio.sources[0];
  assert.deepEqual(origin.message_pairs,[[0,1],[5,6]]);assert.equal(origin.snapshot_sha256,sha(JSON.stringify(selected.messages)));
  assert.equal(origin.local_id,'local-id');assert.equal(Object.hasOwn(origin,'uri'),false);
  const corpus=await datasetFromSources({},[selected],{mode:'pretrain'});
  assert.ok(corpus.content.includes('tool:\ntool output'));assert.ok(!corpus.content.includes('changed'));
  await assert.rejects(datasetFromSources({},[chatSource({id:'empty',messages:[],title:'Empty'})]),/no messages/);
  await assert.rejects(datasetFromSources({},[chatSource({id:'tool',messages:[{role:'tool',text:'x'}]})],{mode:'sft'}),/no adjacent/);
});
test('Unavailable, oversized and duplicate sources fail without producing a replacement draft',async()=>{
  const item=source(),api=transport({[writing]:'a'});
  await assert.rejects(datasetFromSources(api,[item,item]),/more than once/);
  await assert.rejects(datasetFromSources(api,[]),/Select 1/);
  await assert.rejects(datasetFromSources(api,Array(33).fill(item)),/Select 1/);
  await assert.rejects(datasetFromSources({resource:async()=>({revision:'2',deleted:false,resource:{uri:writing,content:'new head'}})},[item]),/unavailable/);
  await assert.rejects(datasetFromSources({resource:async()=>({revision:item.revision,deleted:true})},[item]),/unavailable/);
  await assert.rejects(datasetFromSources(transport({[writing]:'a\0b'}),[item]),/Null/);
  await assert.rejects(datasetFromSources(transport({[writing]:'a'.repeat(datasetLimit+1)}),[item]),/1 MiB/);
  assert.throws(()=>resourceSource({uri:writing,revision:1},'workspace'),/saved/);
  assert.throws(()=>chatSource({id:'large',messages:[{role:'user',text:'x'.repeat(datasetLimit)},{role:'assistant',text:'x'}]}),/1 MiB/);
  await assert.rejects(datasetFromSources(api,[item],{mode:'sft',prompt:'q'.repeat(65536)}),/provenance/);
});
test('Typed source catalogs include Writing and Explore, bound sparse scans and reject repeated cursors',async()=>{
  assert.equal(sourceKind(lore),'explore');assert.equal(sourceKind(world+'/saves/66666666-7777-4888-8999-aaaaaaaaaaaa'),null);
  let count=0;
  const page=await sourcePage({resources:async(ns,after,scope)=>{
    assert.equal(scope,'all');count++;return count===3?{resources:[{uri:lore}],next:''}:{resources:[{uri:writing}],next:String(count)};
  }},'workspace','explore');
  assert.deepEqual(page,{resources:[{uri:lore}],next:''});assert.equal(count,3);
  count=0;assert.equal((await sourcePage({resources:async()=>({resources:[],next:String(++count)})},'workspace','writing')).next,'25');
  await assert.rejects(sourcePage({resources:async()=>({resources:[],next:'same'})},'workspace','writing'),/Repeated/);
});
test('Derived titles honor native UTF-8 limits without splitting characters',async()=>{
  const selected=source();selected.title='👋'.repeat(256);
  const result=await datasetFromSources(transport({[writing]:'text'}),[selected]);
  assert.equal(new TextEncoder().encode(result.title).length,1024);
  assert.equal(result.title,'👋'.repeat(256));
});
