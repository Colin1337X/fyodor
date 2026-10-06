import {test} from 'node:test';
import assert from 'node:assert/strict';
import {inspectContextReceipt,contextSource,contextSourceKey,readContextSource} from '../src/context-receipt.js';
import {contextReceiptSources} from '../src/context-receipt-view.js';
import {datasetFromSources} from '../src/dataset-sources.js';

const uri='fyodor://writing/documents/11111111-2222-4333-8444-555555555555';
const other='fyodor://writing/notes/66666666-7777-4888-8999-aaaaaaaaaaaa';
const principal='11111111-2222-4333-8444-555555555555';
const receiptId='66666666-7777-4888-8999-aaaaaaaaaaaa';
function receipt(prompt='한글\n\nUSER',sources=[{uri,revision:3,offset:0,length:6,original_length:10,layer:'explicit'}],contextBytes=6) {
  return {prompt,output:'OUTPUT MUST NOT ENTER DATASET',sources_json:JSON.stringify(sources),metadata_json:JSON.stringify({schema:1,context_bytes:contextBytes})};
}
test('Executed source ranges slice exact UTF-8 prefixes, preserve BOM and exclude user prompt/output',()=>{
  const inspected=inspectContextReceipt(receipt());
  assert.equal(inspected.sources[0].content,'한글');assert.equal(inspected.sources[0].revision,'3');assert.equal(inspected.sources[0].status,'truncated');
  assert.equal(inspectContextReceipt(receipt('\ufeffx\n\nUSER',[{uri,revision:1,offset:0,length:4,original_length:4,layer:'session'}],4)).sources[0].content,'\ufeffx');
  assert.throws(()=>inspectContextReceipt(receipt('é\nUSER',[{uri,revision:1,offset:0,length:1,original_length:2,layer:'explicit'}],1)),/UTF-8/);
  assert.throws(()=>inspectContextReceipt(receipt(undefined,[{uri,revision:3,offset:0,length:8,original_length:10,layer:'explicit'}],6)),/range/);
});
test('Receipt parsing keeps large native numeric revisions exact and rejects unsupported ranges/shapes',()=>{
  const big=receipt('abc',[],3);big.sources_json=`[{"uri":"${uri}","revision":9007199254740993,"offset":0,"length":3,"original_length":3,"layer":"explicit"}]`;
  assert.equal(inspectContextReceipt(big).sources[0].revision,'9007199254740993');
  for(const patch of [{offset:-1},{length:7},{original_length:2},{layer:'unknown'},{revision:0},{revision:1.5}]){
    const row={uri,revision:1,offset:0,length:3,original_length:3,layer:'explicit',...patch};
    assert.throws(()=>inspectContextReceipt(receipt('abc',[row],3)));
  }
  assert.throws(()=>inspectContextReceipt({...receipt(),metadata_json:'null'}),/boundary/);
  assert.throws(()=>inspectContextReceipt({...receipt(),sources_json:'{}'}),/sources/);
  assert.throws(()=>inspectContextReceipt(receipt('abc',[{uri,revision:1,offset:0,length:2,original_length:2,layer:'explicit'},{uri:other,revision:1,offset:1,length:1,original_length:1,layer:'workspace'}],3)),/range/);
});
test('Omitted/empty sources remain inspectable but cannot become dataset selections',()=>{
  const rows=inspectContextReceipt(receipt('',[{uri,revision:1,offset:0,length:0,original_length:3,layer:'retrieved'},{uri:other,revision:1,offset:0,length:0,original_length:0,layer:'global'}],0)).sources;
  assert.deepEqual(rows.map(row=>row.status),['omitted','empty']);
  assert.throws(()=>contextSource(rows[0],'workspace',principal,receiptId),/included/);
});
test('Context dataset preparation authorizes freshly and never calls administrative resource reads',async()=>{
  const saved=receipt('abc\n\ndef\n\nUSER',[{uri,revision:3,offset:0,length:3,original_length:30,layer:'session'},{uri:other,revision:9,offset:5,length:3,original_length:3,layer:'workspace'}],8);
  const sources=inspectContextReceipt(saved).sources.map(row=>contextSource(row,'workspace',principal,receiptId));
  let allowed=true,reads=0;
  const api={contextReceipt:async(ns,who,id)=>{reads++;assert.equal(ns,'workspace');assert.equal(who,principal);assert.equal(id,receiptId);if(!allowed)throw Error('Context access denied');return saved;},resource:()=>assert.fail('Administrative fallback must not be used')};
  const result=await datasetFromSources(api,sources,{mode:'cpt'});
  assert.equal(result.content,'abc\n\ndef');assert.equal(reads,1);
  const origin=result.provenance.dataset_studio.sources[0];
  assert.equal(origin.receipt_id,receiptId);assert.equal(origin.revision,'3');assert.equal(origin.receipt_source.original_length,30);
  assert.ok(result.provenance.dataset_studio.mapping.endsWith(';receipt-source-prefix'));
  allowed=false;await assert.rejects(datasetFromSources(api,sources,{mode:'cpt'}),/denied/);assert.equal(reads,2);
  allowed=true;assert.equal((await datasetFromSources(api,[sources[0]],{mode:'sft',prompt:'Question'})).content,'Question\tabc');
  await assert.rejects(readContextSource(api,{...sources[0],revision:'4'}),/unavailable/);
  assert.notEqual(contextSourceKey(sources[0]),contextSourceKey({...sources[0],receipt_id:principal}));
});
test('Readable source inspection escapes executed text and reports actual inclusion without invented causes',()=>{
  const text='<script>inert</script>';
  const html=contextReceiptSources(receipt(text,[{uri,revision:2,offset:0,length:text.length,original_length:text.length,layer:'explicit'}],text.length));
  assert.ok(html.includes('&lt;script&gt;inert&lt;/script&gt;'));assert.ok(!html.includes('<script>'));
  assert.ok(html.includes('Revision 2'));assert.ok(html.includes('Included'));
  assert.ok(contextReceiptSources(receipt()).includes('Prefix included'));
});
