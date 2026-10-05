import {test} from 'node:test';
import assert from 'node:assert/strict';
import {inspectDataset,decodeDataset,transformDataset,seededShuffle,savedDatasetMode,datasetPage,datasetLimit} from '../src/dataset-data.js';

test('Structured formats match native tab counts, blank-line and empty-completion rules',()=>{
  const good=inspectDataset('\tanswer\r\n\nquestion\tresponse\n','sft');
  assert.equal(good.issues.length,0);assert.equal(good.rows.length,2);assert.equal(good.rows[0].fields[0],'');
  assert.equal(inspectDataset('q\tchosen\trejected','dpo').issues.length,0);
  for(const [text,mode] of [['q\t','sft'],['q\ta\textra','sft'],['q\ta','dpo'],['q\t\tb','dpo'],[' ','sft']])assert.ok(inspectDataset(text,mode).issues.length);
  assert.ok(inspectDataset('\n\r\n','sft').issues.length);
  assert.equal(inspectDataset('q\t \t ','dpo').issues.length,0);
  assert.ok(inspectDataset('q\ta\n'.repeat(100000),'sft').issues.some(issue=>issue.line===0));
});
test('Corpus checks do not pretend to know model token counts or normalize original text',()=>{
  assert.equal(inspectDataset('x','pretrain').issues.length,0);
  assert.equal(inspectDataset('한글\r\nline','cpt').rows.length,2);
  assert.ok(inspectDataset('','pretrain').issues.length);
  assert.ok(inspectDataset('a\0b','pretrain').issues.length);
  assert.ok(inspectDataset('a'.repeat(datasetLimit+1),'pretrain').issues.length);
  assert.equal(inspectDataset('a\na\nb','pretrain').duplicates,1);
});
test('Deterministic transforms preserve complete records and split without overlap',()=>{
  const source='a\tA\nb\tB\nc\tC\nd\tD';
  assert.deepEqual(seededShuffle([1,2,3,4],42),seededShuffle([1,2,3,4],42));
  assert.equal(source,'a\tA\nb\tB\nc\tC\nd\tD');
  const split=transformDataset(source,'sft','split',{seed:42,validationPercent:25});
  assert.equal(split.training.split('\n').length,3);assert.equal(split.validation.split('\n').length,1);
  assert.deepEqual(new Set([...split.training.split('\n'),...split.validation.split('\n')]),new Set(source.split('\n')));
  assert.equal(transformDataset('A\ta\nA\ta\nb\tb','sft','deduplicate').content,'A\ta\nb\tb');
  assert.equal(transformDataset(source,'sft','filter',{query:'C'}).content,'c\tC');
  assert.equal(transformDataset(source,'sft','sample',{seed:1,count:2}).content.split('\n').length,2);
  assert.throws(()=>transformDataset(source,'sft','filter',{query:'missing'}),/empty/);
  assert.throws(()=>transformDataset('q\t','sft','shuffle',{seed:1}),/Fix format/);
  assert.throws(()=>transformDataset('a','cpt','split',{seed:42,validationPercent:20}),/two records/);
  assert.throws(()=>seededShuffle([1,2],-1),/Seed/);
});
test('Dataset imports are strict UTF-8; mode metadata must be explicitly supported',()=>{
  assert.equal(decodeDataset(new TextEncoder().encode('\ufeffq\ta\r\n'),'sample.tsv'),'q\ta\r\n');
  assert.throws(()=>decodeDataset(new Uint8Array([0xff]),'invalid.txt'),/UTF-8/);
  assert.throws(()=>decodeDataset(new Uint8Array([0]),'invalid.tsv'),/null/);
  assert.equal(savedDatasetMode({metadata:{dataset_studio:{version:1,mode:'sft'}}}),'sft');
  assert.equal(savedDatasetMode({metadata:{dataset_studio:{version:2,mode:'sft'}}}),null);
  assert.equal(savedDatasetMode({metadata:{dataset_studio:{version:1,mode:'constructor'}}}),null);
});
test('Catalog filtering continues through sparse pages and bounds scans',async()=>{
  let count=0;const uri='fyodor://datasets/11111111-2222-4333-8444-555555555555';
  const api={resources:async()=>++count<3?{resources:[],next:String(count)}:{resources:[{uri}],next:''}};
  assert.deepEqual(await datasetPage(api,'workspace'),{resources:[{uri}],next:''});
  assert.equal(count,3);
  count=0;assert.equal((await datasetPage({resources:async()=>({resources:[],next:String(++count)})},'workspace')).next,'25');
});
