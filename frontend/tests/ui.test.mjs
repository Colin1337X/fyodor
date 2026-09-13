import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readCompletionStream} from '../src/stream.js';
import {messageBody} from '../src/message.js';
import {parseBenchmark, benchmarkMarkup} from '../src/benchmark.js';
import {formatConversation, normalizeChat, generationOptions} from '../src/session.js';

test('Benchmark resources identify CUTLASS and avoid counting library scratch twice',()=>{
  const record={schema_version:1,backend:'cuda',model:'model.gguf',prefill_implementation:'cutlass-3xtf32',
    device_weights_bytes:1048576,device_kv_bytes:2097152,device_scratch_bytes:4194304,external_matmul_bytes:1048576,
    results:[{test:'pp512',tokens_per_second:10,stddev:1,ms_per_token:100,cutlass_matmul_calls:3}]};
  const parsed=parseBenchmark(JSON.stringify(record)),html=benchmarkMarkup([parsed]);
  assert.match(html,/CUTLASS 3×TF32/);assert.match(html,/weights 1\.0 MiB · KV 2\.0 MiB · scratch 4\.0 MiB/);
  assert.throws(()=>parseBenchmark(JSON.stringify({...record,device_kv_bytes:-1})),/resource metadata/);
  assert.throws(()=>parseBenchmark(JSON.stringify({...record,engine:{toString:'bad'}})),/description/);
  assert.throws(()=>parseBenchmark(JSON.stringify({...record,results:[{...record.results[0],cutlass_matmul_calls:1.5}]})),/counter/);
});

function response(text, stride=1) {
  const bytes=new TextEncoder().encode(text);let offset=0;
  return new Response(new ReadableStream({pull(c){if(offset===bytes.length){c.close();return;}c.enqueue(bytes.slice(offset,offset+stride));offset=Math.min(bytes.length,offset+stride);}}));
}
test('SSE preserves UTF-8, CRLF and multiline records across byte boundaries',async()=>{
  let seen='';
  const text=': heartbeat\r\n\r\ndata: {"choices":[{"delta":{"content":"Hello 한글"}}]}\r\n\r\ndata: {"choices":[],\ndata: "usage":{"completion_tokens":4}}\n\ndata: [DONE]\n\n';
  assert.deepEqual(await readCompletionStream(response(text),s=>seen=s),{text:'Hello 한글',tokens:4});
  assert.equal(seen,'Hello 한글');
});
test('SSE accepts finish_reason at EOF and rejects truncated/malformed/error streams',async()=>{
  assert.deepEqual(await readCompletionStream(response('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'),()=>{}),{text:'ok',tokens:0});
  for(const source of ['data: {"choices":[{"delta":{"content":"partial"}}]}\n\n','data: {broken}\n\n','data: {"error":{"message":"failed"}}\n\n'])
    await assert.rejects(readCompletionStream(response(source),()=>{}));
});
test('SSE bounds unfinished records and rejects streamed tools',async()=>{
  await assert.rejects(readCompletionStream(response('x'.repeat(1_000_001),65536),()=>{}),/size limit/);
  await assert.rejects(readCompletionStream(response('data: {"choices":[{"delta":{"tool_calls":[{}]}}]}\n\n'),()=>{}),/tool calls/);
});
test('Model text stays inert inside and outside code fences',()=>{
  const html=messageBody('<img src=x onerror=alert(1)>\n```html\n</code><script>alert(1)</script>\n```');
  assert(!html.includes('<script>'));assert(!html.includes('<img'));assert(html.includes('&lt;/code&gt;'));
  assert(html.includes('data-copy-code'));assert(messageBody('```c\nint x;').includes('int x;'));
});
test('Benchmark parser accepts measured results and rejects invalid data',()=>{
  const record={schema_version:1,backend:'cuda',model:'<img>.gguf',results:[{test:'pp512',tokens_per_second:100,stddev:1,ms_per_token:10}]};
  assert.equal(parseBenchmark(JSON.stringify(record)).results[0].tokens_per_second,100);
  assert(!benchmarkMarkup([record]).includes('<img>'));
  for(const value of [{...record,backend:'fake'},{...record,results:[{...record.results[0],stddev:-1}]},null])assert.throws(()=>parseBenchmark(JSON.stringify(value)));
});
test('Conversation validation and inference controls remain bounded',()=>{
  assert.throws(()=>normalizeChat({messages:[{role:'system',text:'bad'}]},'id'));
  assert.throws(()=>generationOptions({temperature:8,top_p:1,top_k:1,max_tokens:4,seed:''}));
  assert(formatConversation([{role:'user',content:'hi'}],'gemma4').includes('hi'));
});
