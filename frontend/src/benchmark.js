import { escapeHtml as esc } from './message.js';
const providerNames=new Map([['native','Native'],['cublas-f32','cuBLAS F32'],['cutlass-3xtf32','CUTLASS 3×TF32']]);
// Import actual fyodor-bench output, never generate sample performance numbers.
// Bounded input and explicit schema checks keep untrusted files out of markup.
export function parseBenchmark(text) {
  if (text.length > 1024*1024) throw new Error('Benchmark files are limited to 1 MiB.');
  const data = JSON.parse(text);
  if (data?.schema_version !== 1 || !['cpu','cuda','vulkan'].includes(data.backend) ||
      typeof data.model !== 'string' || !Array.isArray(data.results) || data.results.length > 32) throw new Error('Expected fyodor-bench JSON schema 1.');
  for (const r of data.results) {
    if (typeof r.test !== 'string' || !Number.isFinite(r.tokens_per_second) || r.tokens_per_second < 0 ||
        !Number.isFinite(r.stddev) || r.stddev < 0 || !Number.isFinite(r.ms_per_token) || r.ms_per_token < 0) throw new Error('Invalid benchmark measurement.');
    for (const key of ['prefix_tokens','cutlass_matmul_calls','external_matmul_calls','tiled_attention_calls'])
      if (r[key] !== undefined && (!Number.isSafeInteger(r[key]) || r[key] < 0)) throw new Error('Invalid benchmark counter.');
  }
  for (const key of ['device_weights_bytes','device_kv_bytes','device_scratch_bytes','prefill_batch','batch'])
    if (data[key] !== undefined && (!Number.isSafeInteger(data[key]) || data[key] < 0)) throw new Error('Invalid benchmark resource metadata.');
  for (const key of ['engine','machine','execution','prefill_implementation','kv_type'])
    if (data[key] !== undefined && typeof data[key] !== 'string') throw new Error('Invalid benchmark description.');
  return data;
}
function providerLabel(d) {
  return providerNames.get(d.prefill_implementation) || d.engine || 'Provider not reported';
}
function memoryLabel(d) {
  const fields=[['device_weights_bytes','weights'],['device_kv_bytes','KV'],['device_scratch_bytes','scratch']];
  if (!fields.every(([key])=>Number.isSafeInteger(d[key]))) return '';
  // External-library scratch is already included; do not double-count it.
  return `<p class="muted">Device allocations: ${fields.map(([key,label])=>`${label} ${(d[key]/1048576).toFixed(1)} MiB`).join(' · ')}</p>`;
}
export function benchmarkMarkup(records) {
  return `<div class="workspace-view benchmark-view"><section class="workspace-intro"><div><label class="eyebrow">MEASURE</label><h1>Benchmarks</h1><p>Inspect measured inference throughput from fyodor-bench.</p></div><label class="secondary file-button">Import result<input id="benchmark-file" type="file" accept=".json" hidden></label></section>
    <section class="panel"><h2>Run on your machine</h2><pre>fyodor-bench -m model.gguf -b cuda -p 512 -n 128 -r 3 --warmup 1 --json</pre><p class="muted">Run the CLI, save its JSON output, then import it here. Model loading and tokenization are excluded. Compare matching models, batches, context, thread counts and KV precision.</p></section>
    ${records.map(d => `<section class="panel benchmark-result"><header><div><label>${esc(d.backend.toUpperCase())} · ${esc(d.execution || d.engine || '')}</label><h2>${esc(d.model.split(/[\\/]/).pop())}</h2></div><span class="provider-tag">${esc(d.repetitions)} repetitions</span></header><p class="muted">${esc(d.machine || 'Machine not reported')} · batch ${esc(d.prefill_batch ?? d.batch ?? 'not reported')} · ${esc(d.kv_type || 'f32')} KV · ${esc(providerLabel(d))}</p>${memoryLabel(d)}<div class="table-scroll"><table><thead><tr><th>Workload</th><th>Tokens/s</th><th>Std. dev.</th><th>ms/token</th><th>Prefix</th></tr></thead><tbody>${d.results.map(r => `<tr><td>${esc(r.test)}</td><td>${r.tokens_per_second.toFixed(2)}</td><td>± ${r.stddev.toFixed(2)}</td><td>${r.ms_per_token.toFixed(3)}</td><td>${esc(r.prefix_tokens || 0)}</td></tr>`).join('')}</tbody></table></div></section>`).join('')}
    ${records.length ? '' : '<p class="quiet-empty">No measurements imported. Results stay in this window.</p>'}</div>`;
}
