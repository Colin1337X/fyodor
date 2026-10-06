// Human-readable attribution is an adapter over the native receipt, with the
// original source/metadata JSON retained for exact inspection and compatibility.
import {escapeHtml as esc} from './message.js';
import {inspectContextReceipt} from './context-receipt.js';
export function contextReceiptSources(receipt) {
  try {
    const report=inspectContextReceipt(receipt);
    return `<div class="dataset-stats"><span>${report.sources.length} sources</span><span>${report.contextBytes.toLocaleString()} context bytes</span><span>${report.sources.filter(source=>source.length>0).length} included</span></div>
      <ol class="context-source-details">${report.sources.map(source=>`<li><b>${esc(source.uri)}</b><div class="resource-toolbar"><span>Revision ${esc(source.revision)}</span><span>${esc(source.layer)} layer</span><span>${source.length.toLocaleString()} / ${source.original_length.toLocaleString()} bytes</span><span>${{included:'Included',truncated:'Prefix included',omitted:'Not included',empty:'Empty source'}[source.status]}</span></div>
        ${source.length?`<details><summary>Text used from this source</summary><pre>${esc([...source.content].slice(0,2000).join(''))}</pre>${[...source.content].length>2000?'<p class="muted">Showing the first 2,000 characters. The exact prompt below contains the full included text.</p>':''}</details>`:''}</li>`).join('')||'<li class="muted">This request used no resource sources.</li>'}</ol>`;
  } catch(error) { return `<p class="danger-action">${esc(error.message)}</p>`; }
}
