export const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// Deliberately small document renderer: text and fenced code only. Model HTML,
// URLs, SVG, and event handlers stay inert. A code fence cannot escape its node.
export function messageBody(text) {
  const parts = String(text).split(/```([^\n`]*)\n([\s\S]*?)(?:```|$)/g);
  let html = '';
  for (let i = 0; i < parts.length; i += 3) {
    if (parts[i]) html += `<p>${escapeHtml(parts[i])}</p>`;
    if (parts[i+2] !== undefined) html += `<div class="code-block"><header><span>${escapeHtml(parts[i+1].trim() || 'Code')}</span><button type="button" data-copy-code>Copy code</button></header><pre><code>${escapeHtml(parts[i+2])}</code></pre></div>`;
  }
  return html;
}
