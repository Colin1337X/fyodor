import {escapeHtml as esc} from './message.js';

// Render-only component. The parent workspace owns requests and revisions;
// the native store validates identity, namespace, duplicates and permissions.
export function writingLoreUris(resource) {
  return Array.isArray(resource?.metadata?.writing_lore)
    ? resource.metadata.writing_lore.filter(uri => typeof uri === 'string') : [];
}

export function writingLorePanel(resource, entries, next, titles = new Map()) {
  const links = writingLoreUris(resource);
  const names = new Map([...titles, ...entries.map(entry => [entry.uri, entry.title])]);
  return `<details class="writing-lore">
    <summary>Lore references (${links.length})</summary>
    <ul class="writing-lore-links">
      ${links.map((uri, index) => `<li>
        <span title="${esc(uri)}">${esc(names.get(uri) || `Lore entry ${index + 1}`)}</span>
        <button data-lore-action="remove" data-lore-uri="${esc(uri)}">Unlink</button>
      </li>`).join('') || '<li class="muted">No lore linked.</li>'}
    </ul>
    <div class="resource-toolbar">
      <button data-lore-action="browse">Browse lore</button>
      <button data-lore-action="next" ${next ? '' : 'disabled'}>Next page</button>
      <button data-lore-action="grant" ${links.length ? '' : 'disabled'}>Allow model access</button>
    </div>
    <ul class="writing-lore-links">
      ${entries.map(entry => `<li>
        <span>${esc(entry.title || 'Untitled lore')}</span>
        <button data-lore-action="add" data-lore-uri="${esc(entry.uri)}"
          ${links.includes(entry.uri) || links.length >= 32 ? 'disabled' : ''}>
          ${links.includes(entry.uri) ? 'Linked' : 'Link'}
        </button>
      </li>`).join('')}
    </ul>
  </details>`;
}
