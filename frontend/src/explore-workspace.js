import {backend} from './api.js';
import {escapeHtml as esc} from './message.js';
import {exploreKind, explorePage, newExploreResource} from './explore-data.js';

// Drafts live in memory, separate from canonical resources. Updates send only
// title/text so native metadata, provenance and integer precision stay intact.
const state = {
  namespace: 'workspace', world: null, items: [], next: '', draft: null,
  revision: '0', dirty: false, busy: false, error: '', notice: '',
  history: [], historyNext: '0', preview: null, confirmDelete: false,
};
let mount = 0;
let repaint = () => {};
window.addEventListener('beforeunload', event => {
  if (state.dirty) { event.preventDefault(); event.returnValue = ''; }
});

export function exploreWorkspace(root) {
  const generation = ++mount;
  const current = () => generation === mount && document.body.dataset.view === 'explore';

  function paint() {
    if (!current()) return;
    const draft = state.draft;
    const kind = draft && exploreKind(draft.uri);
    root.innerHTML = `<section class="resource-workspace explore-workspace">
      <header class="view-heading"><div><p class="eyebrow">WORLD BUILDER</p><h1>Explore</h1><p class="muted">Create a world. Give its people, places and stories a home.</p></div></header>
      <div class="resource-toolbar"><label>Namespace<input id="explore-namespace" maxlength="64" value="${esc(state.namespace)}"></label>
      <button data-explore="refresh">Refresh</button><button data-explore="new-world">New world</button>
      ${state.world ? '<button data-explore="worlds">All worlds</button><button data-explore="new-lore">New lore</button>' : ''}</div>
      <p role="status" aria-live="polite" class="${state.error ? 'danger-action' : 'muted'}">${esc(state.error || state.notice || (state.busy ? 'Working…' : ''))}</p>
      ${state.world ? `<div class="explore-breadcrumb"><button data-explore="open-world">${esc(state.world.title || 'Untitled world')}</button><span aria-hidden="true">/</span><span>Lore library</span></div>` : ''}
      <div class="resource-columns"><section aria-label="${state.world ? 'Lore library' : 'World library'}"><h2>${state.world ? 'Lore' : 'Your worlds'}</h2>
        <ul class="resource-list">${state.items.map(item => `<li><button data-explore-uri="${esc(item.uri)}" aria-pressed="${draft?.uri === item.uri}"><b>${esc(item.title || 'Untitled')}</b><small>${state.world ? 'Lore entry' : 'World'} · Revision ${esc(item.revision)}</small></button></li>`).join('') || `<li class="explore-empty"><p>${state.next ? 'No matches on this page. Continue to the next page.' : state.world ? 'Add a person, place or piece of history.' : 'Your next story starts here.'}</p><button data-explore="${state.world ? 'new-lore' : 'new-world'}">${state.world ? 'Create lore' : 'Create a world'}</button></li>`}</ul>
        <button data-explore="next" ${state.next ? '' : 'disabled'}>Next page</button>
      </section><section class="panel resource-editor" aria-label="World editor">
        ${draft ? `<p class="eyebrow">${kind === 'world' ? 'WORLD' : 'LORE ENTRY'}</p><p class="fine-print">${state.revision === '0' ? 'New resource' : 'Revision ' + esc(state.revision)}${state.dirty ? ' · Unsaved changes' : ''}</p>
          <label>Title<input id="explore-title" maxlength="1024" value="${esc(draft.title)}"></label>
          <label>${kind === 'world' ? 'World description' : 'Lore text'}<textarea id="explore-content" rows="14" placeholder="${kind === 'world' ? 'Describe the setting, atmosphere and premise…' : 'Describe what makes this part of your world memorable…'}">${esc(draft.content)}</textarea></label>
          <div class="resource-toolbar"><button class="primary" data-explore="save">Save ${kind}</button><button data-explore="discard">Discard editor</button>
          ${state.revision !== '0' ? `<button data-explore="reload">Reload saved version</button><button data-explore="export">Export package</button>${kind === 'world' ? '<button data-explore="lore">Open lore library</button>' : ''}<button class="danger-action" data-explore="delete">Delete ${kind}</button>` : ''}</div>
          ${state.confirmDelete ? `<div class="explore-delete" role="group" aria-label="Confirm deletion"><p>Delete “${esc(draft.title || 'Untitled')}”?</p><button class="danger-action" data-explore="confirm-delete">Confirm delete</button><button data-explore="cancel-delete">Keep ${kind}</button></div>` : ''}
        ` : '<div class="explore-empty"><h2>Make room for a story</h2><p>Select a world or create one to begin.</p></div>'}
      </section></div>
      ${draft && state.revision !== '0' ? `<section class="panel resource-history"><h2>Revision history</h2><div class="resource-toolbar"><button data-explore="history">Latest revisions</button><button data-explore="history-next" ${state.historyNext !== '0' ? '' : 'disabled'}>Older revisions</button></div>
        <ul>${state.history.map(item => `<li><button data-explore-revision="${esc(item.revision)}">Revision ${esc(item.revision)}${item.deleted ? ' · Deleted' : ''}</button><small>${esc(new Date(Number(item.modified_ms)).toLocaleString())}</small></li>`).join('') || '<li>No revisions loaded.</li>'}</ul>
        ${state.preview ? `<h3>${esc(state.preview.resource.title)} · Revision ${esc(state.preview.revision)}</h3><pre id="explore-history-content">${esc(state.preview.resource.content)}</pre><button data-explore="restore-text">Copy title and text to editor</button>` : ''}</section>` : ''}
    </section>`;
    root.querySelectorAll('[data-explore]').forEach(button => button.addEventListener('click', () => void action(button.dataset.explore)));
    root.querySelectorAll('[data-explore-uri]').forEach(button => button.addEventListener('click', () => void run(async () => {
      requireClean(); await open(button.dataset.exploreUri);
    })));
    root.querySelectorAll('[data-explore-revision]').forEach(button => button.addEventListener('click', () => void run(async () => {
      state.preview = await backend.resource(state.namespace, state.draft.uri, button.dataset.exploreRevision);
    })));
    for (const key of ['title', 'content']) root.querySelector(`#explore-${key}`)?.addEventListener('input', event => {
      state.draft[key] = event.target.value; state.dirty = true; state.confirmDelete = false;
      root.querySelector('.resource-editor .fine-print').textContent = `${state.revision === '0' ? 'New resource' : 'Revision ' + state.revision} · Unsaved changes`;
      root.querySelector('.explore-delete')?.remove();
    });
    root.querySelector('#explore-namespace').addEventListener('change', event => {
      const namespace = event.target.value;
      void run(async () => {
        requireClean();
        if (!/^[A-Za-z0-9_-]{1,64}$/.test(namespace)) throw new Error('Use 1–64 letters, digits, underscores or hyphens.');
        state.namespace = namespace; state.world = null; clearDraft(); await load();
      });
    });
    if (state.busy) root.querySelectorAll('button,input,textarea').forEach(control => control.disabled = true);
  }

  function requireClean() {
    if (state.dirty) throw new Error('Save or discard the current editor first.');
  }

  function clearDraft() {
    Object.assign(state, {draft: null, revision: '0', dirty: false, history: [], historyNext: '0', preview: null, confirmDelete: false});
  }

  async function load(after = '') {
    const page = await explorePage(backend, state.namespace, state.world?.uri, after);
    state.items = page.resources; state.next = page.next;
  }

  async function history(before = '0') {
    const page = await backend.resourceHistory(state.namespace, state.draft.uri, before);
    state.history = page.revisions; state.historyNext = page.next;
  }

  async function open(uri) {
    const result = await backend.resource(state.namespace, uri);
    clearDraft(); state.draft = result.resource; state.revision = result.revision;
    await history();
  }

  async function run(work) {
    if (state.busy) return;
    state.busy = true; state.error = ''; state.notice = ''; paint();
    try { await work(); } catch (error) { state.error = error.message; }
    finally { state.busy = false; repaint(); }
  }

  async function action(name) {
    await run(async () => {
      if (name.startsWith('new-')) {
        requireClean();
        if (name === 'new-world' && state.world) { state.world = null; await load(); }
        clearDraft();
        state.draft = newExploreResource(name === 'new-world' ? 'world' : 'lore', state.world?.uri);
        state.dirty = true;
      } else if (name === 'save') {
        const draft = state.draft;
        const saved = state.revision === '0'
          ? await backend.saveResource(state.namespace, draft, '0')
          : await backend.updateResource(state.namespace, draft, state.revision);
        state.revision = saved.revision;
        // A successful write must clear dirty state even if the subsequent read
        // fails. Refresh/reload is then recoverable without duplicating creation.
        state.dirty = false;
        await open(draft.uri);
        if (state.world?.uri === draft.uri) state.world.title = draft.title;
        await load(); state.notice = 'Saved to your world library.';
      } else if (name === 'discard') clearDraft();
      else if (name === 'reload') { await open(state.draft.uri); state.notice = 'Saved version loaded.'; }
      else if (name === 'refresh') { requireClean(); await load(); }
      else if (name === 'next') { requireClean(); if (state.next) await load(state.next); }
      else if (name === 'worlds') { requireClean(); state.world = null; clearDraft(); await load(); }
      else if (name === 'lore') {
        requireClean(); state.world = {uri: state.draft.uri, title: state.draft.title}; clearDraft(); await load();
      } else if (name === 'open-world') { requireClean(); await open(state.world.uri); }
      else if (name === 'history' || name === 'history-next') await history(name === 'history' ? '0' : state.historyNext);
      else if (name === 'restore-text') {
        state.draft.title = state.preview.resource.title; state.draft.content = state.preview.resource.content;
        state.dirty = true; state.notice = 'Revision text copied to the editor. Save to keep it.';
      } else if (name === 'export') {
        requireClean();
        const text = await backend.exportResource(state.namespace, state.draft.uri);
        const url = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
        const link = document.createElement('a'); link.href = url; link.download = `${exploreKind(state.draft.uri)}-${state.draft.uri.split('/').at(-1)}.json`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } else if (name === 'delete') { requireClean(); state.confirmDelete = true; }
      else if (name === 'cancel-delete') state.confirmDelete = false;
      else if (name === 'confirm-delete' && state.confirmDelete) {
        requireClean(); await backend.deleteResource(state.namespace, state.draft.uri, state.revision);
        if (state.world?.uri === state.draft.uri) state.world = null;
        clearDraft(); await load(); state.notice = 'Resource deleted.';
      }
    });
  }

  repaint = paint;
  paint();
  if (!state.busy) void run(() => load());
}
