import {escapeHtml as esc} from './message.js';

export function writingTextStats(text) {
  const value = String(text || '');
  return {words: value.trim() ? value.trim().split(/\s+/u).length : 0, characters: [...value].length};
}

export function writingStatsLabel(text) {
  const {words, characters} = writingTextStats(text);
  return `${words.toLocaleString()} words · ${characters.toLocaleString()} characters`;
}

const guidanceFields = [['hook', 'Core hook'], ['style', 'Style guide'], ['rules', 'World rules'], ['characters', 'Characters and arcs']];
export const writingGuidanceKey = (namespace, uri) => `${namespace}\n${uri}`;

export function writingGuidancePanel(values = {}) {
  return `<details class="writing-guidance"><summary>Story guidance (this session)</summary>${guidanceFields.map(([key, label]) => `<label>${label}<textarea data-writing-guidance="${key}" rows="2" maxlength="512">${esc(values[key] || '')}</textarea></label>`).join('')}</details>`;
}

// Guidance is explicit user-authored prompt text, not retrieved resources or
// inherited metadata. Empty controls preserve the existing executed prompt.
export function writingPrompt(instructions, values = {}) {
  const notes = guidanceFields.filter(([key]) => values[key]?.trim()).map(([key, label]) => `${label}: ${values[key].trim()}`);
  return [instructions, ...notes].filter(Boolean).join('\n\n');
}

// Editor-first composition inspired by Snep's scene/library/creative layout.
// The resource controller still owns all listeners, CAS writes and receipts.
// This function moves existing nodes, preserving their identity and behavior.
export function arrangeWritingStudio(root, state) {
  const columns = root.querySelector('.resource-columns');
  const library = columns.firstElementChild;
  const editor = root.querySelector('.resource-editor');
  columns.classList.add('writing-studio');
  const main = document.createElement('div'); main.className = 'writing-main';
  const sidebar = document.createElement('aside'); sidebar.className = 'writing-sidebar'; sidebar.setAttribute('aria-label', 'Writing tools');
  const catalog = document.createElement('details'); catalog.className = 'panel writing-library'; catalog.open = true;
  catalog.innerHTML = '<summary>Library</summary>';
  const toolbar = root.querySelector('.resource-toolbar');
  const folderPicker = toolbar.querySelector('#writing-folder')?.closest('label');
  const actions = document.createElement('details'); actions.className = 'writing-library-actions';
  actions.innerHTML = '<summary>Create & import</summary><div class="resource-toolbar"></div>';
  const actionBar = actions.querySelector('.resource-toolbar');
  actionBar.insertAdjacentHTML('beforeend', '<label class="file-button">Import Markdown / text<input id="writing-text-import" type="file" accept=".md,.markdown,.txt,text/plain,text/markdown" hidden></label>');
  for (const control of toolbar.querySelectorAll('#resource-namespace, [data-resource="new-note"], [data-resource="new-character"], [data-resource="new-project"], .file-button'))
    actionBar.append(control.id === 'resource-namespace' ? control.closest('label') : control);
  catalog.append(toolbar);
  if (folderPicker) catalog.append(folderPicker);
  catalog.append(actions, library);
  sidebar.append(catalog);
  const creative = editor.querySelector('.writing-generate');
  if (creative) {
    creative.classList.add('panel'); creative.querySelector('h2').textContent = 'Creative controls';
    creative.querySelector('label').insertAdjacentHTML('afterend', writingGuidancePanel(state.guidance.get(writingGuidanceKey(state.namespace, state.draft.uri))));
    sidebar.append(creative);
  }
  const lore = editor.querySelector('.writing-lore');
  if (lore) { lore.classList.add('panel'); sidebar.append(lore); }
  // The remaining direct section is the dedicated folder move control.
  const folder = editor.querySelector(':scope > section');
  if (folder) {
    const details = document.createElement('details'); details.className = 'panel writing-placement';
    details.innerHTML = '<summary>Project / folder</summary>';
    folder.querySelector('h2')?.remove(); details.append(folder); sidebar.append(details);
  }
  if (state.draft) {
    const header = document.createElement('div'); header.className = 'writing-editor-heading';
    header.innerHTML = `<div><p class="eyebrow">TEXT EDITOR</p><span id="writing-count">${esc(writingStatsLabel(state.draft.content))}</span></div><button type="button" data-writing-focus aria-pressed="${state.focus}">${state.focus ? 'Show writing tools' : 'Focus on text'}</button>`;
    editor.prepend(header);
    const editorActions = editor.querySelector(':scope > .resource-toolbar');
    if (editorActions) {
      editorActions.insertAdjacentHTML('beforeend', '<details class="writing-export"><summary>Export draft</summary><div class="resource-toolbar"><button data-resource="export-markdown">Markdown (.md)</button><button data-resource="export-text">Plain text (.txt)</button></div></details>');
      if (state.revision !== '0') editorActions.insertAdjacentHTML('beforeend', '<button data-resource="reload">Reload saved version</button>');
      editor.querySelector(':scope > label').before(editorActions);
    }
    root.querySelector('#resource-content').placeholder = 'Begin your next scene…';
    root.querySelector('#resource-content').setAttribute('aria-label', 'Writing editor');
  }
  main.append(editor);
  const history = root.querySelector('.resource-history'); if (history) main.append(history);
  columns.replaceChildren(main, sidebar);
  root.querySelector('.resource-workspace').classList.add('writing-workspace');
  // An empty editor must always expose the library, including after discarding
  // a focused draft. Otherwise there is no remaining control to reopen tools.
  root.querySelector('.resource-workspace').classList.toggle('writing-focus', state.focus && !!state.draft);
}
