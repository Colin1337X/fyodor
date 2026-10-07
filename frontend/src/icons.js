// A single stroke system keeps controls legible at small sizes. All paths are
// static application data; no model output is ever treated as SVG markup.
const paths = {
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  evaluations: '<path d="M9 3h6v4H9zM9 5H5v16h14V5h-4M8 12l2 2 5-5M8 18h8"/>',
  datasets: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  explore: '<circle cx="12" cy="12" r="9"/><path d="m16 8-3 5-5 3 3-5 5-3Z"/>',
  writing: '<path d="M4 20h4L20 8l-4-4L4 16v4Zm9-13 4 4M4 20h16"/>',
  workspace: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  chat: '<path d="M21 11a8 8 0 0 1-8 8H6l-4 3V11a9 9 0 0 1 19 0Z"/><path d="M7 9h10M7 13h6"/>',
  playground: '<path d="m8 4-6 8 6 8m8-16 6 8-6 8M14 3l-4 18"/>',
  models: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>',
  training: '<path d="M4 20V10m8 10V4m8 16v-7M2 20h20"/>',
  api: '<path d="m8 5-6 7 6 7m8-14 6 7-6 7m-2-16-4 20"/>',
  logs: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m6 8 3 3-3 3m6 1h5"/>',
  settings: '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="16" cy="17" r="3"/>',
  sidebar: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  benchmark: '<path d="M4 19a10 10 0 1 1 16 0M12 13l5-6M8 19h8"/><circle cx="12" cy="13" r="2"/>',
};
export const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.workspace}</svg>`;
export const navigationGroups = [['Use', [['workspace','Overview'],['chat','Chat'],['models','Models']]],
    ['Create', [['writing','Writing'],['explore','Explore'],['datasets','Datasets'],['resources','Resources'],['context','Context'],['playground','Playground'],['training','Training']]],
    ['Run', [['evaluations','Evaluations'],['benchmark','Benchmarks'],['logs','Logs']]], ['Serve', [['api','API access']]]];
export function navigationMarkup() {
  return navigationGroups.map(([title, entries]) => `<div class="nav-group"><label>${title}</label>${entries.map(([key,label]) =>
    `<button data-view="${key}" title="${label}" aria-label="${label}">${icon(key)}<span>${label}</span></button>`).join('')}</div>`).join('');
}
