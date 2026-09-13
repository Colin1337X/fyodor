/* Blocking, local script in <head>: resolve the palette before CSS paints.
   Preferences are deliberately separate from backend/session credentials. */
(() => {
  const themes = ['system', 'blue-black', 'blue-white', 'orange-white', 'graphite', 'forest', 'violet'];
  const media = matchMedia('(prefers-color-scheme: dark)');
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('fyodor-appearance')) || {}; } catch {}
  let state = {
    theme: themes.includes(saved.theme) ? saved.theme : 'blue-black',
    compact: saved.compact === true,
    collapsed: saved.collapsed === true,
  };
  function apply() {
    document.documentElement.dataset.theme = state.theme === 'system' ? (media.matches ? 'blue-black' : 'blue-white') : state.theme;
    document.documentElement.dataset.density = state.compact ? 'compact' : 'comfortable';
    document.documentElement.dataset.sidebar = state.collapsed ? 'collapsed' : 'expanded';
  }
  window.fyodorAppearance = Object.freeze({
    themes: Object.freeze(themes),
    get: () => ({ ...state }),
    set: patch => {
      state = { theme: themes.includes(patch.theme) ? patch.theme : state.theme,
        compact: typeof patch.compact === 'boolean' ? patch.compact : state.compact,
        collapsed: typeof patch.collapsed === 'boolean' ? patch.collapsed : state.collapsed };
      apply();
      try { localStorage.setItem('fyodor-appearance', JSON.stringify(state)); } catch {}
    },
  });
  media.addEventListener('change', apply);
  apply();
})();
