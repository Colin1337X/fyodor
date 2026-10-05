/* Blocking, local script in <head>: resolve the palette before CSS paints.
   Preferences are deliberately separate from backend/session credentials. */
(() => {
  const presets = window.fyodorThemes;
  const themes = ['system', ...presets.map(t => t.id), 'graphite', 'forest', 'violet'];
  const aliases = {'blue-black':'fyodor-dark', 'blue-white':'fyodor', 'orange-white':'fyodor-mocha'};
  const media = matchMedia('(prefers-color-scheme: dark)');
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('fyodor-appearance')) || {}; } catch {}
  saved = typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  saved.theme = aliases[saved.theme] || saved.theme;
  const fonts={system:"system-ui,-apple-system,'Segoe UI',sans-serif",serif:"Georgia,'Times New Roman',serif",sans:"Arial,Helvetica,sans-serif"};
  const monos={system:"ui-monospace,'Cascadia Code',Consolas,monospace",consolas:"Consolas,'Liberation Mono',monospace",courier:"'Courier New',monospace"};
  const defaults={accent:'',codeColor:'',syntaxKeyword:'',syntaxString:'',syntaxComment:'',syntaxNumber:'',radius:18,fontSize:14,font:'system',mono:'system',sidebarWidth:190,spacing:24,chatLayout:'centered',borders:true,shadows:true,translucency:false,motion:'system'};
  function normalize(input,base=defaults){
    const next={...base};
    for(const key of ['accent','codeColor','syntaxKeyword','syntaxString','syntaxComment','syntaxNumber'])if(typeof input[key]==='string' && /^(|#[0-9a-f]{6})$/i.test(input[key]))next[key]=input[key];
    for(const [key,min,max] of [['radius',0,24],['fontSize',12,20],['sidebarWidth',160,320],['spacing',12,36]])
      if(Number.isInteger(input[key]) && input[key]>=min && input[key]<=max)next[key]=input[key];
    for(const key of ['borders','shadows','translucency'])if(typeof input[key]==='boolean')next[key]=input[key];
    for(const [key,values] of [['font',Object.keys(fonts)],['mono',Object.keys(monos)],['chatLayout',['centered','wide']],['motion',['system','reduced','full']]])
      if(values.includes(input[key]))next[key]=input[key];
    return next;
  }
  let state = {
    theme: themes.includes(saved.theme) ? saved.theme : 'fyodor',
    compact: saved.compact === true,
    collapsed: saved.collapsed === true,
    ...normalize(saved),
  };
  function apply() {
    document.documentElement.dataset.theme = state.theme === 'system' ? (media.matches ? 'fyodor-dark' : 'fyodor') : state.theme;
    document.documentElement.dataset.density = state.compact ? 'compact' : 'comfortable';
    document.documentElement.dataset.sidebar = state.collapsed ? 'collapsed' : 'expanded';
    const root=document.documentElement;
    for(const key of ['borders','shadows','translucency','motion','chatLayout'])root.dataset[key]=String(state[key]);
    for(const [key,value] of Object.entries({'--radius':state.radius+'px','--ui-size':state.fontSize+'px','--ui-font':fonts[state.font],'--mono':monos[state.mono],'--rail':state.sidebarWidth+'px','--space':(state.compact?Math.max(12,state.spacing-8):state.spacing)+'px'}))root.style.setProperty(key,value);
    for(const key of ['--accent','--accent-hover','--on-accent','--accent-text'])root.style.removeProperty(key);
    if(state.accent){
      const rgb=state.accent.slice(1).match(/../g).map(c=>parseInt(c,16)/255).map(c=>c<=.04045?c/12.92:((c+.055)/1.055)**2.4);
      const luminance=rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722;
      const ink=luminance>.179?'#000000':'#ffffff';
      root.style.setProperty('--accent',state.accent);root.style.setProperty('--accent-hover',state.accent);
      root.style.setProperty('--on-accent',ink);root.style.setProperty('--accent-text',ink);
    }
    if(state.codeColor)root.style.setProperty('--code-text',state.codeColor);else root.style.removeProperty('--code-text');
    for(const [key,role] of [['syntaxKeyword','keyword'],['syntaxString','string'],['syntaxComment','comment'],['syntaxNumber','number']]){
      if(state[key])root.style.setProperty('--syntax-'+role,state[key]);else root.style.removeProperty('--syntax-'+role);
    }
  }
  window.fyodorAppearance = Object.freeze({
    themes: Object.freeze(themes),
    label: id => presets.find(t => t.id === id)?.name || ({system:'System', graphite:'Graphite (legacy)',forest:'Forest (legacy)',violet:'Violet (legacy)'})[id] || id,
    get: () => ({ ...state }),
    reset: () => { state={theme:'fyodor',compact:false,collapsed:state.collapsed,...defaults};apply();try{localStorage.setItem('fyodor-appearance',JSON.stringify(state));}catch{} },
    set: patch => {
      if(!patch || typeof patch!=='object')return;
      state = { ...normalize(patch,state),theme: themes.includes(patch.theme) ? patch.theme : state.theme,
        compact: typeof patch.compact === 'boolean' ? patch.compact : state.compact,
        collapsed: typeof patch.collapsed === 'boolean' ? patch.collapsed : state.collapsed };
      apply();
      try { localStorage.setItem('fyodor-appearance', JSON.stringify(state)); } catch {}
    },
  });
  media.addEventListener('change', apply);
  apply();
})();
