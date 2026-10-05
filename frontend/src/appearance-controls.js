// Backdrop blur is progressive enhancement. OS reduced motion always wins;
// syntax colors cover C/C++, JS/TS, Python and JSON code fences.
import {escapeHtml} from './message.js';
export function syncAppearanceColors(a){
  const styles=getComputedStyle(document.documentElement);
  for(const [key,role] of Object.entries({accent:'accent',codeColor:'text',syntaxKeyword:'info',syntaxString:'success',syntaxComment:'muted',syntaxNumber:'warning'})){
    const input=document.querySelector(`[data-appearance="${key}"]`),button=document.querySelector(`[data-appearance-clear="${key}"]`);
    if(input){const value=a[key]||styles.getPropertyValue('--'+role).trim();if(/^#[0-9a-f]{6}$/i.test(value))input.value=value;}
    if(button){button.disabled=!a[key];button.textContent=a[key]?'Use theme':'From theme';}
  }
}
export function appearanceControls(a){
  const select=(key,label,choices)=>`<label>${label}<select data-appearance="${key}">${choices.map(([value,name])=>`<option value="${value}" ${a[key]===value?'selected':''}>${name}</option>`).join('')}</select></label>`;
  const number=(key,label,min,max)=>`<label>${label}<input data-appearance="${key}" type="number" min="${min}" max="${max}" value="${a[key]}"></label>`;
  const check=(key,label)=>`<label class="check"><input data-appearance="${key}" type="checkbox" ${a[key]?'checked':''}>${label}</label>`;
  const color=(key,label)=>`<label>${label}<span class="appearance-color"><input data-appearance="${key}" type="color" value="${escapeHtml(a[key]||'#215ea8')}" aria-label="${label}"><button type="button" data-appearance-clear="${key}">Use theme</button></span></label>`;
  return `<details class="appearance-custom"><summary>Customize appearance</summary><div class="appearance-controls">
    ${color('accent','Accent override')}${color('codeColor','Code text color')}
    ${color('syntaxKeyword','Syntax keywords')}${color('syntaxString','Syntax strings')}${color('syntaxComment','Syntax comments')}${color('syntaxNumber','Syntax numbers')}
    ${number('radius','Corner radius (px)',0,24)}${number('fontSize','Font size (px)',12,20)}
    ${select('font','Text font',[['system','System'],['sans','Sans serif'],['serif','Serif']])}
    ${select('mono','Code font',[['system','System monospace'],['consolas','Consolas'],['courier','Courier New']])}
    ${number('sidebarWidth','Sidebar width (px)',160,320)}${number('spacing','Workspace spacing (px)',12,36)}
    ${select('chatLayout','Chat layout',[['centered','Centered'],['wide','Wide']])}
    ${select('motion','Animation',[['system','Follow system'],['reduced','Reduced'],['full','Standard']])}
    ${check('borders','Panel borders')}${check('shadows','Shadows')}${check('translucency','Translucent panels')}
    </div><button type="button" id="appearance-reset">Reset appearance</button></details>`;
}
