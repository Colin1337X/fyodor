// Small lexical highlighter, never an HTML parser or language evaluator.
// Unsupported languages and large blocks remain inert plain text.
const escape=value=>value.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const supported=new Set(['c','h','cpp','c++','js','javascript','ts','typescript','json','py','python']);
const keywords=new Set('auto break case char const continue default do double else enum extern float for goto if int long register return short signed sizeof static struct switch typedef union unsigned void volatile while class function let var new null true false undefined async await import export from def elif except finally in is lambda nonlocal not or and pass raise try with yield None True False as assert del global'.split(' '));
export function highlightCode(text,language){
  const lang=language.trim().toLowerCase();
  if(!supported.has(lang)||text.length>65536)return escape(text);
  const python=lang==='py'||lang==='python';
  const comment=python?'#[^\\n]*':'//[^\\n]*|/\\*[\\s\\S]*?(?:\\*/|$)';
  const tokens=new RegExp(`${comment}|"(?:\\\\[\\s\\S]|[^"\\\\])*"|'(?:\\\\[\\s\\S]|[^'\\\\])*'|\\b(?:0[xX][0-9a-fA-F]+|\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b|\\b[A-Za-z_][A-Za-z_0-9]*\\b`,'g');
  let result='',position=0;
  for(const match of text.matchAll(tokens)){
    result+=escape(text.slice(position,match.index));const value=match[0];
    const type=(python?value.startsWith('#'):value.startsWith('//')||value.startsWith('/*'))?'comment':
      value[0]==='"'||value[0]==="'"?'string':/^\d/.test(value)?'number':keywords.has(value)?'keyword':null;
    result+=type?`<span class="syntax-${type}">${escape(value)}</span>`:escape(value);position=match.index+value.length;
  }
  return result+escape(text.slice(position));
}
