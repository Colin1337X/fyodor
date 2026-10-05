import {test} from 'node:test';
import assert from 'node:assert/strict';
import {highlightCode} from '../src/syntax.js';
test('Syntax rendering keeps executable markup inert and token boundaries intact',()=>{
  const text='const n = 42; // <script>alert(1)</script>\nreturn "<img onerror=x>";';
  const html=highlightCode(text,'js');
  assert.match(html,/<span class="syntax-keyword">const<\/span>/);
  assert.match(html,/<span class="syntax-number">42<\/span>/);
  assert.match(html,/syntax-comment/);assert.match(html,/syntax-string/);
  assert.ok(!html.includes('<script>')&&!html.includes('<img'));
  assert.equal(html.replace(/<\/?span[^>]*>/g,'').replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&quot;','"'),text);
  assert.match(highlightCode('# comment\nreturn None','python'),/syntax-comment/);
  assert.equal(highlightCode('<svg>','html'),'&lt;svg&gt;');
  assert.equal(highlightCode('x'.repeat(65537),'c'),'x'.repeat(65537));
  assert.equal(highlightCode('unterminated "<img>','json'),'unterminated &quot;&lt;img&gt;');
});
