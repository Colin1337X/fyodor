import {test} from 'node:test';
import assert from 'node:assert/strict';
import {decodeWritingFile, writingDownloadName, writingFileLimit} from '../src/writing-files.js';
const bytes = text => new TextEncoder().encode(text);

test('Writing imports preserve Markdown, Unicode and line endings, stripping only the UTF-8 BOM', () => {
  const text = '# 한글 👋\r\n\r\n<script>inert</script>\n';
  assert.deepEqual(decodeWritingFile(bytes('\ufeff'+text), 'story.MD'), {title:'story',content:text});
  assert.deepEqual(decodeWritingFile(bytes(''), 'empty.txt'), {title:'empty',content:''});
  assert.equal(decodeWritingFile(bytes('a'), 'C:\\notes\\chapter.markdown').title,'chapter');
});

test('Writing imports reject binary/invalid UTF-8, wrong formats and oversized content without replacement decoding', () => {
  assert.throws(()=>decodeWritingFile(new Uint8Array([0xc3,0x28]),'broken.txt'),/UTF-8/);
  assert.throws(()=>decodeWritingFile(bytes('a\0b'),'binary.txt'),/null characters/);
  assert.throws(()=>decodeWritingFile(bytes('{}'),'package.json'),/Markdown/);
  assert.throws(()=>decodeWritingFile(new Uint8Array(writingFileLimit+1),'large.txt'),/1 MiB/);
  const imported = decodeWritingFile(bytes('x'), '한'.repeat(500)+'.txt');
  assert.ok(bytes(imported.title).byteLength<=1024);
  assert.equal(imported.title.length,341);
});

test('Draft download names preserve Unicode and avoid filesystem separators and Windows device names', () => {
  assert.equal(writingDownloadName('한글 / draft: one','md'), '한글 _ draft_ one.md');
  assert.equal(writingDownloadName('CON','txt'),'Writing-CON.txt');
  assert.equal(writingDownloadName('LPT1.notes','md'),'Writing-LPT1.notes.md');
  assert.equal(writingDownloadName('...','txt'),'Untitled.txt');
  assert.throws(()=>writingDownloadName('story','exe'),/Unsupported/);
});
