import {test} from 'node:test';
import assert from 'node:assert/strict';
import {writingTextStats, writingStatsLabel, writingPrompt, writingGuidancePanel, writingGuidanceKey} from '../src/writing-studio.js';

test('Writing statistics count prose and Unicode code points without interpreting markup', () => {
  assert.deepEqual(writingTextStats(' \n\t'), {words:0,characters:3});
  assert.deepEqual(writingTextStats('Hello\n한글 👋'), {words:3,characters:10});
  assert.deepEqual(writingTextStats('<script> inert'), {words:2,characters:14});
  assert.equal(writingStatsLabel('one two'), '2 words · 7 characters');
});

test('Session guidance is explicit escaped prose and empty guidance preserves existing instructions', () => {
  assert.equal(writingPrompt('Continue the scene.'), 'Continue the scene.');
  assert.equal(writingPrompt('', {style:'  Past tense.  ',rules:'No magic.'}), 'Style guide: Past tense.\n\nWorld rules: No magic.');
  assert.equal(writingPrompt('Continue.',{characters:'  '}),'Continue.');
  assert(!writingGuidancePanel({hook:'</textarea><script>alert(1)</script>'}).includes('<script>'));
  assert.notEqual(writingGuidanceKey('one','uri'), writingGuidanceKey('two','uri'));
});
