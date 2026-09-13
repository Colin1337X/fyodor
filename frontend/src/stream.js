/* Read SSE records across arbitrary UTF-8/network boundaries. The adapter only
   accepts textual deltas: tool calls need the separate non-streaming agent path.
   Bound unfinished records and accumulated output to avoid an unbounded DOM or
   buffer when a remote endpoint misbehaves. */
export async function readCompletionStream(response, onUpdate) {
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let pending = '', text = '', usage = null, finished = false, ended = false;
  const record = raw => {
    const data = raw.split(/\r?\n/).filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n').trim();
    if (!data) return;
    if (data === '[DONE]') { ended = true; return; }
    const item = JSON.parse(data);
    if (item.error) throw new Error(item.error.message || 'Streaming endpoint error.');
    if (item.usage) usage = item.usage;
    const choice = item.choices?.[0];
    if (choice?.delta?.tool_calls) throw new Error('Streaming tool calls are unsupported; enable Agent mode for tools.');
    const delta = choice?.delta?.content ?? choice?.text;
    if (typeof delta === 'string') {
      if (text.length + delta.length > 2_000_000) throw new Error('Response exceeds the 2 million character limit.');
      text += delta;
      onUpdate(text);
    }
    if (choice?.finish_reason != null) finished = true;
  };
  try {
    while (!ended) {
      const {value, done} = await reader.read();
      pending += decoder.decode(value, {stream:!done});
      let match;
      while ((match = /\r?\n\r?\n/.exec(pending))) {
        const raw = pending.slice(0,match.index); pending = pending.slice(match.index+match[0].length);
        if (raw.length > 1_000_000) throw new Error('Streaming event exceeds the size limit.');
        record(raw);
        if (ended) break;
      }
      if (pending.length > 1_000_000) throw new Error('Streaming event exceeds the size limit.');
      if (done) { if (pending.trim() && !ended) record(pending); break; }
    }
    if (!ended && !finished) throw new Error('Stream ended before completion.');
    return {text, tokens:Number.isFinite(usage?.completion_tokens) ? usage.completion_tokens : 0};
  } finally {
    try { await reader.cancel(); } catch {}
    reader.releaseLock();
  }
}
