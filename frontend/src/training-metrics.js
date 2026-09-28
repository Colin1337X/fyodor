// Older trainers report only loss. Missing or invalid telemetry stays absent.
export function currentTrainingLines(logs) {
  const lines=logs.filter(line=>line.includes('[trainer]'));
  const start=lines.findLastIndex(line=>line.includes('[trainer] starting '));
  return lines.slice(start<0?0:start).slice(-200);
}

export function parseTrainingMetric(line) {
  const match = /\bstep=(\d+) loss=([-+\d.eE]+)/.exec(line);
  if (!match) return null;
  const step = Number(match[1]), loss = Number(match[2]);
  if (!Number.isSafeInteger(step) || step < 0 || !Number.isFinite(loss)) return null;
  const optional = key => {
    const value = new RegExp(`\\b${key}=([-+\\d.eE]+)`).exec(line);
    if (!value) return null;
    const number = Number(value[1]);
    return Number.isFinite(number) && number >= 0 ? number : null;
  };
  const threads = optional('cpu_threads');
  return {step, loss, tokensPerSecond: optional('tokens_per_second'), graphBytes: optional('graph_bytes'),
    cpuThreads: Number.isInteger(threads) && threads >= 1 && threads <= 64 ? threads : null};
}

export function parseEvaluationMetric(line) {
  const match = /\beval_step=(\d+) validation_loss=([-+\d.eE]+)/.exec(line);
  if (!match) return null;
  const step=Number(match[1]), loss=Number(match[2]);
  if (!Number.isSafeInteger(step) || step<0 || !Number.isFinite(loss)) return null;
  const records=/\beval_records=(\d+)\b/.exec(line), count=records?Number(records[1]):null;
  if (count!==null && (!Number.isSafeInteger(count) || count<1)) return null;
  return {step,loss,records:count};
}
