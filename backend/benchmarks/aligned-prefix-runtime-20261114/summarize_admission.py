"""Check observed scratch deltas and memory-driven batch reduction."""
import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parent
destination = out / 'admission-summary.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite evidence')
names = [f'admission-cap260-mib0-{mode}' for mode in ('on', 'off', 'noReuse', 'noSplit', 'portable')]
names.append('admission-cap8195-mib8500-on')
cases = {}
for name in names:
    data = json.loads((out / f'{name}.json').read_text())
    assert data['exit_code'] == 0 and data['parse_error'] is None and len(data['rows']) == 1
    assert data['rows'][0]['resident']
    cases[name] = data
on, off = (cases[f'admission-cap260-mib0-{mode}'] for mode in ('on', 'off'))
baseline, candidate = off['rows'][0], on['rows'][0]
assert candidate['batch'] == baseline['batch'] == 260
assert candidate['kv'] == baseline['kv'] and candidate['weights'] == baseline['weights']
assert on['partial_queries'] == [260]
# Gemma full heads: 16 heads * (512 value columns + 2 statistics) * 4 parts.
# Ordinary selective batch splitting reserves four queries, aligned uses 260.
expected_extra = (260 - 4) * 16 * (512 + 2) * 4 * 4
assert candidate['scratch'] - baseline['scratch'] == expected_extra
for mode in ('off', 'noReuse', 'noSplit', 'portable'):
    assert cases[f'admission-cap260-mib0-{mode}']['partial_queries'] == []
for mode in ('noReuse', 'portable'):
    assert cases[f'admission-cap260-mib0-{mode}']['rows'][0]['scratch'] == baseline['scratch']
limited = cases['admission-cap8195-mib8500-on']
row = limited['rows'][0]
assert 1 < row['batch'] < 512 and limited['partial_queries'] == [row['batch']]
assert row['weights'] + row['kv'] + row['scratch'] <= 8500 * 1024 * 1024
summary = dict(accepted_benchmark=False, extra_partial_bytes=expected_extra,
               constrained_batch=row['batch'], constrained_partial_queries=limited['partial_queries'],
               input_sha256={name: hashlib.sha256((out / f'{name}.json').read_bytes()).hexdigest() for name in names},
               limitations=['Plan allocation evidence only; no throughput measurement.',
                            'The configured cache budget is not a hard process VRAM limit; external libraries have separate storage.'])
with destination.open('x') as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
