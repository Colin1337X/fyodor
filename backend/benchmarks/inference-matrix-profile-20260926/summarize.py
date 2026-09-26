"""Validate diagnostic matrix dispatch counts and summarize measured shapes."""
import hashlib
import json
from pathlib import Path
import re

out = Path(__file__).resolve().parent
root = out.parents[2]
pattern = re.compile(r'CUDA matrix profile: path=(\S+) type=(\d+) rows=(\d+) columns=(\d+) batch=(\d+) calls=(\d+) milliseconds=([\d.]+)')
summary = {}
for name in ('prefill', 'short', 'long'):
    log = (out / 'baseline' / f'profile-{name}.log').read_text()
    rows = [dict(zip(('path','type','rows','columns','batch','calls','milliseconds'),
                    (m[0], *map(int,m[1:6]), float(m[6])))) for m in pattern.findall(log)]
    assert rows and not any(r['path'] == 'overflow' for r in rows)
    vendor_calls = sum(r['calls'] for r in rows if r['path'] == 'vendor-gemm')
    dot_calls = sum(r['calls'] for r in rows if r['path'] == 'dot')
    # TinyLlama: 22 layers * 7 projections, one final output projection.
    # Three full sequences, including one warmup. Prefill only returns final logits.
    assert vendor_calls == {'prefill': 3*154, 'short': 0, 'long': 3*2*154}[name]
    assert dot_calls == {'prefill': 3, 'short': 3*128*155, 'long': 3*128*155+3}[name]
    total = sum(r['milliseconds'] for r in rows)
    aggregate = float(re.search(r'matmul=([\d.]+)', log)[1])
    assert total <= aggregate + 0.001  # Aggregate also includes request metadata kernels.
    summary[name] = {'vendor_calls':vendor_calls, 'dot_calls':dot_calls,
                     'matrix_milliseconds':total, 'shapes':sorted(rows,key=lambda r:-r['milliseconds'])}
sources = ('CUDA/resident.inc','CUDA/resident.cu','CUDA/matvec.cu','CUDA/gemm.cu')
summary['source_sha256'] = {s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ('prefill','short','long'):
    print(name, 'vendor',summary[name]['vendor_calls'],'dot',summary[name]['dot_calls'])
    for row in summary[name]['shapes'][:3]:
        print(row, 'share', round(100*row['milliseconds']/summary[name]['matrix_milliseconds'],2))
