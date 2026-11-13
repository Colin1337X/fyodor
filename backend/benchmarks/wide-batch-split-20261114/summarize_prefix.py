"""Summarize diagnostic differences, without setting a new KV tolerance."""
import hashlib,json,math,sys
from pathlib import Path
out=Path(__file__).resolve().parent
if len(sys.argv) not in (2,3) or sys.argv[1] not in ('control','native-matmul','unsplit-attention','unfused-projections'):
 raise SystemExit('Choose control, native-matmul, unsplit-attention or unfused-projections [prefix >= 3]')
prefix=8192
if len(sys.argv)==3:
 if not sys.argv[2].isascii() or not sys.argv[2].isdecimal():raise SystemExit('Invalid prefix')
 prefix=int(sys.argv[2])
 if not 3<=prefix<=262142:raise SystemExit('Invalid prefix')
tag=sys.argv[1] if prefix==8192 else f'{sys.argv[1]}-ctx{prefix}'
path=out/f'prefix-kv-{tag}.json';meta=json.loads(path.read_text())
assert meta.get('prefix',8192)==prefix
assert meta['exit_code']==0
rows=[r for r in meta['rows'] if r['kind']=='kv'];logits=[r for r in meta['rows'] if r['kind']=='logits']
assert len(rows)==144 and len(logits)==1
assert {(r['layer'],r['position']) for r in rows}=={(l,p) for l in range(48) for p in (0,prefix//2,prefix-1)}
assert all(math.isfinite(r[k]) for r in rows for k in ('key_scaled','value_scaled','key_absolute','value_absolute'))
layers=[]
for layer in range(48):
 samples=[r for r in rows if r['layer']==layer]
 layers.append({'layer':layer,'key_max_scaled':max(r['key_scaled'] for r in samples),
                'value_max_scaled':max(r['value_scaled'] for r in samples),'all_samples_exact':all(r['exact'] for r in samples)})
summary={'accepted_benchmark':False,'prefix':prefix,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'logits':logits[0],'layers':layers,
         'largest_sample_differences':sorted(rows,key=lambda r:max(r['key_scaled'],r['value_scaled']),reverse=True)[:8],
         'note':'KV differences localize the numerical drift; no new KV acceptance tolerance is defined.'}
with (out/f'prefix-kv-{tag}-summary.json').open('x') as output:json.dump(summary,output,indent=2)
print(json.dumps({'logits':summary['logits'],'largest_sample_differences':summary['largest_sample_differences']},indent=2))
