from pathlib import Path
import json,math
here=Path('backend/benchmarks/gemma4-portable-decode-20261029');root=here/'reference'
meta=json.loads((root/'metadata.json').read_text());gates=json.loads((root/'environment.json').read_text())
assert len(meta['runs'])==len(gates['activities'])==12
assert all(r['exit_code']==0 for r in meta['runs']) and all(g['passed'] for g in gates['activities'])
rows=[]
for context in (0,1024):
 row={'context':context,'engines':{}}
 for engine in ('baseline','fixed','llama'):
  docs=[json.loads((root/f'{context}-{engine}-{order}.json').read_text()) for order in ('a','b')]
  assert all(d['repetitions']==3 and d['kv_type']=='f32' for d in docs)
  runs=[d['results'][0] for d in docs];n=6;mean=sum(r['tokens_per_second']*3 for r in runs)/n
  variance=sum(2*r['stddev']**2+3*(r['tokens_per_second']-mean)**2 for r in runs)/(n-1)
  row['engines'][engine]={'observations':n,'mean_tokens_per_second':mean,'pooled_sample_stddev':math.sqrt(variance)}
 b,f,l=(row['engines'][e]['mean_tokens_per_second'] for e in ('baseline','fixed','llama'))
 row.update(portable_percent_of_llama=100*f/l,native_percent_of_llama=100*b/l,portable_change_percent=100*(f/b-1))
 rows.append(row)
summary={'decision':'Do not promote unpartitioned portable attention for Gemma decode','environment_gates_passed':12,'contract':meta['contract'],'results':rows}
(here/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
