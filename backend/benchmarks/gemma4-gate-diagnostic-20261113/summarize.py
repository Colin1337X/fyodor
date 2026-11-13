from pathlib import Path
import json,math
here=Path('backend/benchmarks/gemma4-gate-diagnostic-20261113');root=here/'raw'
meta=json.loads((root/'metadata.json').read_text());assert not meta['accepted'] and len(meta['runs'])==8 and all(r['exit_code']==0 for r in meta['runs'])
rows=[]
for context in (0,1024):
 row={'context':context,'engines':{}}
 for engine in ('baseline','fused'):
  docs=[json.loads((root/f'{context}-{engine}-{order}.json').read_text()) for order in ('a','b')]
  assert all(d['repetitions']==3 and d['kv_type']=='f32' and d['execution']=='resident' for d in docs)
  runs=[d['results'][0] for d in docs];mean=sum(r['tokens_per_second'] for r in runs)/2
  variance=sum(2*r['stddev']**2+3*(r['tokens_per_second']-mean)**2 for r in runs)/5
  row['engines'][engine]={'observations':6,'mean_tokens_per_second':mean,'pooled_sample_stddev':math.sqrt(variance),'weight_bytes':[d['device_weights_bytes'] for d in docs],'kernel_launches':[r['kernel_launches'] for r in runs]}
 b,f=(row['engines'][e] for e in ('baseline','fused'));assert b['weight_bytes']==f['weight_bytes']
 assert all(a-z==96*32*3 for a,z in zip(b['kernel_launches'],f['kernel_launches']))
 row['diagnostic_change_percent']=100*(f['mean_tokens_per_second']/b['mean_tokens_per_second']-1);rows.append(row)
summary={'accepted':False,'decision':'Do not promote fusion from these contaminated measurements. No accepted performance score changes.','results':rows}
(here/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
