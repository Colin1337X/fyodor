from pathlib import Path
import json,math
here=Path(__file__).resolve().parent;raw=here/'raw'
meta=json.loads((raw/'metadata.json').read_text())
assert not meta['accepted'] and len(meta['runs'])==12 and all(r['exit_code']==0 for r in meta['runs'])
rows=[]
for context in (0,8192):
 row={'context':context,'engines':{}}
 for engine in ('ring','wide','llama'):
  docs=[json.loads((raw/f'{context}-{engine}-{phase}.json').read_text()) for phase in ('a','b')]
  assert all(d['repetitions']==3 and d['warmup']==1 and d['kv_type']=='f32' for d in docs)
  if engine!='llama':assert all(d['execution']=='resident' for d in docs)
  else:assert all(d['flash_attention']==False for d in docs)
  runs=[d['results'][0] for d in docs]
  assert all(r['tokens']==32 and r['prefix_tokens']==context for r in runs)
  mean=sum(r['tokens_per_second'] for r in runs)/2
  var=sum(2*r['stddev']**2+3*(r['tokens_per_second']-mean)**2 for r in runs)/5
  row['engines'][engine]={'observations':6,'mean_tokens_per_second':mean,'pooled_sample_stddev':math.sqrt(var),'order_means':[r['tokens_per_second'] for r in runs]}
  if engine!='llama':
   for field in ('device_weights_bytes','device_kv_bytes','device_scratch_bytes'):
    row['engines'][engine][field]=[d[field] for d in docs]
   row['engines'][engine]['kernel_launches']=[r['kernel_launches'] for r in runs]
  monitors=[json.loads((raw/f'{context}-{engine}-{phase}-environment.json').read_text()) for phase in ('a','b')]
  assert all(m['samples'] and not m['accepted'] and all('error' not in sample for sample in m['samples']) for m in monitors)
 for field in ('device_weights_bytes','device_kv_bytes','device_scratch_bytes','kernel_launches'):
  assert row['engines']['ring'][field]==row['engines']['wide'][field],(context,field)
 b,r,l=(row['engines'][name]['mean_tokens_per_second'] for name in ('ring','wide','llama'))
 row.update(diagnostic_wide_percent_of_llama=100*r/l,diagnostic_ring_percent_of_llama=100*b/l,diagnostic_wide_change_percent=100*(r/b-1))
 rows.append(row)
summary={'accepted':False,'decision':'Diagnostic only; keep wide value reduction opt-in. Do not replace accepted llama.cpp scores or claim the 125% target.','contract':meta['contract'],'results':rows}
(here/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
