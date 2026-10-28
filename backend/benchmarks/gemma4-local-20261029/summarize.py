from pathlib import Path
import json,math
p=Path(__file__).resolve().parent;d=p/'reference-clean'
m=json.loads((d/'metadata.json').read_text());e=json.loads((d/'environment.json').read_text());assert len(m['runs'])==len(e['activities'])==16;assert all(a['passed'] for a in e['activities'])
rows=[]
for prefix in (0,1024,4096,8192):
 row={'prefix':prefix}
 for engine in ('fixed','llama'):
  ds=[json.loads((d/f'{prefix}-{engine}-{phase}.json').read_text()) for phase in ('a','b')];n=sum(x['repetitions'] for x in ds);mean=sum(x['repetitions']*x['results'][0]['tokens_per_second'] for x in ds)/n
  sd=math.sqrt(sum((x['repetitions']-1)*x['results'][0]['stddev']**2+x['repetitions']*(x['results'][0]['tokens_per_second']-mean)**2 for x in ds)/(n-1))
  row[engine]={'samples':n,'tokens_per_second':mean,'sample_stdev':sd}
 row['fyodor_llama_percent']=100*row['fixed']['tokens_per_second']/row['llama']['tokens_per_second'];rows.append(row)
(p/'reference-summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
