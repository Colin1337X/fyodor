"""Validate all recorded activities and pool every timed sample."""
from pathlib import Path
import json,math,statistics
root=Path(__file__).resolve().parent

def read(path):
 data=path.read_bytes()
 return json.loads(data.decode('utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'))
def pool(parts):
 n=sum(p[0] for p in parts);mean=sum(p[0]*p[1] for p in parts)/n
 var=sum((p[0]-1)*p[2]**2+p[0]*(p[1]-mean)**2 for p in parts)/(n-1)
 return {'samples':n,'tokens_per_second':mean,'sample_stdev':math.sqrt(var)}
clean=root/'confirmation';meta=read(clean/'metadata.json');environment=read(clean/'environment.json')
assert len(meta['runs'])==8 and len(environment['activities'])==8 and 'failure' not in environment
for a in environment['activities']:
 assert a['passed'] and a['samples'] and all(not s['competitors'] for s in a['samples'])
 for stage in ('before','after'):
  samples=a[stage][-3:];assert len(samples)==3
  assert all(0<=s['utilization']<=30 and float(s['power'].split()[0])<=15 for s in samples)
groups={}
for r in meta['runs']:
 assert r['exit_code']==0
 prefix,engine,phase=r['name'].split('-');d=read(clean/(r['name']+'.json'))
 assert d['execution']=='resident' and d['kv_type']=='f32' and d['warmup']==2 and d['repetitions']==5
 x=d['results'][0];assert x['test']=='tg128' and x['prefix_tokens']==int(prefix)
 assert x['graph_replays']==640 and x['uploads']==0 and x['downloads']==640
 assert x['kernel_launches']==(185600+(14080 if int(prefix) else 0))
 groups.setdefault((int(prefix),engine),[]).append((5,x['tokens_per_second'],x['stddev']))
record={'convention':'Fyodor throughput / reference throughput * 100; 100 is parity','self_comparison':[]}
for prefix in (0,512):
 before=pool(groups[prefix,'before']);after=pool(groups[prefix,'after'])
 record['self_comparison'].append({'prefix':prefix,'before':before,'after':after,'throughput_gain_percent':(after['tokens_per_second']/before['tokens_per_second']-1)*100})

(root/'confirmation-summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
