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
clean=root/'clean';meta=read(clean/'metadata.json');environment=read(clean/'environment.json')
assert len(meta['runs'])==12 and len(environment['activities'])==12 and 'failure' not in environment
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
 assert x['kernel_launches']==((157440 if engine=='after' else 185600)+(14080 if int(prefix) else 0))
 groups.setdefault((int(prefix),engine),[]).append((5,x['tokens_per_second'],x['stddev']))
record={'convention':'Fyodor throughput / reference throughput * 100; 100 is parity','self_comparison':[]}
for prefix in (0,512,1024):
 before=pool(groups[prefix,'before']);after=pool(groups[prefix,'after'])
 record['self_comparison'].append({'prefix':prefix,'before':before,'after':after,'throughput_gain_percent':(after['tokens_per_second']/before['tokens_per_second']-1)*100})

ref=root/'reference-20261015'
meta=read(ref/'metadata.json');assert len(meta['runs'])==8
groups={}
for r in meta['runs']:
 assert r['exit_code']==0 and 'gpu_idle_after' in r and not r.get('contaminated') and not r.get('idle_failure')
 for stage in ('gpu_idle_before','gpu_idle_after'):
  samples=r[stage][-3:];assert len(samples)==3
  assert all(0<=int(s.split('%')[0])<=30 and float(s.split(',')[5].split()[0])<=15 for s in samples)
 resources=read(ref/(r['name']+'-resources.json'))
 assert resources and all(s.get('whole_gpu') and 'competing_compute_processes' in s and not s['competing_compute_processes'] for s in resources)
 _,prefix,engine,phase=r['name'].split('-');d=read(ref/(r['name']+'.json'))
 assert d['kv_type']=='f32' and d['warmup']==2 and d['repetitions']==5
 assert engine=='llama' or d['execution']=='resident'
 assert len(d['results'])==1
 x=d['results'][0];assert x['test']=='tg128' and x['prefix_tokens']==int(prefix)
 if engine=='after':assert x['graph_replays']==640 and x['kernel_launches']==157440+(14080 if int(prefix) else 0)
 groups.setdefault((int(prefix),engine),[]).append((5,x['tokens_per_second'],x['stddev']))
record['reference_comparison']=[]
for prefix in (0,1024):
 for engine in ('after','llama'):assert len(groups[prefix,engine])==2
 a=pool(groups[prefix,'after']);b=pool(groups[prefix,'llama'])
 record['reference_comparison'].append({'prefix':prefix,'fyodor':a,'llama':b,'percentage':100*a['tokens_per_second']/b['tokens_per_second']})
(root/'summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
