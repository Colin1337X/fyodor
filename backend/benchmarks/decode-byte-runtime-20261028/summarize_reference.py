"""Verify the complete matched protocol and pool all forty samples per cell."""
from pathlib import Path
import json,math,re
root=Path(__file__).resolve().parent;path=root/'reference'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pool(rows):
 n=sum(r[0] for r in rows);mean=sum(r[0]*r[1] for r in rows)/n
 var=sum((r[0]-1)*r[2]**2+r[0]*(r[1]-mean)**2 for r in rows)/(n-1)
 return {'samples':n,'tokens_per_second':mean,'sample_stdev':math.sqrt(var)}
meta=read(path/'metadata.json');environment=read(path/'environment.json')
assert len(meta['runs'])==len(environment['activities'])==12 and 'failure' not in environment
assert meta['model_sha256']=='9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0'
groups={}
for run,activity in zip(meta['runs'],environment['activities']):
 assert run['exit_code']==0 and activity['passed'] and activity['samples']
 assert all(not s['competitors'] for s in activity['samples'])
 for stage in ('before','after'):
  samples=activity[stage][-3:];assert len(samples)==3
  assert all(0<=s['utilization']<=30 and float(s['power'].split()[0])<=15 for s in samples)
 prefix,engine,phase=run['name'].split('-');prefix=int(prefix)
 d=read(path/(run['name']+'.json'));x=d['results'][0]
 assert d['backend']=='cuda' and d['kv_type']=='f32' and d['warmup']==10 and d['repetitions']==20
 assert len(d['results'])==1 and x['test']=='tg128' and x['prefix_tokens']==prefix and x['tokens']==128
 assert run['fdpp']=='short' and run['fixed']=='1' and run['integer']=='1' and run['integer_dot']=='1' and run['integer_q6']=='1' and run['byte_gate']==('1' if engine=='fixed' else '0')
 if engine=='llama':
  assert d['flash_attention']==False and d['batch']==512 and d['threads']==6
  match=re.search(rb'offloaded (\d+)/(\d+) layers to GPU',(path/(run['name']+'.log')).read_bytes())
  assert match and match[1]==match[2]
 else:
  assert d['execution']=='resident' and x['graph_replays']==x['downloads']==2560 and x['uploads']==0
 groups.setdefault((prefix,engine),[]).append((20,x['tokens_per_second'],x['stddev']))
result={'convention':'Fyodor throughput / llama.cpp throughput * 100; 100 is parity',
        'policy':'integer scale-product remains opt-in; selective FDPP used by both Fyodor variants','comparisons':[]}
for prefix in (0,1024):
 row={'prefix':prefix}
 for engine in ('baseline','fixed','llama'):
  assert len(groups[prefix,engine])==2;row[engine]=pool(groups[prefix,engine])
 for engine in ('baseline','fixed'):
  row[engine+'_llama_percent']=100*row[engine]['tokens_per_second']/row['llama']['tokens_per_second']
 row['fixed_baseline_gain_percent']=100*(row['fixed']['tokens_per_second']/row['baseline']['tokens_per_second']-1)
 result['comparisons'].append(row)
(root/'reference-summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
