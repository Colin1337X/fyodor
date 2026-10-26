"""Summarize every post-cleanup observation; exclude both earlier attempts."""
from pathlib import Path
import json,math,statistics
root=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pool(rows):
 n=sum(r[0] for r in rows);mean=sum(r[0]*r[1] for r in rows)/n
 variance=sum((r[0]-1)*r[2]**2+r[0]*(r[1]-mean)**2 for r in rows)/(n-1)
 return {'samples':n,'tokens_per_second':mean,'sample_stdev':math.sqrt(variance)}
result={'excluded':['clean','longer-short'],'comparisons':[],'monitor_diagnostics':[]}
for prefix,folder in ((0,'clean-longer-short'),(1024,'clean-longer-long')):
 path=root/folder;meta=read(path/'metadata.json');env=read(path/'environment.json')
 assert len(meta['runs'])==len(env['activities'])==4 and 'failure' not in env
 groups={'before':[],'after':[]}
 for run,activity in zip(meta['runs'],env['activities']):
  assert run['exit_code']==0 and activity['passed'] and activity['samples']
  assert all(not s['competitors'] for s in activity['samples'])
  for stage in ('before','after'):
   samples=activity[stage][-3:];assert len(samples)==3
   assert all(0<=s['utilization']<=30 and float(s['power'].split()[0])<=15 for s in samples)
  _,engine,_=run['name'].split('-');data=read(path/(run['name']+'.json'))
  assert data['execution']=='resident' and data['kv_type']=='f32'
  assert data['repetitions']==20 and data['warmup']==10
  assert run['fdpp']==('1' if engine=='after' else '0')
  row=data['results'][0]
  assert row['test']=='tg128' and row['prefix_tokens']==prefix
  assert row['graph_replays']==row['downloads']==2560 and row['uploads']==0
  groups[engine].append((20,row['tokens_per_second'],row['stddev']))
  samples=activity['samples'];busy=[s['cpu']['busy_percent'] for s in samples if s['cpu'].get('busy_percent') is not None]
  result['monitor_diagnostics'].append({'run':run['name'],'scope':'includes loading and warmup; not per-repetition causality',
    'cpu_busy_mean':statistics.mean(busy),'cpu_busy_max':max(busy),
    'sm_clocks':sorted(set(s['sm_clock'] for s in samples)),
    'memory_clocks':sorted(set(s['memory_clock'] for s in samples)),
    'pstates':sorted(set(s['pstate'] for s in samples))})
 before,after=pool(groups['before']),pool(groups['after'])
 result['comparisons'].append({'prefix':prefix,'native':before,'fdpp_short_split':after,
    'throughput_gain_percent':100*(after['tokens_per_second']/before['tokens_per_second']-1)})
(root/'clean-summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result['comparisons'],indent=2))
