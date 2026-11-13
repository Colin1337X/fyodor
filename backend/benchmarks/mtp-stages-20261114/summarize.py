"""Aggregate diagnostic stage wall time; never use it as accepted throughput."""
import json,math,re,statistics
from pathlib import Path
out=Path(__file__).resolve().parent
meta=json.loads((out/'profile.json').read_text())
assert len(meta['runs'])==2 and all(r['exit_code']==0 for r in meta['runs'])
names=('setup','query','attention','attn_output','ffn','head','recurrence','softcap','validation')
results={}
for context in ('short','long'):
 rows=[json.loads(line) for line in (out/(context+'.stdout')).read_text().splitlines() if line.startswith('{"round":')]
 assert len(rows)==12 and all(r['parity'] and r['generated']==32 for r in rows)
 cases={};current=None
 for line in (out/(context+'.stderr')).read_text().splitlines():
  match=re.fullmatch(r'CASE round=(\d+) mode=(\S+)',line)
  if match:
   key=(int(match[1]),match[2]);assert key not in cases
   current={'calls':[]};cases[key]=current
  if line.startswith('MTP stages: '):
   values={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
   assert set(values)==set(names) and all(math.isfinite(v) and v>=0 for v in values.values())
   current['calls'].append(values)
  match=re.fullmatch(r'Generation assistant profile: calls=(\d+) forward_ms=([\d.]+)',line)
  if match:
   assert len(current['calls'])==int(match[1])
   current['forward_ms']=float(match[2])
 assert set(cases)=={(r,m) for r in range(3) for m in ('ordinary','mtp-serial2','mtp-batch2','mtp-batch4')}
 modes={}
 for mode in ('mtp-serial2','mtp-batch2','mtp-batch4'):
  observations=[]
  for round_index in (1,2):
   case=cases[(round_index,mode)];assert 'forward_ms' in case and case['calls']
   totals={name:sum(call[name] for call in case['calls']) for name in names}
   observations.append({'round':round_index,'calls':len(case['calls']),'stage_ms':totals,'forward_ms':case['forward_ms'],'outside_stage_ms':case['forward_ms']-sum(totals.values())})
  modes[mode]={'observations':observations,'mean_stage_ms':{name:statistics.mean(o['stage_ms'][name] for o in observations) for name in names},'mean_forward_ms':statistics.mean(o['forward_ms'] for o in observations)}
  modes[mode]['stdev_stage_ms']={name:statistics.stdev(o['stage_ms'][name] for o in observations) for name in names}
 for round_index in range(3):assert not cases[(round_index,'ordinary')]['calls']
 results[context]=modes
summary={'accepted_llama_score':False,'contract':'Assistant QPC stage wall times, including synchronous transfers and assisted GPU calls. Three generation rounds; round zero excluded. Logging and outer overhead are not charged to stages. No CUDA event profiling. Not isolated kernel time or throughput.','contexts':results}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({context:{mode:row['mean_stage_ms'] for mode,row in modes.items()} for context,modes in results.items()},indent=2))
