"""Summarize completed diagnostic runs without asserting clean performance acceptance."""
from pathlib import Path
import json,statistics,struct,math
out=Path(__file__).resolve().parent;work=Path('D:/fyodor-validation/training-profile-20261010')
r=json.loads((out/'baseline.json').read_text());runs=[v for v in r['runs'] if 'measured_mean' in v];summary={'completed_runs':len(runs),'expected_runs':18,'performance_accepted':False,'reason':'NVIDIA overlays respawned; reference installation overlaps later runs','groups':[],'parity':[]}
for context in (8,32,128):
 for compute in ('cpu','cuda'):
  group=[v for v in runs if v['name'].split('-')[1:]==[str(context),compute]]
  if not group:continue
  assert len({v['checkpoint_sha256'] for v in group})==1,'Same-device nondeterminism'
  values=[v['measured_mean']['tokens_per_second'] for v in group]
  summary['groups'].append({'context':context,'compute':compute,'repetitions':len(group),'tokens_per_second_mean':statistics.mean(values),'between_run_sd':statistics.stdev(values) if len(values)>1 else None,'phase_means_ms':{k:statistics.mean(v['measured_mean'][k] for v in group) for k in group[0]['measured_mean'] if k.endswith('_ms')}})
 a=work/f'0-{context}-cpu.ckpt';b=work/f'0-{context}-cuda.ckpt'
 if not a.exists() or not b.exists():continue
 a=a.read_bytes()[16:];b=b.read_bytes()[16:];assert a[:48]==b[:48];count=struct.unpack_from('<Q',a,8)[0];offset=48;maximum=0
 for parameter in range(count):
  rows,columns=struct.unpack_from('<QQ',a,offset);assert a[offset:offset+16]==b[offset:offset+16];offset+=16
  for state in range(4):
   size=rows*columns*4
   for (expected,),(actual,) in zip(struct.iter_unpack('<f',a[offset:offset+size]),struct.iter_unpack('<f',b[offset:offset+size])):
    error=abs(actual-expected)/(1+abs(expected));assert math.isfinite(actual) and error<=1e-5,(context,parameter,state,error);maximum=max(maximum,error)
   offset+=size
 assert offset+8==len(a)==len(b)
 summary['parity'].append({'context':context,'maximum_scaled_state_error':maximum,'tolerance':1e-5})
(out/'diagnostic-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
