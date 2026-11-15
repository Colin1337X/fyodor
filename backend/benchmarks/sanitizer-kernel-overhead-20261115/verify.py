"""Verify retained overhead controls without initializing CUDA."""
import hashlib,json,re,statistics
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
labels=['Baseline','NoMatch','CubinBaseline','CubinNoMatch','CubinNoPatching','CubinPatchedRepeat','CubinUnpatchedRepeat','CubinSubstring','CubinExclude']
records=[];evidence={}
for label in labels:
 p=out/(label+'.json');r=json.loads(p.read_text())
 assert r['passed'] and not r['accepted_benchmark'] and len(r['results'])==9,label
 for name,h in r['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 for i,row in enumerate(r['results']):
  assert row['numerical_passed'] and len(row['gpu_ms'])==5 and len(row['wall_ms'])==5
  assert row['unused_allocation_bytes']==[0,8*1024**3,0][i//3]
  assert row['kernel']==['nya_matvec','nya_dot_12','nya_q4_pair_rows'][i%3]
 if 'Baseline' not in label:
  text=(out/(label+'.stdout')).read_text()+(out/(label+'.stderr')).read_text()
  assert re.findall(r'^========= ERROR SUMMARY:.*$',text,re.M)==['========= ERROR SUMMARY: 0 errors'],label
 command=out/(label+'-command.json')
 if command.exists():
  c=json.loads(command.read_text());assert c['passed'] and c['exit_code']==0 and not c['safety_evidence']
  for name,h in c['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 records.append(dict(label=label,kernels=[dict(kernel=row['kernel'],unused_allocation_bytes=row['unused_allocation_bytes'],mean_gpu_ms=row['mean_gpu_ms'],sample_stddev_gpu_ms=statistics.stdev(row['gpu_ms'])) for row in r['results']]))
 for suffix in ('.json','.stdout','.stderr','-command.json'):
  f=out/(label+suffix)
  if f.exists():evidence[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
model=root/'backend/benchmarks/lifecycle-progress-20261115/OverheadNoPatching5/result.json';r=json.loads(model.read_text())
assert r['passed'] and r['binary_patching_disabled'] and not r['safety_evidence'] and not r['accepted_benchmark']
for name,h in r['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
evidence[str(model.relative_to(root))]=hashlib.sha256(model.read_bytes()).hexdigest()
result=dict(passed=True,accepted_benchmark=False,safety_evidence=False,records=records,model_unpatched_elapsed_s=r['elapsed_s'],sha256=evidence)
with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2)
print('PASS: nine kernel runs, 81 case records, 405 GPU samples, model control, all evidence hashes')
