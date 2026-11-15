"""Validate compile-time instrumentation evidence without starting CUDA."""
import hashlib,json
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
kernel=root/'backend/benchmarks/sanitizer-kernel-overhead-20261115'
paths=[kernel/'CompileTimeAll.json',kernel/'CompileTimeAll-command.json',kernel/'CompileTimeNoMatch.json',kernel/'CompileTimeNoMatch-command.json',kernel/'compile-time-fault-checks.json',kernel/'shared-fault-checks.json',out/'v2-build.json',out/'CheckC17/result.json',out/'CheckC23/result.json']
records=[]
for p in paths:
 r=json.loads(p.read_text())
 assert r.get('passed',r.get('build_passed',r.get('coverage_equivalent'))),p
 for name,h in r['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 if p.name=='result.json':
  assert r['compile_time_instrumentation'] and not r['filtered'] and not r['accepted_benchmark'] and len(r['module_records'])==1
 records.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
with (out/'verification.json').open('x') as f:json.dump(dict(passed=True,accepted_benchmark=False,long_candidate_gate_satisfied=False,records=records),f,indent=2)
print('PASS: raw kernel controls, global/shared fault detection, strict builds, C17/C23 model checks and input hashes')
