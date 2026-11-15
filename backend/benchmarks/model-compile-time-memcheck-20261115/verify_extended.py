import hashlib,json
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
proofs=[out/'CheckC23Prefix257/result.json',out/'AttentionModelModule/result.json',out/'long-trace-tests.json']
records=[]
for p in proofs:
 r=json.loads(p.read_text());assert r['passed'],p
 for name,h in r['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 records.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
paths=[out/name for name in ('run_v2.py','run_long.py','long_trace.py','test_long_trace.py','attention_common.py','check_attention.py','run_attention.py')]
result=dict(passed=True,accepted_benchmark=False,long_model_gate_satisfied=False,proofs=records,prepared_long_runner_executed=False,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
with (out/'extended-verification.json').open('x') as f:json.dump(result,f,indent=2)
print('PASS: prefix-257 model, 24 instrumented attention cases, 47 adversarial trace cases and hashes')
