"""Profile the validated joint-base binary; never accept profiler throughput."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root=Path.cwd();out=Path(__file__).resolve().parent
joint=out.parent/'mtp-joint-base-v2-20261114'
if len(sys.argv)!=2 or sys.argv[1] not in ('short','long'):raise SystemExit('Choose short or long')
context=sys.argv[1];folder=out/context
if folder.exists():raise SystemExit('Refusing to overwrite profile evidence')
proofs=[joint/'generation-build.json',joint/'model-memcheck-short/result.json']
for path in proofs:
    proof=json.loads(path.read_text())
    if 'passed' in proof and proof['passed'] is not True:raise SystemExit('Sanitizer did not pass')
    for name,expected in proof['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise SystemExit(f'Changed prerequisite: {name}')
sys.path.insert(0,str(joint))
from model_identity import verify_gemma
identity=verify_gemma()
spec=importlib.util.spec_from_file_location('gate',out.parent/'gemma4-qkv-20261029/environment.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):raise SystemExit('Another GPU workload remains')
settings_path=joint/'monitored-short-A/timing.json'
settings=json.loads(settings_path.read_text())['environment']
settings.update(NYA_CUDA_PROFILE='1',NYA_CUDA_PROFILE_FROM='25' if context=='short' else '8205')
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(settings)
folder.mkdir()
command=[str(root/'build-cuda/joint-base-generation-v2-20261114/generation.exe'),context]
result=dict(accepted_benchmark=False,passed=False,command=command,environment=settings,model_identity=identity,
    limitations=['Profiling disables graphs; wall times cannot be used as throughput.',
                 'No continuous contamination acceptance; kernel timings locate costs only.',
                 'Aggregate target-plan profiling includes borrowed helper operations while profiling is active.'])
try:
    with (folder/'stdout').open('xb') as stdout,(folder/'stderr').open('xb') as stderr:
        result['exit_code']=subprocess.run(command,env=env,stdout=stdout,stderr=stderr).returncode
    if result['exit_code']:raise RuntimeError('Profile process failed')
    rows=[json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.startswith('{"round":')]
    modes=('ordinary','mtp-serial2','mtp-batch2','mtp-joint3')
    order=[(r,m) for r in range(3) for m in (reversed(modes) if r==1 else modes)]
    if [(r['round'],r['mode']) for r in rows]!=order or any(r['parity'] is not True or r['generated']!=32 for r in rows):
        raise RuntimeError('Generation inventory or parity failed')
    cases=[];current=None
    for line in (folder/'stderr').read_text().splitlines():
        match=re.fullmatch(r'CASE round=(\d+) mode=(\S+)',line)
        if match:
            current=dict(round=int(match[1]),mode=match[2],matrices=[],target_attention_ms=0.0);cases.append(current)
        if line.startswith('CUDA kernel GPU milliseconds:'):
            current['gpu_ms']={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
        match=re.fullmatch(r'CUDA matrix profile: path=(\S+) type=(\d+) rows=(\d+) columns=(\d+) batch=(\d+) calls=(\d+) milliseconds=([\d.]+)',line)
        if match:current['matrices'].append(dict(path=match[1],type=int(match[2]),rows=int(match[3]),columns=int(match[4]),batch=int(match[5]),calls=int(match[6]),milliseconds=float(match[7])))
        match=re.search(r'CUDA attention profile: .* milliseconds=([\d.]+)',line)
        if match:current['target_attention_ms']+=float(match[1])
    if [(c['round'],c['mode']) for c in cases]!=order or any('gpu_ms' not in c or not c['matrices'] for c in cases):
        raise RuntimeError('Incomplete kernel profile')
    result.update(passed=True,cases=cases)
finally:
    inputs=proofs+[Path(__file__),settings_path,joint/'model_identity.py',folder/'stdout',folder/'stderr']
    result['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
print(context,'profile complete; all twelve outputs match; no accepted timing')
