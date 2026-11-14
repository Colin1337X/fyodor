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
base=out.parent/'batch-graphs-20261114'
if len(sys.argv)!=2 or sys.argv[1] not in ('short','long'):raise SystemExit('Choose short|long')
context=sys.argv[1];integer=True;folder=out/context
if folder.exists():raise SystemExit('Refusing to overwrite profile evidence')
proofs=[base/'build.json',base/'model-memcheck-short/result.json',base/'boundary-matrix-audit.json',
        base/'recovery-v2-audit.json',base/f'generation-{context}-c17/result.json']
inputs=set(proofs)
for path in proofs:
    proof=json.loads(path.read_text())
    if proof.get('passed',proof.get('build_passed')) is not True:raise ValueError('Failed prerequisite: '+str(path))
    for name,expected in proof['sha256'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Changed prerequisite: '+name)
        inputs.add(p)
sys.path.insert(0,str(joint))
from model_identity import verify_gemma
identity=verify_gemma()
spec=importlib.util.spec_from_file_location('gate',out.parent/'gemma4-qkv-20261029/environment.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):raise SystemExit('Another GPU workload remains')
settings_path=base/f'generation-{context}-c17/result.json'
settings=json.loads(settings_path.read_text())['environment']
settings.update(NYA_CUDA_PROFILE='1',NYA_CUDA_PROFILE_FROM='25' if context=='short' else '8205',
                NYA_CUDA_INTERLEAVED_INTEGER='1' if integer else '0')
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(settings)
folder.mkdir()
command=[str(root/'build-cuda/batch-graphs-20261114/generation-c17.exe'),context]
inputs.update([Path(__file__).resolve(),settings_path,joint/'model_identity.py',Path(gate.__file__),Path(command[0])])
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
result=dict(accepted_benchmark=False,passed=False,command=command,environment=settings,model_identity=identity,
    limitations=['Profiling disables graphs; wall times cannot be used as throughput.',
                 'No continuous contamination acceptance; kernel timings locate costs only.',
                 'Aggregate target-plan profiling includes borrowed helper operations while profiling is active.'])
try:
    with (folder/'stdout').open('xb') as stdout,(folder/'stderr').open('xb') as stderr:
        child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
        (folder/'running.json').write_text(json.dumps(dict(process_id=child.pid,command=command)))
        try:result['exit_code']=child.wait()
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=15)
    if result['exit_code']:raise RuntimeError('Profile process failed')
    rows=[json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.startswith('{"round":')]
    modes=('ordinary','mtp-serial2','mtp-batch2','mtp-joint3')
    order=[(r,m) for r in range(3) for m in (reversed(modes) if r==1 else modes)]
    if [(r['round'],r['mode']) for r in rows]!=order or any(r['parity'] is not True or r['generated']!=32 or r['prompt']!=(25 if context=='short' else 8205) for r in rows):
        raise RuntimeError('Generation inventory or parity failed')
    trace=(folder/'stderr').read_text()
    selected=[int(v) for v in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)',trace)]
    if integer and (len(selected)!=3 or min(selected)<=0):raise RuntimeError('Integer kernel selection missing')
    if not integer and selected:raise RuntimeError('Unexpected integer kernel selection')
    result['integer_calls']=selected
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
    if 'CUDA native metadata Q4: selected' not in trace:raise ValueError('Metadata kernel not selected')
    for name,expected in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed input: '+name)
    result.update(passed=True,cases=cases)
finally:
    result['sha256']={**hashes,**{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'stdout',folder/'stderr')}}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
print(context,'profile complete; all twelve outputs match; no accepted timing')
