"""GPU topology transitions with F32 fixtures; no accepted timing."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=4 or sys.argv[1] not in ('c17','c23') or sys.argv[2] not in ('256','512') or sys.argv[3] not in ('0','1'):
    raise SystemExit('Usage: run_boundaries.py c17|c23 256|512 0|1')
standard,width,ring=sys.argv[1:]
folder=out/f'boundaries-{standard}-{width}-ring{ring}'
if folder.exists(): raise SystemExit('Refusing to overwrite boundary evidence')
inputs={Path(__file__).resolve()}
for path in (out/'boundary-build.json',out/'lifecycle-c17-8192/result.json'):
    if not path.exists(): raise SystemExit('Wait for completed prerequisite: '+str(path))
    proof=json.loads(path.read_text());inputs.add(path)
    if proof.get('build_passed',proof.get('passed')) is not True: raise ValueError(str(path))
    for name,expected in proof['sha256'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError(name)
        inputs.add(p)
sys.path.insert(0,str(out.parent/'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
settings_path=out/'lifecycle-c17-257/result.json'
settings=json.loads(settings_path.read_text())['environment']
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(settings);env['PYTHONDONTWRITEBYTECODE']='1'
inputs.update([settings_path,out.parent/'gemma4-qkv-20261029/environment.py'])
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
fixture=root/f'build-cuda/batch-graphs-20261114/boundary-{standard}-{width}-ring{ring}.gguf'
if fixture.exists(): raise SystemExit('Refusing to overwrite fixture')
command=[str(root/f'build-cuda/batch-graphs-20261114/boundaries-{standard}.exe'),str(fixture),width,ring]
folder.mkdir();result=dict(passed=False,accepted_benchmark=False,command=command,environment=settings)
try:
    with (folder/'stdout').open('xb') as stdout,(folder/'stderr').open('xb') as stderr:
        result['exit_code']=subprocess.run(command,env=env,stdout=stdout,stderr=stderr).returncode
    if result['exit_code']!=0: raise ValueError('Boundary executable failed')
    output=(folder/'stdout').read_text();trace=(folder/'stderr').read_text()
    cases=re.findall(r'^reset=(\d+) position=(\d+) count=(\d+) trial=(\d+) parity=1$',output,re.M)
    expected=[(str(reset),str(pos),str(count),str(trial)) for reset in range(2)
              for pos,count in ((250,4),(254,2),(256,2),(258,4),(8190,2),(8192,4),(8196,2))
              for trial in range(3)]
    if cases!=expected: raise ValueError('Incomplete/reordered boundary inventory')
    graphs=re.findall(r'CUDA batched graph captures: (\d+) replays: (\d+)',trace)
    if graphs!=[('6','42')]: raise ValueError('Expected six captured topologies and 42 replays')
    for name,expected_hash in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected_hash: raise ValueError(name)
    result.update(passed=True,cases=cases,graph_counts=graphs)
finally:
    paths=[p for p in (folder/'stdout',folder/'stderr',fixture) if p.exists()]
    result['sha256']={**hashes,**{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
print('42 boundary/reset replays passed',standard,width,ring,'; no timing acceptance')
