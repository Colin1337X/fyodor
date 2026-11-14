"""Compile recovery fixtures against the isolated fault-injection objects."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/batch-graph-faults-v2-20261114'
manifest=out/'recovery-build-v2.json'
if manifest.exists():raise SystemExit('Refusing to overwrite recovery build')
proof=json.loads((out/'fault-build-v2.json').read_text())
if proof.get('build_passed') is not True:raise ValueError('Missing fault build')
for name,expected in proof['sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError(name)
source=out/'recovery_v2.c'
inputs=[source,Path(__file__).resolve(),out/'fault-build-v2.json']
before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
commands=[];outputs=[]
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
for standard in ('c17','c23'):
    old=work/f'lifecycle-{standard}.exe';exe=work/f'recovery-{standard}.exe'
    if exe.exists():raise SystemExit('Refusing to overwrite binary')
    matches=[c for c in proof['commands'] if str(old) in c]
    if len(matches)!=1:raise ValueError('Expected one lifecycle command')
    command=[str(source) if arg==str(work/'lifecycle.c') else str(exe) if arg==str(old) else arg for arg in matches[0]]
    subprocess.run(command,env=env,check=True);commands.append(command);outputs.append(exe)
for name,expected in {**proof['sha256'],**before}.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Input changed: '+name)
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
manifest.write_text(json.dumps(dict(build_passed=True,gpu_validated=False,accepted_benchmark=False,
    commands=commands,sha256={**proof['sha256'],**before,**hashes}),indent=2))
print('C17/C23 recovery fixtures compiled; GPU recovery remains untested')
