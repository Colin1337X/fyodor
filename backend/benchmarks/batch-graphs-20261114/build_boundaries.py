"""Compile topology boundary fixtures without starting a GPU workload."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();out=Path(__file__).resolve().parent
destination=out/'boundary-build.json'
if destination.exists(): raise SystemExit('Refusing to overwrite boundary build')
proof=json.loads((out/'build.json').read_text())
if proof['build_passed'] is not True: raise ValueError('Missing model build')
for name,expected in proof['sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected: raise ValueError(name)
source=out/'boundaries.c'
work=root/'build-cuda/batch-graphs-20261114'
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
commands=[];outputs=[]
for standard in ('c17','c23'):
    old=work/f'lifecycle-{standard}.exe';exe=work/f'boundaries-{standard}.exe'
    if exe.exists(): raise SystemExit('Refusing to overwrite binary')
    matches=[c for c in proof['commands'] if str(old) in c]
    if len(matches)!=1: raise ValueError('Expected one lifecycle link command')
    command=[str(source) if arg==str(work/'lifecycle.c') else str(exe) if arg==str(old) else arg for arg in matches[0]]
    subprocess.run(command,env=env,check=True);commands.append(command);outputs.append(exe)
inputs=[source,Path(__file__).resolve(),out/'build.json',*outputs]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
destination.write_text(json.dumps(dict(build_passed=True,gpu_validated=False,accepted_benchmark=False,
    commands=commands,sha256={**proof['sha256'],**hashes}),indent=2))
print('C17/C23 topology boundary fixtures compiled; no GPU execution')
