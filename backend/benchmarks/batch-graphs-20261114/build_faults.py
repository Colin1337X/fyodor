"""Compile separate failure-injection binaries without touching active inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();out=Path(__file__).resolve().parent
old=root/'build-cuda/batch-graphs-20261114'
work=root/'build-cuda/batch-graph-faults-20261114'
manifest=out/'fault-build.json'
if work.exists() or manifest.exists():raise SystemExit('Refusing to overwrite fault build')
proof=json.loads((out/'build.json').read_text())
if proof.get('build_passed') is not True:raise ValueError('Missing base build')
inputs={out/'build.json',out/'fault_hooks.h',Path(__file__).resolve()}
for name,expected in proof['sha256'].items():
    p=root/name
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(name)
    inputs.add(p)
before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
resident=(old/'resident.inc').read_text()
for field in ('capture_begin','capture_end','graph_instantiate','graph_launch'):
    anchor=f'c->{field}('
    if anchor not in resident:raise ValueError('Missing call: '+field)
    resident=resident.replace(anchor,f'nya_test_{field}(c,')
cuda=(old/'cuda.c').read_text()
anchor='#include "resident.inc"'
if cuda.count(anchor)!=1:raise ValueError('Missing resident include')
cuda=cuda.replace(anchor,'#include "fault_hooks.h"\n'+anchor)
work.mkdir()
for name in ('cuda_source.h','lifecycle.c'):(work/name).write_bytes((old/name).read_bytes())
(work/'fault_hooks.h').write_bytes((out/'fault_hooks.h').read_bytes())
(work/'resident.inc').write_text(resident)
(work/'cuda.c').write_text(cuda)
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
commands=[]
for original in proof['commands']:
    command=[arg.replace(str(old),str(work)) for arg in original]
    subprocess.run(command,env=env,check=True);commands.append(command)
for name,expected in before.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Input changed: '+name)
outputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
manifest.write_text(json.dumps(dict(build_passed=True,gpu_validated=False,accepted_benchmark=False,
    commands=commands,sha256={**before,**outputs}),indent=2))
print('C17/C23 fault binaries compiled; injected recovery has NOT run')
