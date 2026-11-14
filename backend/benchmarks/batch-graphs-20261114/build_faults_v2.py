"""Compile separate failure-injection binaries without touching active inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();out=Path(__file__).resolve().parent
old=root/'build-cuda/batch-graphs-20261114'
work=root/'build-cuda/batch-graph-faults-v2-20261114'
manifest=out/'fault-build-v2.json'
if work.exists() or manifest.exists():raise SystemExit('Refusing to overwrite fault build')
proof=json.loads((out/'build.json').read_text())
if proof.get('build_passed') is not True:raise ValueError('Missing base build')
inputs={out/'build.json',out/'fault_hooks_v2.h',Path(__file__).resolve()}
for name,expected in proof['sha256'].items():
    p=root/name
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(name)
    inputs.add(p)
before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
resident=(old/'resident.inc').read_text()
for field in ('capture_begin','capture_end','graph_instantiate','graph_launch','graph_free'):
    anchor=f'c->{field}('
    if anchor not in resident:raise ValueError('Missing call: '+field)
    resident=resident.replace(anchor,f'nya_test_{field}(c,')
# Inject only the requested operation; retain all original cleanup paths.
def replace_once(old_text,new_text):
    global resident
    if resident.count(old_text)!=1:raise ValueError('Nonunique injection anchor: '+old_text)
    resident=resident.replace(old_text,new_text)
resident=resident.replace('p->cuda->graph_exec_free(', 'nya_test_graph_exec_free(p->cuda,')
replace_once('if(plan_grid(p,p->batch_request,1,1,1,1,args))',
             'if(nya_test_graph_fault("request") || plan_grid(p,p->batch_request,1,1,1,1,args))')
replace_once('RUN(plan_launch(p,p->graph_gather,(d*count+255)/256,gather));',
             'RUN(plan_launch(p,p->graph_gather,(d*count+255)/256,gather));\n            if(nya_test_graph_fault("lowering")) goto failure;')
replace_once('if (c->stream_sync(c->stream) != CUDA_SUCCESS) return -1;',
             'if (c->stream_sync(c->stream) != CUDA_SUCCESS || nya_test_graph_fault("sync")) return -1;')
replace_once('if (c->download(output, p->logits, elements * sizeof(float)) != CUDA_SUCCESS) return -1;',
             'if (c->download(output, p->logits, elements * sizeof(float)) != CUDA_SUCCESS || nya_test_graph_fault("readback")) return -1;')
cuda=(old/'cuda.c').read_text()
anchor='#include "resident.inc"'
if cuda.count(anchor)!=1:raise ValueError('Missing resident include')
cuda=cuda.replace(anchor,'#include "fault_hooks_v2.h"\n'+anchor)
work.mkdir()
for name in ('cuda_source.h','lifecycle.c'):(work/name).write_bytes((old/name).read_bytes())
(work/'fault_hooks_v2.h').write_bytes((out/'fault_hooks_v2.h').read_bytes())
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
