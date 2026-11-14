"""Build isolated C17/C23 batched-graph model binaries from verified inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from transform import transform

root=Path.cwd();out=Path(__file__).resolve().parent
base=out.parent/'integer-metadata-20261114'
request=out.parent/'batch-graph-request-20261114'
old_work=root/'build-cuda/integer-metadata-20261114'
work=root/'build-cuda/batch-graphs-20261114'
manifest=out/'build.json'
if work.exists() or manifest.exists(): raise SystemExit('Refusing to overwrite build')
inputs={Path(__file__).resolve(),out/'transform.py'}
for path in (base/'build.json',base/'model-memcheck-short/result.json',request/'memcheck-run.json'):
    proof=json.loads(path.read_text())
    if proof.get('build_passed',proof.get('passed')) is not True: raise ValueError(str(path))
    inputs.add(path)
    for name,expected in proof['sha256'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError(name)
        inputs.add(p)
    for name,expected in proof.get('evidence_sha256',{}).items():
        p=path.parent/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError(name)
        inputs.add(p)
candidate=root/'optcpp/graph/batch_request.cu';inputs.add(candidate)
before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
header=(old_work/'cuda_source.h').read_text()
marker='static const unsigned char nya_cuda_source[] = {'
if header.count(marker)!=1: raise ValueError('Unexpected embedded source')
start=header.index(marker)+len(marker);end=header.index('};',start)
raw=bytes(int(v.strip(),0) for v in header[start:end].split(',') if v.strip())
if raw[-1:]!=b'\0' or b'nya_decode_batch_request' in raw: raise ValueError('Unexpected source baseline')
raw=raw[:-1]+b'\n'+candidate.read_bytes()+b'\0'
header=header[:start]+','.join(str(v) for v in raw)+header[end:]
resident=transform((old_work/'resident.inc').read_text())
lifecycle=(old_work/'lifecycle.c').read_text()
anchor='            CHECK(after.downloads-before.downloads==1);'
if lifecycle.count(anchor)!=1: raise ValueError('Unexpected lifecycle anchor')
lifecycle=lifecycle.replace(anchor,anchor+'''
            if(getenv("NYA_CUDA_BATCH_GRAPHS") && !strcmp(getenv("NYA_CUDA_BATCH_GRAPHS"),"1")) {
                CHECK(after.graph_replays-before.graph_replays==1);
                CHECK(after.graph_captures-before.graph_captures<=1);
            }''')
work.mkdir()
(work/'cuda_source.h').write_text(header)
(work/'resident.inc').write_text(resident)
(work/'lifecycle.c').write_text(lifecycle)
(work/'cuda.c').write_bytes((old_work/'cuda.c').read_bytes())
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
commands=[]
for original in json.loads((base/'build.json').read_text())['commands']:
    command=[arg.replace(str(old_work),str(work)) for arg in original]
    subprocess.run(command,env=env,check=True);commands.append(command)
for name,expected in before.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected: raise ValueError(name)
outputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
manifest.write_text(json.dumps(dict(build_passed=True,gpu_validated=False,accepted_benchmark=False,
    commands=commands,sha256={**before,**outputs}),indent=2))
print('Batched graph C17/C23 model builds linked; GPU validation pending')
