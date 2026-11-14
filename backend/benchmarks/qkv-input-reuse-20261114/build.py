"""Build isolated C17/C23 QKV-reuse binaries from verified graph inputs."""
import hashlib,json,os,subprocess
from pathlib import Path
from transform import transform
root=Path.cwd();out=Path(__file__).resolve().parent;base=out.parent/'batch-graphs-20261114'
old=root/'build-cuda/batch-graphs-20261114';work=root/'build-cuda/qkv-input-reuse-20261114'
if work.exists() or (out/'build.json').exists():raise SystemExit('Refusing build overwrite')
inputs={Path(__file__).resolve(),out/'transform.py'}
for path in (base/'build.json',base/'model-memcheck-short/result.json',base/'boundary-matrix-audit.json',base/'recovery-v2-audit.json'):
    proof=json.loads(path.read_text());inputs.add(path)
    if proof.get('passed',proof.get('build_passed')) is not True:raise ValueError(str(path))
    for name,h in proof['sha256'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError(name)
        inputs.add(p)
before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
resident=transform((old/'resident.inc').read_text());work.mkdir()
for name in ('cuda.c','cuda_source.h','lifecycle.c'):(work/name).write_bytes((old/name).read_bytes())
(work/'resident.inc').write_text(resident)
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
commands=[]
for original in json.loads((base/'build.json').read_text())['commands']:
    command=[arg.replace(str(old),str(work)) for arg in original]
    subprocess.run(command,env=env,check=True);commands.append(command)
for name,h in before.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed input: '+name)
outputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
(out/'build.json').write_text(json.dumps(dict(build_passed=True,gpu_validated=False,accepted_benchmark=False,commands=commands,sha256={**before,**outputs}),indent=2))
print('C17/C23 QKV reuse binaries built; GPU validation pending')
