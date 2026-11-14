"""Build isolated paired-row C17/C23 model binaries from verified evidence."""
import hashlib,json,os,subprocess
from pathlib import Path
from transform import provider,resident
root=Path.cwd();out=Path(__file__).resolve().parent
base=out.parent/'model-pair-rows-20261114';raw=out.parent/'q6-pair-four-20261114'
old=root/'build-cuda/model-pair-rows-20261114';work=root/'build-cuda/model-q6-pair-20261114'
if work.exists() or (out/'build.json').exists():raise SystemExit('Refusing overwrite')
inputs={Path(__file__).resolve(),out/'transform.py',root/'optcpp/matvec/q6_pair_rows_four.cu'}
proofs=[base/'build.json',base/'model-memcheck-short/result.json',base/'generation-long-c17/result.json',raw/'full-oracle.json']
proofs += [raw/(tool+'-run.json') for tool in ('memcheck','racecheck','synccheck')]
for path in proofs:
    proof=json.loads(path.read_text());inputs.add(path)
    if proof.get('passed',proof.get('build_passed')) is not True:raise ValueError(str(path))
    for key in ('sha256','input_sha256','evidence_sha256'):
        for name,h in proof.get(key,{}).items():
            p=path.parent/name if key=='evidence_sha256' else root/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError(name)
            inputs.add(p)
for name in ('timing-upload-sync.json','timing-reverse.json'):
    path=raw/name;proof=json.loads(path.read_text());inputs.add(path)
    script=raw/('probe.py' if 'upload' in name else 'probe_reverse.py');inputs.add(script)
    if hashlib.sha256(script.read_bytes()).hexdigest()!=proof['harness_sha256']:raise ValueError(name)
    selected=[r for r in proof['results'] if (r['columns'],r['rows']) in ((3840,2048),(8192,3840),(15360,3840))]
    if len(selected)!=3 or any(r['means_ms']['pair']>=r['means_ms']['integer'] for r in selected):raise ValueError('Measured shapes no longer improve')
def key(p):return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
before={key(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
header=(old/'cuda_source.h').read_text();marker='static const unsigned char nya_cuda_source[] = {'
if header.count(marker)!=1:raise ValueError('Embedded source marker')
start=header.index(marker)+len(marker);end=header.index('};',start)
source=bytes(int(v.strip(),0) for v in header[start:end].split(',') if v.strip())
if source[-1:]!=b'\0' or b'nya_q6_pair_rows_four' in source:raise ValueError('Unexpected embedded source')
source=source[:-1]+b'\n'+(root/'optcpp/matvec/q6_pair_rows_four.cu').read_bytes()+b'\0'
header=header[:start]+','.join(str(v) for v in source)+header[end:]
work.mkdir();(work/'cuda_source.h').write_text(header)
(work/'cuda.c').write_text(provider((old/'cuda.c').read_text()))
(work/'resident.inc').write_text(resident((old/'resident.inc').read_text()))
(work/'lifecycle.c').write_bytes((old/'lifecycle.c').read_bytes())
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
commands=[]
for original in json.loads((base/'build.json').read_text())['commands']:
    command=[arg.replace(str(old),str(work)) for arg in original]
    subprocess.run(command,env=env,check=True);commands.append(command)
for name,h in before.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed: '+name)
outputs={key(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
(out/'build.json').write_text(json.dumps(dict(build_passed=True,gpu_validated=False,accepted_benchmark=False,commands=commands,sha256={**before,**outputs}),indent=2))
print('Isolated C17/C23 paired-row builds passed; model validation pending')
