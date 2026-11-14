"""Isolated metadata-kernel model build on the validated readback baseline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
base = out.parent / 'logit-readback-20261114'
old_work = root / 'build-cuda/logit-readback-20261114'
work = root / 'build-cuda/integer-metadata-20261114'
manifest = out / 'build.json'
if work.exists() or manifest.exists():
    raise SystemExit('Refusing to overwrite model build')
inputs = {Path(__file__).resolve(), out/'memcheck-run.json',
          base/'build.json', base/'model-memcheck-short/result.json'}
for path in (base/'build.json', base/'model-memcheck-short/result.json'):
    proof = json.loads(path.read_text())
    assert proof.get('build_passed',proof.get('passed')) is True
    for name, expected in proof['sha256'].items():
        p = root/name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
        inputs.add(p)
proof = json.loads((out/'memcheck-run.json').read_text())
assert proof['passed'] is True and proof['exit_code'] == 0
for name, expected in proof['input_sha256'].items():
    p = Path(name)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
    inputs.add(p)
for name, expected in proof['evidence_sha256'].items():
    p = out/name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
    inputs.add(p)
candidate = root/'optcpp/matvec/integer_metadata.cu'
inputs.add(candidate)
before = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
header = (old_work/'cuda_source.h').read_text()
marker = 'static const unsigned char nya_cuda_source[] = {'
assert header.count(marker) == 1
start = header.index(marker)+len(marker); end = header.index('};',start)
raw = bytes(int(v.strip(),0) for v in header[start:end].split(',') if v.strip())
assert raw[-1:] == b'\0' and b'nya_integer_metadata_12_4' not in raw
raw = raw[:-1]+b'\n'+candidate.read_bytes()+b'\0'
header = header[:start]+','.join(str(v) for v in raw)+header[end:]
host = (old_work/'cuda.c').read_text()
old = '''if(c->function_get(&candidate,c->module,"nya_integer_launch_8_3")==CUDA_SUCCESS)
                    c->interleaved_integer_q4=candidate;'''
assert host.count(old) == 1
host = host.replace(old, '''if(c->function_get(&candidate,c->module,"nya_integer_metadata_12_4")==CUDA_SUCCESS) {
                    c->interleaved_integer_q4=candidate;
                    fprintf(stderr,"CUDA native metadata Q4: selected\\n");
                }''')
work.mkdir()
(work/'cuda_source.h').write_text(header)
(work/'cuda.c').write_text(host)
for name in ('resident.inc','lifecycle.c'):
    (work/name).write_bytes((old_work/name).read_bytes())
env = dict(os.environ, PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
commands = []
for original in json.loads((base/'build.json').read_text())['commands']:
    command = [arg.replace(str(old_work),str(work)) for arg in original]
    subprocess.run(command,env=env,check=True)
    commands.append(command)
for name, expected in before.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
outputs = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
manifest.write_text(json.dumps(dict(build_passed=True,gpu_validated=False,
    accepted_benchmark=False,commands=commands,sha256={**before,**outputs}),indent=2))
print('Isolated metadata C17/C23 builds and four executables linked; GPU validation pending')
