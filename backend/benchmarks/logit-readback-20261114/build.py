"""Build isolated contiguous-logit readback on the FFN reuse candidate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
base = out.parent / 'interleave-pair-20261114'
old_work = root / 'build-cuda/interleave-pair-20261114'
work = root / 'build-cuda/logit-readback-20261114'
if work.exists() or (out / 'build.json').exists():
    raise SystemExit('Refusing to overwrite build')
proofs = [json.loads((base / name).read_text()) for name in ('build.json','executables.json')]
assert proofs[0]['compile_passed'] and proofs[1]['link_passed']
inputs = {Path(__file__).resolve(), out / 'readback.inc', out / 'test_readback.c',
          base / 'build.json', base / 'executables.json'}
for proof in proofs:
    for name, expected in proof['sha256'].items():
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        inputs.add(path)
before = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
source = (old_work / 'resident.inc').read_text()
marker = 'static int plan_read_logits_row(nya_cuda_plan *p,float *output,size_t row)'
assert source.count(marker) == 1
source = source.replace(marker, (out / 'readback.inc').read_text() + '\n' + marker)
marker = '            for(size_t i=0;i<count;++i) RUN(plan_read_logits_row(p,output+i*m->tokenizer.vocabulary_size,i));'
assert source.count(marker) == 1
source = source.replace(marker, '            RUN(plan_read_logits_bank(p,output,count));')
work.mkdir()
(work / 'resident.inc').write_text(source)
for name in ('cuda.c','cuda_source.h'):
    (work / name).write_bytes((old_work / name).read_bytes())
lifecycle_path = out.parent / 'integer-launch-20261114/verify-prefix257-ring1-count4.c'
lifecycle = lifecycle_path.read_text()
assert 'size_t count=4,' in lifecycle
marker = '            CHECK(after.downloads-before.downloads==count);'
assert lifecycle.count(marker) == 1
lifecycle = lifecycle.replace(marker, '''            CHECK(getenv("NYA_CUDA_BATCH_LOGITS") && !strcmp(getenv("NYA_CUDA_BATCH_LOGITS"),"1"));
            CHECK(after.prefill_batch>=count);
            CHECK(after.downloads-before.downloads==1);''')
(work / 'lifecycle.c').write_text(lifecycle)
env = os.environ.copy()
env['PATH'] = str(root / '.tools/w64devkit/bin') + os.pathsep + env['PATH']
commands = []
for index, standard in enumerate(('c17','c23')):
    command = [str(work / 'cuda.c') if arg == str(old_work / 'cuda.c') else
               str(work / f'cuda-{standard}.o') if arg == str(old_work / f'cuda-{standard}.o') else arg
               for arg in proofs[0]['commands'][index]]
    subprocess.run(command,env=env,check=True)
    commands.append(command)
    for kind in range(2):
        name = 'generation' if kind == 0 else 'lifecycle'
        substitutions = {str(old_work / f'cuda-{standard}.o'): str(work / f'cuda-{standard}.o'),
            str(old_work / f'{name}-{standard}.exe'): str(work / f'{name}-{standard}.exe'),
            str(lifecycle_path): str(work / 'lifecycle.c')}
        command = [substitutions.get(arg,arg) for arg in proofs[1]['commands'][2*index+kind]]
        subprocess.run(command,env=env,check=True)
        commands.append(command)
for name, expected in before.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
outputs = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
(out / 'build.json').write_text(json.dumps(dict(build_passed=True,gpu_validated=False,
    accepted_benchmark=False,commands=commands,sha256={**before,**outputs}),indent=2))
print('C17/C23 host builds and four executables linked; GPU validation pending')
