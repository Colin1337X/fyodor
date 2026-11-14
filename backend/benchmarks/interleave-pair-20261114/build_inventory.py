"""Instrument baseline/candidate host dispatch, without altering validated builds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
work = root / 'build-cuda/interleave-pair-inventory-20261114'
manifest = out / 'inventory-build.json'
if work.exists() or manifest.exists():
    raise SystemExit('Refusing to overwrite inventory build')
proof_paths = [out / 'build.json', out / 'executables.json']
inputs = set(proof_paths + [Path(__file__).resolve()])
proofs = [json.loads(p.read_text()) for p in proof_paths]
assert proofs[0]['compile_passed'] and proofs[1]['link_passed']
for proof in proofs:
    for name, expected in proof['sha256'].items():
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        inputs.add(path)
before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
base_work = root / 'build-cuda/interleave-pair-20261114'
env = os.environ.copy()
env['PATH'] = str(root / '.tools/w64devkit/bin') + os.pathsep + env['PATH']
commands = []
work.mkdir()
for variant in ('baseline', 'reuse'):
    folder = work / variant
    folder.mkdir()
    source = ((root / 'CUDA/resident.inc') if variant == 'baseline' else (base_work / 'resident.inc')).read_text()
    source = source.replace('#include "../backend/core/llm_internal.h"', '#include "llm_internal.h"')
    marker = '    size_t interleaved_columns;'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n    unsigned long long inventory_converted, inventory_reused;')
    marker = '    const char *mtp_trace=getenv("NYA_MTP_TRACE");'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
    if(mtp_trace && !strcmp(mtp_trace,"1") && (p->inventory_converted || p->inventory_reused))
        fprintf(stderr,"FFN input inventory: converted=%llu reused=%llu projections=%llu\\n",
            p->inventory_converted,p->inventory_reused,p->interleaved_calls+p->interleaved_two_calls);
''')
    marker = ('            if(plan_launch(p,convert,(columns+255)/256,convert_args))return -1;'
              if variant == 'baseline' else
              '            if(!p->reuse_interleaved_input && plan_launch(p,convert,(columns+255)/256,convert_args))return -1;')
    assert source.count(marker) == 1
    replacement = '''            if(plan_launch(p,convert,(columns+255)/256,convert_args))return -1;
            ++p->inventory_converted;'''
    if variant == 'reuse':
        replacement = '''            if(p->reuse_interleaved_input) ++p->inventory_reused;
            else {
                if(plan_launch(p,convert,(columns+255)/256,convert_args))return -1;
                ++p->inventory_converted;
            }'''
    source = source.replace(marker, replacement)
    (folder / 'resident.inc').write_text(source)
    for name in ('cuda.c', 'cuda_source.h'):
        (folder / name).write_bytes((base_work / name).read_bytes())
    command = [str(folder / 'cuda.c') if arg == str(base_work / 'cuda.c') else
               str(folder / 'cuda.o') if arg == str(base_work / 'cuda-c17.o') else arg
               for arg in proofs[0]['commands'][0]]
    subprocess.run(command, env=env, check=True)
    commands.append(command)
    original = proofs[1]['commands'][1]
    command = [str(folder / 'cuda.o') if arg == str(base_work / 'cuda-c17.o') else
               str(folder / 'lifecycle.exe') if arg == str(base_work / 'lifecycle-c17.exe') else arg
               for arg in original]
    subprocess.run(command, env=env, check=True)
    commands.append(command)
for name, expected in before.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
outputs = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in work.rglob('*') if p.is_file()}
manifest.write_text(json.dumps(dict(build_passed=True, gpu_validated=False,
    accepted_benchmark=False, commands=commands, sha256={**before, **outputs}), indent=2))
print('Baseline/reuse inventory builds linked; no GPU execution')
