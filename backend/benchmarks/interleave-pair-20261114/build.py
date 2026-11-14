"""Build an isolated host-only candidate; never modifies the running GPU build."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
base = out.parent / 'integer-launch-20261114'
work = root / 'build-cuda/interleave-pair-20261114'
if work.exists() or (out / 'build.json').exists():
    raise SystemExit('Refusing to overwrite experiment')
proof_path = base / 'model-build.json'
proof = json.loads(proof_path.read_text())
assert proof['passed'] is True
for name, digest in proof['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
original = root / 'CUDA/resident.inc'
source = original.read_text()
old = '    size_t interleaved_columns;'
assert source.count(old) == 1
source = source.replace(old, old + '\n    int reuse_interleaved_input; /* Only the immediately adjacent gate/up pair. */')
old = '            if(plan_launch(p,convert,(columns+255)/256,convert_args))return -1;'
assert source.count(old) == 1
source = source.replace(old, '            if(!p->reuse_interleaved_input && plan_launch(p,convert,(columns+255)/256,convert_args))return -1;')
old = '            RUN(plan_matvec(p, p->up, w->feed_forward_up, p->norm, h, d));'
assert source.count(old) == 1
source = source.replace(old, '''            /* The gate projection leaves norm and its packed copy untouched.
               Exclude vendor and aligned-prefix paths: those do not produce
               this scratch layout. Never carry reuse beyond the adjacent call. */
            p->reuse_interleaved_input = (p->batch == 2 || p->batch == 4) &&
                plan_interleaved_kernel(p,w->feed_forward_gate->type,d) &&
                plan_interleaved_kernel(p,w->feed_forward_up->type,d) &&
                !(p->blas_storage && nya_blas_bytes(p->cuda,h,d,p->batch));
            int up_result = plan_matvec(p, p->up, w->feed_forward_up, p->norm, h, d);
            p->reuse_interleaved_input = 0;
            RUN(up_result);''')
# Preserve the relative backend include from the generated build directory.
source = source.replace('#include "../backend/core/llm_internal.h"', '#include "llm_internal.h"')
work.mkdir()
(work / 'resident.inc').write_text(source)
header = root / 'build-cuda/integer-launch-20261114/cuda_source.h'
(work / 'cuda_source.h').write_bytes(header.read_bytes())
host = root / 'build-cuda/integer-launch-20261114/cuda.c'
(work / 'cuda.c').write_bytes(host.read_bytes())
flags = (root / 'build-cuda/CMakeFiles/nya-engine.dir/flags.make').read_text()
defines = re.search(r'^C_DEFINES = (.*)$',flags,re.M)[1].split()
cflags = re.search(r'^C_FLAGS = (.*)$',flags,re.M)[1].split()
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = os.environ.copy()
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
commands = []
for standard in ('c17', 'c23'):
    command = [str(compiler), *defines,
        *[f'-std={standard}' if f.startswith('-std=') else f for f in cflags],
        '-I',str(root/'CUDA'),'-I',str(root/'backend/core'),
        '@'+str(root/'build-cuda/CMakeFiles/nya-engine.dir/includes_C.rsp'),
        '-c',str(work/'cuda.c'),'-o',str(work/f'cuda-{standard}.o')]
    subprocess.run(command,env=env,check=True)
    commands.append(command)
# Compilation must leave the validated baseline inputs unchanged.
for name, digest in proof['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
paths = [Path(__file__).resolve(), original, proof_path, header, host]
paths += list(work.iterdir())
(out/'build.json').write_text(json.dumps(dict(compile_passed=True,
    gpu_validated=False, accepted_benchmark=False, commands=commands,
    sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
print('C17/C23 compile passed; no GPU execution, correctness or speed claim')
