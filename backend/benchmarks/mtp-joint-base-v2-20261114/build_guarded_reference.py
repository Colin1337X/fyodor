"""Compile the external reference only; never launch GPU work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'guarded-reference-build.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite build evidence')
upstream = root / '.tools/reference-llama-b10809'
libraries = root / '.tools/llama-b10809-cuda'
work = root / 'build-cuda/guarded-reference-20261114'
work.mkdir(exist_ok=True)
original = out.parent / 'aligned-prefix-runtime-20261114/llama_generation.c'
source = original.read_text()
assert source.count('        double start=seconds();') == 1
assert source.count('        double end=seconds();') == 1
source = source.replace('        double start=seconds();', '        Sleep(2000); /* Isolate the decode interval from prefill sampling. */\n        double start=seconds();')
source = source.replace('        double end=seconds();', '        double end=seconds();\n        Sleep(2000); /* Keep post-decode work outside its PDH brackets. */')
(out / 'llama_guarded.c').write_text(source)
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = os.environ.copy()
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
inputs = [out / 'llama_guarded.c', Path(__file__), original, upstream / 'include/llama.h',
          upstream / 'ggml/include/ggml.h', upstream / 'ggml/include/ggml-backend.h',
          upstream / 'LICENSE'] + [libraries / name for name in
          ('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cuda.dll', 'cublas64_13.dll', 'cublasLt64_13.dll')]
commands = []
for standard in ('c17', 'c23'):
    exe = work / f'llama-generation-{standard}.exe'
    command = [str(compiler), f'-std={standard}', '-O3', '-Wall', '-Wextra', '-Wpedantic',
               '-Wconversion', '-Wshadow', '-Werror', '-I', str(upstream / 'include'),
               '-I', str(upstream / 'ggml/include'), str(inputs[0]),
               *[str(libraries / name) for name in ('llama.dll', 'ggml.dll', 'ggml-base.dll')],
               '-o', str(exe)]
    subprocess.run(command, env=env, check=True)
    commands.append(command)
    inputs.append(exe)
metadata = dict(quiet_gap_seconds=2, accepted_benchmark=False, executed_generation=False, commands=commands,
                upstream_revision=subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                limitations=['Two-second gaps may change clocks; attribution diagnostic only, not comparable performance.',
                             'Needs the recorded external DLL directory in PATH for execution.',
                             'Must run after sanitizer completion under continuous GPU observation.',
                             'Generation timing cannot replace fixed-forward benchmark scores.'])
with destination.open('x') as result:
    json.dump(metadata, result, indent=2)
print('Reference C17/C23 compiled without GPU execution')
