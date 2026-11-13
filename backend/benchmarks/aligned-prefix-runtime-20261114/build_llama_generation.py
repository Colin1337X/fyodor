"""Compile the external reference only; never launch GPU work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'llama-generation-build.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite build evidence')
upstream = root / '.tools/reference-llama-b10809'
libraries = root / '.tools/llama-b10809-cuda'
work = root / 'build-cuda/aligned-generation-20261114'
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = os.environ.copy()
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
inputs = [out / 'llama_generation.c', Path(__file__), upstream / 'include/llama.h',
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
metadata = dict(accepted_benchmark=False, executed_generation=False, commands=commands,
                upstream_revision=subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                limitations=['Compilation only; prompt-token equality and generated-output parity remain unverified.',
                             'Needs the recorded external DLL directory in PATH for execution.',
                             'Must run after sanitizer completion under continuous GPU observation.',
                             'Generation timing cannot replace fixed-forward benchmark scores.'])
with destination.open('x') as result:
    json.dump(metadata, result, indent=2)
print('Reference C17/C23 compiled without GPU execution')
