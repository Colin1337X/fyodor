"""Link isolated generation and lifecycle checks without executing GPU work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
work = root / 'build-cuda/interleave-pair-20261114'
manifest = out / 'executables.json'
if manifest.exists() or any(work.glob('*.exe')):
    raise SystemExit('Refusing to overwrite executables')
proofs = [out/'build.json', out.parent/'integer-launch-20261114/generation-build.json']
inputs = set(proofs + [Path(__file__).resolve()])
for path in proofs:
    proof = json.loads(path.read_text())
    if not (proof.get('passed') is True or proof.get('compile_passed') is True):
        raise SystemExit(f'Failed prerequisite: {path}')
    for name, expected in proof['sha256'].items():
        file = root / name
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Changed prerequisite: {name}')
        inputs.add(file)
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = os.environ.copy()
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
library = root / 'build-cuda/libnya-engine.a'
generation = out.parent / 'mtp-joint-base-v2-20261114/generation.c'
lifecycle = out.parent / 'integer-launch-20261114/verify-prefix257-ring1-count4.c'
inputs.update([library, generation, lifecycle])
before = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
commands = []
outputs = []
for standard in ('c17', 'c23'):
    cuda = work / f'cuda-{standard}.o'
    core = root / f'build-cuda/joint-base-generation-v2-20261114/core-{standard}.o'
    for name, source, objects in [('generation',generation,[core,cuda]),
                                  ('lifecycle',lifecycle,[cuda])]:
        exe = work / f'{name}-{standard}.exe'
        command = [str(compiler),f'-std={standard}','-O3','-Wall','-Wextra',
            '-Wpedantic','-Wconversion','-Wshadow','-Werror',
            '-I','backend/include','-I','backend/core',str(source),
            *map(str,objects),str(library),'-lbcrypt','C:/Windows/System32/vulkan-1.dll',
            '-o',str(exe)]
        subprocess.run(command,env=env,check=True)
        commands.append(command)
        outputs.append(exe)
for name, expected in before.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Input changed during link: {name}')
manifest.write_text(json.dumps(dict(link_passed=True, gpu_validated=False,
    accepted_benchmark=False, commands=commands,
    sha256={**before,**{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}),indent=2))
print('Four isolated C17/C23 test executables linked; no GPU execution')
