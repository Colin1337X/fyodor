"""Inspect the verified offline cubin; this does not execute GPU code."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'sass-audit-hoisted.json'
if destination.exists():
    raise SystemExit('Refusing disassembly evidence overwrite')
proof_path = out / 'assembly-hoisted.json'
proof = json.loads(proof_path.read_text())
if proof.get('assembled') is not True:
    raise ValueError('Assembly did not pass')
inputs = {root / name for name in proof['sha256']}
for name, expected in proof['sha256'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError('Changed assembly input: ' + name)
cubin = root / 'build-cuda/q4-tf32-three-hoisted-20261114/module.cubin'
dump = root / '.tools/cuda-sdk/cuda_cuobjdump-windows-x86_64-13.1.115-archive/bin/cuobjdump.exe'
disasm = root / '.tools/cuda-sdk/cuda_nvdisasm-windows-x86_64-13.1.115-archive/bin/nvdisasm.exe'
command = [str(dump), '--function', 'nya_q4_tf32_three_hoisted', '--dump-sass', str(cubin)]
completed = subprocess.run(command, capture_output=True, text=True,
    env=dict(os.environ, PATH=str(disasm.parent) + os.pathsep + os.environ['PATH']))
sass, stderr = out / 'kernel-hoisted.sass', out / 'disassembly-hoisted.stderr'
sass.write_text(completed.stdout); stderr.write_text(completed.stderr)
if completed.returncode:
    raise RuntimeError('Offline disassembly failed; see retained logs')
match = re.search(r'Function : nya_q4_tf32_three_hoisted\s+(.*?)(?=Function :|\Z)', completed.stdout, re.S)
if not match:
    raise ValueError('Candidate function missing')
counts = Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]*)', match[1]))
tensor = {name: count for name, count in counts.items() if name.startswith('HMMA')}
if not tensor:
    raise ValueError('No tensor-core instruction sites found')
inputs.update((Path(__file__).resolve(), proof_path, cubin, dump, disasm, sass, stderr))
result = dict(passed=True, gpu_execution=False, accepted_benchmark=False, command=command,
    tensor_instruction_sites=tensor, static_instruction_counts=dict(sorted(counts.items())),
    note='Offline SM120 static instruction sites, not driver JIT code, dynamic execution, correctness, occupancy or speed.',
    sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
destination.write_text(json.dumps(result, indent=2))
print('Offline tensor instruction sites:', tensor)
