"""Record real planner admission with explicit feature/budget controls."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
if len(sys.argv) != 4:
    raise SystemExit('Usage: run_admission.py CAPACITY BUDGET_MIB MODE')
capacity, budget = map(int, sys.argv[1:3])
mode = sys.argv[3]
if not 1 <= capacity <= 262144 or not 0 <= budget <= 16384:
    raise SystemExit('Invalid capacity or budget (zero means default budget)')
overrides = {'on': {}, 'off': {'NYA_CUDA_ALIGNED_PREFIX': '0'},
             'noReuse': {'NYA_CUDA_SMALL_BATCH_REUSE': '0'},
             'noSplit': {'NYA_CUDA_SPLIT_ATTENTION': '0'},
             'portable': {'NYA_CUDA_FDPP_PORTABLE': '1'}}
if mode not in overrides:
    raise SystemExit('Unknown mode')
tag = f'admission-cap{capacity}-mib{budget}-{mode}'
paths = {ext: out / f'{tag}.{ext}' for ext in ('json', 'log', 'stderr.log')}
exe = root / f'build-cuda/{tag}.exe'
if any(path.exists() for path in (*paths.values(), exe)):
    raise SystemExit('Refusing to overwrite evidence')
source, library = out / 'admission.c', root / 'build-cuda/libnya-engine.a'
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
baseline = json.loads((out.parent / 'fdpp-batch-20261114/aligned-prefix-ctx8192.json').read_text())
settings = {k: v for k, v in baseline['environment'].items() if not k.startswith('NYA_TEST_')}
settings.update(NYA_CUDA_ALIGNED_PREFIX='1', NYA_CUDA_RING_KV='1', NYA_MTP_TRACE='1',
                NYA_CUDA_DEBUG='1', NYA_CUDA_INTERLEAVED_REUSE='1', NYA_CUDA_INTERLEAVED_TWO='1')
settings.update(overrides[mode])
if budget:
    settings['NYA_CUDA_MEMORY_MIB'] = str(budget)
env.update(settings)
build = [str(compiler), '-std=c17', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-Wconversion',
         '-Wshadow', '-Werror', '-I', 'backend/include', '-I', 'backend/core', str(source),
         str(library), '-lbcrypt', 'C:/Windows/System32/vulkan-1.dll', '-o', str(exe)]
subprocess.run(build, env=env, check=True)
command = [str(exe), 'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf', str(capacity)]
inputs = [source, library, exe, Path(__file__), root / 'CUDA/resident.inc']
hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
with paths['log'].open('x') as stdout, paths['stderr.log'].open('x') as stderr:
    result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
trace = paths['stderr.log'].read_text()
rows, parse_error = [], None
try:
    rows = [json.loads(line) for line in paths['log'].read_text().splitlines() if line.startswith('{')]
except ValueError as error:
    parse_error = str(error)
partial_queries = [int(value) for value in re.findall(r'CUDA aligned prefix: partial_queries=(\d+)', trace)]
meta = dict(accepted_benchmark=False, command=command, build=build, environment=settings,
            sha256=hashes, exit_code=result.returncode, rows=rows, parse_error=parse_error,
            partial_queries=partial_queries)
with paths['json'].open('x') as output:
    json.dump(meta, output, indent=2)
if result.returncode or parse_error or len(rows) != 1:
    raise SystemExit(result.returncode or 'Incomplete admission evidence')
expected = mode == 'on' and capacity > 256 and rows[0]['resident']
assert bool(partial_queries) == expected, 'Unexpected aligned admission'
if expected:
    assert partial_queries == [rows[0]['batch']], 'Partial storage did not follow admitted batch'
print(tag, json.dumps(rows[0]), flush=True)
