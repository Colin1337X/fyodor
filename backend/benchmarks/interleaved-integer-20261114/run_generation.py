"""Launch one opt-in Q4 generation cell without replacing prior evidence."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
if len(sys.argv) != 4 or sys.argv[1] not in ('0', '1') or sys.argv[2] not in ('short', 'long'):
    raise SystemExit('Usage: run_generation.py INTEGER_0_OR_1 short|long UNIQUE_LABEL')
setting, context, label = sys.argv[1:]
if not label.isascii() or not label.isalnum():
    raise SystemExit('Label must be ASCII alphanumeric')
folder = out / f'generation-{context}-integer{setting}-{label}'
if folder.exists():
    raise SystemExit('Refusing to overwrite generation evidence')
# Correctness is a prerequisite even for diagnostic timing.
for name in ('verify-prefix257-ring1-count4.json', 'verify-prefix8192-ring1-count4.json'):
    path = out / name
    if not path.exists():
        raise SystemExit(f'Wait for completed model validation: {name}')
    proof = json.loads(path.read_text())
    assert proof['exit_code'] == 0 and proof['integer_selected'] is True
    assert proof['integer_calls'] and min(proof['integer_calls']) > 0
    for source, expected in proof['sha256'].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == expected, source
sanitizer = json.loads((out / 'memcheck-run.json').read_text())
assert sanitizer['passed'] is True and sanitizer['exit_code'] == 0
for name, expected in sanitizer['evidence_sha256'].items():
    assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected, name
build = json.loads((out / 'generation-build.json').read_text())
for name, expected in build['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
# Require the conservative idle gate before launching; no competing GPU workload.
folder.mkdir()
preflight = folder / 'preflight'; preflight.mkdir()
spec = importlib.util.spec_from_file_location('gate', out.parent / 'gemma4-qkv-20261029/environment.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
gate.configure(preflight)
gate.idle()
env = {key:value for key,value in os.environ.items() if not key.startswith('NYA_')}
env.update(PYTHONDONTWRITEBYTECODE='1', NYA_BENCH_GPU_MONITOR='1', NYA_CUDA_PROFILE='0',
           NYA_CUDA_ALIGNED_PREFIX='1', NYA_CUDA_WIDE_BATCH_SPLIT='1', NYA_CUDA_WIDE_BATCH='1',
           NYA_CUDA_SMALL_BATCH_REUSE='1', NYA_CUDA_INTERLEAVED_REUSE='1', NYA_CUDA_INTERLEAVED_TWO='1',
           NYA_CUDA_INTERLEAVED_INTEGER=setting, NYA_CUDA_BATCH_LOGITS='1', NYA_CUDA_ASSISTED_FFN='1',
           NYA_CUDA_MTP_WIDE_VALUE='1', NYA_SPECULATIVE_EARLY_REJECT='1')
command = [sys.executable, str(out.parent / 'mtp-cost-20261113/run.py'), str(folder),
           'build-cuda/integer-generation-20261114/generation.exe', context]
subprocess.run(command, env=env, check=True)
trace = (folder / f'{context}.stderr').read_text()
if setting == '1' and 'CUDA Q4 integer batch-four calls:' not in trace:
    raise SystemExit('Candidate generation did not execute the integer kernel')
if setting == '0' and 'CUDA Q4 integer batch-four calls:' in trace:
    raise SystemExit('Baseline unexpectedly executed the integer kernel')
