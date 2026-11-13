"""Launch the monitored generation diagnostic after integrated sanitizer proof."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
contexts = sys.argv[1:] or ['short', 'long']
if len(set(contexts)) != len(contexts) or any(context not in ('short', 'long') for context in contexts):
    raise SystemExit('Choose distinct short/long contexts')
audit_path = out / 'verify-prefix513-ring1-count4-memcheck-audit.json'
if not audit_path.exists():
    raise SystemExit('Integrated sanitizer audit must complete before generation measurement')
audit = json.loads(audit_path.read_text())
assert audit['completed_child_exit_code'] == 0 and audit['numerical_lifecycle_passed'] is True
assert audit['sanitizer_zero_errors'] is True and max(audit['maximum_scaled_errors']) <= .0005
build = json.loads((out / 'generation-build.json').read_text())
for name, expected in build['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Build input changed: {name}'
for name, expected in audit['verified_launch_inputs'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Sanitizer input changed: {name}'
result = out / 'monitored-generation'
if result.exists():
    raise SystemExit('Refusing to overwrite generation evidence')
env = {key: value for key, value in os.environ.items() if not key.startswith('NYA_')}
env.update(PYTHONDONTWRITEBYTECODE='1', NYA_BENCH_GPU_MONITOR='1', NYA_CUDA_PROFILE='0',
           NYA_CUDA_ALIGNED_PREFIX='1', NYA_CUDA_WIDE_BATCH_SPLIT='1', NYA_CUDA_WIDE_BATCH='1',
           NYA_CUDA_SMALL_BATCH_REUSE='1', NYA_CUDA_INTERLEAVED_REUSE='1', NYA_CUDA_INTERLEAVED_TWO='1',
           NYA_CUDA_BATCH_LOGITS='1', NYA_CUDA_ASSISTED_FFN='1', NYA_CUDA_MTP_WIDE_VALUE='1',
           NYA_SPECULATIVE_EARLY_REJECT='1')
command = [sys.executable, 'backend/benchmarks/mtp-cost-20261113/run.py',
           str(result), 'build-cuda/aligned-generation-20261114/generation.exe', *contexts]
subprocess.run(command, env=env, check=True)
