"""Monitor the validated joint-base binary; coverage alone is not acceptance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
if len(sys.argv) != 3 or sys.argv[1] not in ('short', 'long'):
    raise SystemExit('Usage: run_monitored.py short|long UNIQUE_LABEL')
context, label = sys.argv[1:]
if not label.isascii() or not label.isalnum():
    raise SystemExit('Label must be ASCII alphanumeric')
folder = out / f'monitored-{context}-{label}'
if folder.exists():
    raise SystemExit('Refusing to overwrite evidence')
proofs = [out / name for name in ('generation-build.json', 'fixture-build.json',
                                 'edge-fixture-build.json')]
proofs += [out / f'fixture-{mode}/result.json' for mode in ('cpu', 'ring', 'linear', 'failure')]
proofs += [out / f'edge-fixture-{mode}/result.json' for mode in ('cpu', 'ring', 'linear', 'failure')]
proofs += [out / f'edge-memcheck-{mode}/result.json' for mode in ('ring', 'linear')]
proofs += [out / f'{mode}-correctness/correctness.json' for mode in ('short', 'long')]
for path in proofs:
    proof = json.loads(path.read_text())
    if path.name in ('result.json', 'correctness.json') and proof.get('passed') is not True:
        raise SystemExit(f'Prerequisite failed: {path}')
    for name, expected in proof['sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Prerequisite input changed: {name}')
spec = importlib.util.spec_from_file_location('gate', out.parent / 'gemma4-qkv-20261029/environment.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
folder.mkdir()
preflight = folder / 'preflight'
preflight.mkdir()
gate.configure(preflight)
gate.idle()
env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
env.update(PYTHONDONTWRITEBYTECODE='1', NYA_CUDA_PROFILE='0', NYA_BENCH_GPU_MONITOR='1',
    NYA_BENCH_MODE_SET='joint-base', NYA_CUDA_ALIGNED_PREFIX='1', NYA_CUDA_WIDE_BATCH_SPLIT='1',
    NYA_CUDA_WIDE_BATCH='1', NYA_CUDA_SMALL_BATCH_REUSE='1', NYA_CUDA_INTERLEAVED_REUSE='1',
    NYA_CUDA_INTERLEAVED_TWO='1', NYA_CUDA_INTERLEAVED_INTEGER='0', NYA_CUDA_BATCH_LOGITS='1',
    NYA_CUDA_ASSISTED_FFN='1', NYA_CUDA_MTP_WIDE_VALUE='1', NYA_SPECULATIVE_EARLY_REJECT='1')
runner = out.parent / 'mtp-cost-20261113/run.py'
command = [sys.executable, str(runner), str(folder),
           'build-cuda/joint-base-generation-v2-20261114/generation.exe', context]
completed = subprocess.run(command, env=env)
trace = (folder / f'{context}.stderr').read_text()
selected = trace.count('MTP joint base: enabled; proposal window=3')
evidence = proofs + [Path(__file__), runner]
evidence += [p for p in folder.iterdir() if p.is_file()]
result = dict(accepted_benchmark=False, runner_exit=completed.returncode, joint_rounds=selected,
    command=command, sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in evidence})
(folder / 'result.json').write_text(json.dumps(result, indent=2))
if selected != 3:
    raise SystemExit('Joint mode was not selected in exactly three rounds')
raise SystemExit(completed.returncode)
