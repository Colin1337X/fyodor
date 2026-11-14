"""Validate isolated contiguous logit readback; this is not a throughput run."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
if len(sys.argv) != 3 or sys.argv[1] not in ('c17', 'c23') or sys.argv[2] not in ('257', '8192'):
    raise SystemExit('Usage: run_lifecycle.py c17|c23 257|8192')
standard, prefix = sys.argv[1:]
folder = out / f'lifecycle-{standard}-{prefix}'
if folder.exists():
    raise SystemExit('Refusing to overwrite evidence')
proof_path = out / 'build.json'
proof = json.loads(proof_path.read_text())
assert proof['build_passed'] is True
inputs = {root / name for name in proof['sha256']}
inputs.update((proof_path, Path(__file__).resolve()))
for name, expected in proof['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(out.parent / 'mtp-joint-base-v2-20261114'))
from model_identity import verify_gemma
identity = verify_gemma()
sys.path.insert(0, str(out.parent / 'gemma4-qkv-20261029'))
import environment
initial = environment.snapshot()
if any(not environment.desktop_identity_allowed(p) for p in initial['processes']):
    raise SystemExit('Another GPU workload remains')
settings_path = out.parent / 'integer-launch-20261114/verify-prefix257-ring1-count4.json'
settings = json.loads(settings_path.read_text())['environment']
inputs.add(settings_path)
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
env.update(settings)
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['PATH'] = str(root / '.tools/w64devkit/bin') + os.pathsep + env['PATH']
exe = root / f'build-cuda/integer-metadata-20261114/lifecycle-{standard}.exe'
command = [str(exe), 'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf', prefix]
folder.mkdir()
with (folder / 'stdout').open('x') as stdout, (folder / 'stderr').open('x') as stderr:
    child = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
trace = (folder / 'stderr').read_text()
calls = [int(x) for x in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)', trace)]
changed = [name for name, expected in hashes.items()
           if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
selected = "CUDA native metadata Q4: selected" in trace
passed = selected and child.returncode == 0 and bool(calls) and min(calls) > 0 and not changed
outputs = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in folder.iterdir() if p.is_file()}
(folder / 'result.json').write_text(json.dumps(dict(passed=passed,
    accepted_benchmark=False, single_download_assertions_passed=passed, command=command,
    environment=settings, initial=initial, model_identity=identity,
    exit_code=child.returncode, metadata_selected=selected, integer_calls=calls, changed_inputs=changed,
    sha256={**hashes, **outputs},
    note='Original lifecycle/numerical guards with exactly one download asserted per four-row verification; no throughput acceptance.'), indent=2))
print('Lifecycle', standard, prefix, 'passed' if passed else 'FAILED', flush=True)
raise SystemExit(0 if passed else 1)
