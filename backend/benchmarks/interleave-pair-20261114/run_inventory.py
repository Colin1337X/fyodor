"""Compare real-model baseline/reuse dispatch counts; no speed acceptance."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
folder = out / 'inventory-257'
if folder.exists():
    raise SystemExit('Refusing to overwrite evidence')
proof_path = out / 'inventory-build.json'
proof = json.loads(proof_path.read_text())
assert proof['build_passed'] is True
hashes = dict(proof['sha256'])
settings_path = out / 'lifecycle-c17-257/result.json'
settings = json.loads(settings_path.read_text())['environment']
for path in (proof_path, settings_path, Path(__file__).resolve()):
    hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(out.parent / 'mtp-joint-base-v2-20261114'))
from model_identity import verify_gemma
identity = verify_gemma()
sys.path.insert(0, str(out.parent / 'gemma4-qkv-20261029'))
import environment
initial = environment.snapshot()
if any(not environment.desktop_identity_allowed(p) for p in initial['processes']):
    raise SystemExit('Another GPU workload remains')
env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
env.update(settings)
env['PATH'] = str(root / '.tools/w64devkit/bin') + os.pathsep + env['PATH']
folder.mkdir()
runs = {}
for variant in ('baseline', 'reuse'):
    exe = root / f'build-cuda/interleave-pair-inventory-20261114/{variant}/lifecycle.exe'
    command = [str(exe), 'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf', '257']
    with (folder / f'{variant}.stdout').open('x') as stdout, (folder / f'{variant}.stderr').open('x') as stderr:
        child = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    trace = (folder / f'{variant}.stderr').read_text()
    counts = [dict(zip(('converted', 'reused', 'projections'), map(int, values)))
              for values in re.findall(r'FFN input inventory: converted=(\d+) reused=(\d+) projections=(\d+)', trace)]
    runs[variant] = dict(exit_code=child.returncode, command=command, counts=counts)
    if child.returncode:
        break
changed = [name for name, expected in hashes.items()
           if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
passed = len(runs) == 2 and all(r['exit_code'] == 0 for r in runs.values()) and not changed
if passed:
    baseline, reuse = runs['baseline']['counts'], runs['reuse']['counts']
    passed = bool(baseline) and len(baseline) == len(reuse)
    passed = passed and all(a['reused'] == 0 and a['converted'] == a['projections'] and
        a['projections'] == b['projections'] and b['converted'] + b['reused'] == b['projections'] and
        b['reused'] > 0 and b['converted'] < a['converted'] for a, b in zip(baseline, reuse))
outputs = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in folder.iterdir() if p.is_file()}
(folder / 'result.json').write_text(json.dumps(dict(passed=passed, accepted_benchmark=False,
    initial=initial, model_identity=identity, environment=settings, runs=runs,
    changed_inputs=changed, sha256={**hashes, **outputs},
    note='Host dispatch counts include graph construction; they are not GPU replay counts or throughput.'), indent=2))
print('Dispatch inventory', 'passed' if passed else 'FAILED', flush=True)
raise SystemExit(0 if passed else 1)
