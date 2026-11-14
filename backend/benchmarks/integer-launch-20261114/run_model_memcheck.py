"""Run the isolated Q4 launch-bound joint-base binary under Compute Sanitizer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def validate(code, stdout, stderr, context):
    if type(code) is not int or code != 0:
        raise ValueError('Sanitizer child failed')
    if re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$', stdout + '\n' + stderr, re.M) != ['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    modes = ('ordinary', 'mtp-serial2', 'mtp-batch2', 'mtp-joint3')
    order = [(r, mode) for r in range(3) for mode in (reversed(modes) if r == 1 else modes)]
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith('{"round":')]
    if [(row['round'], row['mode']) for row in rows] != order:
        raise ValueError('Missing or reordered generation cases')
    prompt = {'short': 25, 'long': 8205}[context]
    if any(row.get('parity') is not True or row.get('generated') != 32 or row.get('prompt') != prompt for row in rows):
        raise ValueError('Real-model output parity or workload failed')
    if stderr.count('MTP joint base: enabled; proposal window=3') != 3:
        raise ValueError('Joint mode was not selected in all three rounds')
    calls = [int(v) for v in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)', stderr)]
    if len(calls) != 3 or min(calls) <= 0:
        raise ValueError('Integer kernel was not executed in all three joint rounds')


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    if len(sys.argv) != 2 or sys.argv[1] not in ('short', 'long'):
        raise SystemExit('Choose short or long')
    context = sys.argv[1]
    folder = out / f'model-memcheck-{context}'
    if folder.exists():
        raise SystemExit('Refusing to overwrite sanitizer evidence')
    proofs = [out / 'generation-build.json', out / 'model-build.json',
              out / f'generation-{context}-c17/result.json',
              out / 'verify-prefix257-ring1-count4.json',
              out / 'verify-prefix8192-ring1-count4.json']
    for path in proofs:
        proof = json.loads(path.read_text())
        if 'passed' in proof and proof['passed'] is not True:
            raise SystemExit(f'Failed prerequisite: {path}')
        if 'integer_selected' in proof and (proof['exit_code'] != 0 or not proof['integer_selected'] or not proof['integer_calls'] or min(proof['integer_calls']) <= 0):
            raise SystemExit(f'Failed lifecycle prerequisite: {path}')
        for name, expected in proof['sha256'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise SystemExit(f'Changed prerequisite: {name}')
    identity_path = out.parent / 'mtp-joint-base-v2-20261114/model_identity.py'
    sys.path.insert(0, str(identity_path.parent))
    from model_identity import verify_gemma
    identity = verify_gemma()
    spec = importlib.util.spec_from_file_location('gate', out.parent / 'gemma4-qkv-20261029/environment.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
        raise SystemExit('Another GPU workload remains')
    settings_path = out / f'generation-{context}-c17/result.json'
    settings = json.loads(settings_path.read_text())['environment']
    env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
    env.update(settings)
    sanitizer = root / '.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    binary = root / 'build-cuda/integer-launch-20261114/generation-c17.exe'
    command = [str(sanitizer), '--tool', 'memcheck', '--error-exitcode', '99', str(binary), context]
    inputs = proofs + [Path(__file__), settings_path, identity_path, sanitizer, binary]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    folder.mkdir()
    result = dict(accepted_benchmark=False, passed=False, command=command, environment=settings,
                  model_identity=identity, sanitizer_sha256=hashlib.sha256(sanitizer.read_bytes()).hexdigest(), errors=[])
    started = time.perf_counter()
    try:
        with (folder / 'stdout').open('xb') as stdout, (folder / 'stderr').open('xb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
            result['process_id'] = child.pid
            (folder / 'running.json').write_text(json.dumps(result, indent=2))
            try:
                result['exit_code'] = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=15)
        validate(result['exit_code'], (folder / 'stdout').read_text(), (folder / 'stderr').read_text(), context)
        for p in inputs:
            if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]:
                raise ValueError(f'Input changed during sanitizer: {p}')
        result['passed'] = True
    except BaseException as error:
        result['errors'].append(str(error) or type(error).__name__)
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        result['sha256'] = {**hashes, **{str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'stdout', folder/'stderr')}}
        (folder / 'result.json').write_text(json.dumps(result, indent=2))
    print(context, 'real-model sanitizer passed; no performance acceptance')


if __name__ == '__main__':
    main()
