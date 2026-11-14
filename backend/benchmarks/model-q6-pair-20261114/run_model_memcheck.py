"""Run the isolated QKV input-reuse binary under Compute Sanitizer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def file_key(path, root):
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def validate(code, stdout, stderr, context):
    if type(code) is not int or code != 0:
        raise ValueError('Sanitizer child failed')
    if re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$', stdout + '\n' + stderr, re.M) != ['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    if 'CUDA native metadata Q4: selected' not in stderr:
        raise ValueError('Native metadata kernel was not selected')
    graphs = [(int(a),int(b)) for a,b in re.findall(r'CUDA batched graph captures: (\d+) replays: (\d+)',stderr)]
    if len([1 for captures,replays in graphs if captures > 0 and replays > captures]) != 6:
        raise ValueError('Batched graphs were not captured and reused in all six verification cases')
    hits = [int(v) for v in re.findall(r'CUDA QKV input reuse capture/direct hits: (\d+)', stderr)]
    if len(hits) < 3 or min(hits) <= 0:
        raise ValueError('Insufficient positive QKV input-reuse trace markers')
    pair_lines=[line for line in stderr.splitlines() if 'CUDA paired rows capture/direct calls:' in line]
    matches=[re.fullmatch(r'CUDA paired rows capture/direct calls: (\d+)',line) for line in pair_lines]
    if len(matches)!=3 or not all(matches) or any(int(match[1])<=0 for match in matches):
        raise ValueError('Missing or malformed paired-row execution evidence')
    q6_lines=[line for line in stderr.splitlines() if 'CUDA Q6 paired rows capture/direct calls:' in line]
    q6_matches=[re.fullmatch(r'CUDA Q6 paired rows capture/direct calls: (\d+)',line) for line in q6_lines]
    if len(q6_matches)!=3 or not all(q6_matches) or any(int(match[1])<=0 for match in q6_matches):
        raise ValueError('Missing or malformed Q6 paired-row evidence')
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
    proofs = [(out / 'build.json', 'build_passed'),
              (out / 'control.json', 'passed'),
              (out / 'lifecycle-c23-257/result.json', 'passed'),
              (out / f'generation-{context}-c17/result.json', 'passed'),
              (out / 'generation-short-c23/result.json', 'passed'),
              (out / 'lifecycle-c17-257/result.json', 'passed'),
              (out / 'lifecycle-c17-8192/result.json', 'passed')]
    inputs = {path for path, key in proofs}
    for path, key in proofs:
        if not path.exists():
            raise SystemExit(f'Wait for completed prerequisite: {path}')
        proof = json.loads(path.read_text())
        if proof.get(key) is not True:
            raise SystemExit(f'Failed prerequisite: {path}')
        if 'single_download_assertions_passed' in proof and proof['single_download_assertions_passed'] is not True:
            raise SystemExit(f'Single-download lifecycle prerequisite failed: {path}')
        if 'integer_selected' in proof and (proof['exit_code'] != 0 or not proof['integer_selected'] or not proof['integer_calls'] or min(proof['integer_calls']) <= 0):
            raise SystemExit(f'Failed lifecycle prerequisite: {path}')
        for name, expected in proof['sha256'].items():
            inputs.add(root / name)
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
    if settings.get('NYA_CUDA_Q6_PAIR_ROWS') != '1':
        raise ValueError('Q6 paired rows must be enabled')
    if settings.get('NYA_CUDA_PAIR_ROWS') != '1':
        raise ValueError('Paired rows must be enabled')
    if settings.get('NYA_CUDA_QKV_INPUT_REUSE') != '1':
        raise ValueError('QKV input reuse must be enabled')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
    env.update(settings)
    sanitizer = root / '.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    binary = root / 'build-cuda/model-q6-pair-20261114/generation-c17.exe'
    command = [str(sanitizer), '--tool', 'memcheck', '--error-exitcode', '99', str(binary), context]
    inputs.update([Path(__file__), settings_path, identity_path, sanitizer, binary,
                   out.parent / 'gemma4-qkv-20261029/environment.py'])
    hashes = {file_key(p,root): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
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
            if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[file_key(p,root)]:
                raise ValueError(f'Input changed during sanitizer: {p}')
        result['passed'] = True
    except BaseException as error:
        result['errors'].append(str(error) or type(error).__name__)
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        result['sha256'] = {**hashes, **{file_key(p,root): hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'stdout', folder/'stderr')}}
        (folder / 'result.json').write_text(json.dumps(result, indent=2))
    print(context, 'real-model sanitizer passed; no performance acceptance')


if __name__ == '__main__':
    main()
