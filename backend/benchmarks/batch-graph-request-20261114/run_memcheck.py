"""Require real graph replay parity and one successful Compute Sanitizer summary."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def validate_cases(oracle):
    expected = list(itertools.product((0,1,12,14),(256,3840),(2,4),range(12)))
    cases = oracle.get('cases', [])
    if oracle.get('passed') is not True or oracle.get('errors') != [] or len(cases) != len(expected):
        raise ValueError('Incomplete or failed replay sweep')
    for case, key in zip(cases, expected):
        fields = ('dtype','width','count','round')
        if any(type(case.get(f)) is not int for f in fields) or tuple(case[f] for f in fields) != key:
            raise ValueError('Unexpected replay case inventory/order')
        round_ = key[3]
        if case.get('tokens') != [(round_*3+q*5)%17 for q in range(4)]:
            raise ValueError('Wrong replay token sequence')
        if case.get('position') != (0,255,256,8191,8192,0xfffffffb)[round_%6]:
            raise ValueError('Wrong replay position')
        if case.get('parity') is not True or case.get('guards') is not True:
            raise ValueError('Replay parity or guards failed')


def validate(exit_code, stdout, stderr, oracle):
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError('Sanitizer process failed')
    if re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$', stdout+'\n'+stderr, re.M) != ['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    validate_cases(oracle)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import check_replay
    root, out = check_replay.root, check_replay.out
    inputs = check_replay.prerequisites()  # No driver/context imports.
    raw_path = out/'oracle.json'
    if not raw_path.exists():
        raise SystemExit('Complete ordinary graph replay validation first')
    raw = json.loads(raw_path.read_text())
    validate_cases(raw)
    for name, expected in raw['sha256'].items():
        path = root/name
        if digest(path) != expected:
            raise ValueError(f'Changed raw input: {name}')
        inputs.add(path)
    sanitizer = root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    inputs.update([raw_path, sanitizer, Path(__file__).resolve()])
    paths = [out/name for name in ('memcheck.log','memcheck.stderr.log','memcheck-oracle.json','memcheck-run.json')]
    stdout_path, stderr_path, oracle_path, result_path = paths
    if any(p.exists() for p in paths):
        raise SystemExit('Refusing to overwrite sanitizer evidence')
    hashes = {str(p.relative_to(root)):digest(p) for p in inputs}
    command = [str(sanitizer),'--tool','memcheck','--error-exitcode','99',
               sys.executable,str(out/'check_replay.py'),'--memcheck']
    result = dict(passed=False,accepted_benchmark=False,command=command,sha256=hashes)
    started = time.perf_counter()
    try:
        with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
            child = subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            result['process_id'] = child.pid
            try:
                result['exit_code'] = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=15)
        oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else {}
        validate(result['exit_code'],stdout_path.read_text(),stderr_path.read_text(),oracle)
        if oracle['sha256'] != raw['sha256'] or oracle['options'] != raw['options']:
            raise ValueError('Raw/sanitizer inputs or compilation options differ')
        for name, expected in hashes.items():
            if digest(root/name) != expected:
                raise ValueError(f'Input changed during sanitizer: {name}')
        result['passed'] = True
    except BaseException as error:
        result['failure'] = str(error) or type(error).__name__
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        result['evidence_sha256'] = {p.name:digest(p) for p in paths[:3] if p.exists()}
        with result_path.open('x') as output:
            json.dump(result,output,indent=2)
    print('192 graph replays passed with zero Compute Sanitizer errors')


if __name__ == '__main__':
    main()
