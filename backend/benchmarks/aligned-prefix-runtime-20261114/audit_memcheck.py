"""Audit the completed first integrated memcheck without rewriting its evidence."""
import hashlib
import json
import math
from pathlib import Path
import re


def check_logs(metadata, stdout, stderr):
    if metadata.get('exit_code') != 0 or type(metadata.get('exit_code')) is not int or metadata.get('memcheck') is not True:
        raise ValueError('Sanitizer process did not complete successfully')
    if (metadata.get('prefix'), metadata.get('ring'), metadata.get('count')) != (513, 1, 4):
        raise ValueError('Unexpected workload')
    if metadata.get('aligned_admitted') is not True:
        raise ValueError('Aligned path was not admitted')
    summaries = re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$', stdout + '\n' + stderr, re.M)
    if summaries != ['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer error summary')
    matches = re.findall(r'^reset=(\d+) prefix=513 tokens=4 max_scaled=(\S+) execution=resident$', stdout, re.M)
    if [reset for reset, _ in matches] != ['0', '1']:
        raise ValueError('Incomplete lifecycle output')
    errors = [float(value) for _, value in matches]
    if any(not math.isfinite(value) or not 0 <= value <= .0005 for value in errors):
        raise ValueError('Numerical validation failed')
    for pattern in (r'CUDA aligned prefix: partial_queries=512 FDPP_batch_calls=(\d+)',
                    r'CUDA batched split attention calls: (\d+)'):
        calls = re.findall(pattern, stderr)
        if len(calls) != 2 or any(int(value) <= 0 for value in calls):
            raise ValueError('Required batched attention dispatch not observed in both sessions')
    return errors


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    tag = 'verify-prefix513-ring1-count4-memcheck'
    destination = out / f'{tag}-audit.json'
    if destination.exists():
        raise SystemExit('Refusing to overwrite audit evidence')
    original = out / f'{tag}.json'
    if not original.exists():
        raise SystemExit('Wait for terminal process metadata; a live log is not a result')
    metadata = json.loads(original.read_text())
    stdout, stderr = out / f'{tag}.log', out / f'{tag}.stderr.log'
    errors = check_logs(metadata, stdout.read_text(), stderr.read_text())
    command = metadata['command']
    assert command[1:5] == ['--tool', 'memcheck', '--error-exitcode', '99']
    sources = [out / f'{tag}.c', root / command[5], root / 'CUDA/resident.inc',
               root / 'CUDA/CMakeLists.txt', root / 'optcpp/attention/flash_decoding_pp_batch.cu']
    verified = {}
    for path in sources:
        key = str(path.relative_to(root))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == metadata['sha256'][key], f'Launch input changed: {path}'
        verified[key] = actual
    launch_script = out / 'run_verify_memcheck_v1.py.txt'
    assert hashlib.sha256(launch_script.read_bytes()).hexdigest() == metadata['sha256'][str((out / 'run_verify.py').relative_to(root))]
    expected = (root / 'backend/tests/test_verify.c').read_text().replace('size_t count=17,', 'size_t count=4,')
    assert sources[0].read_text() == expected, 'Lifecycle assertions changed'
    result = dict(accepted_benchmark=False, completed_child_exit_code=metadata['exit_code'],
                  numerical_lifecycle_passed=True, sanitizer_zero_errors=True, maximum_scaled_errors=errors,
                  original_parser_sanitizer_clear=metadata['sanitizer_clear'], verified_launch_inputs=verified,
                  evidence_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (original, stdout, stderr, launch_script)},
                  note='Audits both untouched streams. The original runner inspected stderr alone; sanitizer prints its summary on stdout. This supplements its metadata without rewriting it or rerunning the GPU workload. Capacity518 does not wrap the 1024-token sliding ring.')
    with destination.open('x') as output:
        json.dump(result, output, indent=2)
    print('Completed integrated sanitizer and lifecycle audit passed')


if __name__ == '__main__':
    main()
