"""Run the raw Q4 sweep under Compute Sanitizer and retain terminal evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
if not __debug__: raise SystemExit('Assertions are required')


def validate(exit_code, stdout, stderr, oracle, tool='memcheck'):
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError('Sanitizer process failed')
    expression = (r'^========= RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)\s*$'
                  if tool == 'racecheck' else r'^========= ERROR SUMMARY: (\d+) errors\s*$')
    expected = [('0','0','0')] if tool == 'racecheck' else ['0']
    if re.findall(expression, stdout+'\n'+stderr, re.M) != expected:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    cases = oracle.get('cases', [])
    if oracle.get('passed') is not True or len(cases) != 207:
        raise ValueError('Incomplete raw correctness sweep')
    rejected=oracle.get('rejection_cases',[])
    if len(rejected)!=7 or any(case.get('output_unchanged') is not True for case in rejected):
        raise ValueError('Incomplete unsupported-input checks')
    if any(case.get('byte_exact') is not True or case.get('guards_clear') is not True for case in cases):
        raise ValueError('Raw parity or guards failed')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ('memcheck','racecheck','synccheck'): raise SystemExit('Choose memcheck, racecheck or synccheck')
    tool=sys.argv[1]
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    audit_path = out.parent / 'batch-graphs-20261114/boundary-memcheck-c23-512-ring1/result.json'
    if not audit_path.exists():
        raise SystemExit('Wait for the completed boundary sanitizer matrix')
    for standard in ('c17','c23'):
        for width in ('256','512'):
            for ring in ('0','1'):
                boundary = out.parent / f'batch-graphs-20261114/boundary-memcheck-{standard}-{width}-ring{ring}/result.json'
                if not boundary.exists():
                    raise SystemExit('Pending boundary sanitizer: '+str(boundary))
                state=json.loads(boundary.read_text())
                if state.get('passed') is not True or state.get('exit_code')!=0:
                    raise ValueError('Failed boundary sanitizer: '+str(boundary))
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] is True and audit['exit_code'] == 0
    raw_path = out / 'oracle.json'
    if not raw_path.exists():
        raise SystemExit('Complete the ordinary raw correctness sweep first')
    raw = json.loads(raw_path.read_text())
    assert raw['passed'] is True and len(raw['cases']) == 207
    sanitizer = root / '.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    paths = [root / 'CUDA' / name for name in ('matvec.cu', 'gemm.cu', 'resident.cu')]
    paths += [root / 'optcpp/matvec' / name for name in
              ('small_batch.cu', 'small_batch_reuse.cu', 'interleaved_reuse.cu', 'interleaved_integer.cu', 'integer_launch.cu', 'integer_metadata.cu', 'integer_metadata_index32.cu')]
    assert hashlib.sha256(b'\n'.join(path.read_bytes() for path in paths)).hexdigest() == raw['source_sha256']
    for name, expected in raw['input_sha256'].items():
        assert digest(root/name)==expected,name
    for name, expected in raw['harness_sha256'].items():
        assert digest(out / name) == expected, name
    stdout_path, stderr_path = out / (tool+'.log'), out / (tool+'.stderr.log')
    oracle_path, result_path = out / (tool+'-oracle.json'), out / (tool+'-run.json')
    if any(path.exists() for path in (stdout_path, stderr_path, oracle_path, result_path)):
        raise SystemExit('Refusing to overwrite sanitizer evidence')
    inputs = paths + [out / 'check_kernel.py', out / 'common.py', Path(__file__), raw_path, audit_path, sanitizer]
    hashes = {str(path): digest(path) for path in inputs}
    command = [str(sanitizer), '--tool', tool, '--error-exitcode', '99',
               sys.executable, str(out / 'check_kernel.py'), '--'+tool]
    result = dict(accepted_benchmark=False, passed=False, command=command, input_sha256=hashes)
    start = time.perf_counter()
    try:
        with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
            process = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
        result['exit_code'] = process.returncode
        oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else {}
        validate(process.returncode, stdout_path.read_text(), stderr_path.read_text(), oracle, tool)
        assert oracle['cases'] == raw['cases'], 'Sanitized case inventory changed'
        assert oracle['rejection_cases']==raw['rejection_cases'], 'Sanitized rejection inventory changed'
        assert oracle['source_sha256'] == raw['source_sha256']
        assert oracle['harness_sha256'] == raw['harness_sha256']
        for path in inputs:
            assert digest(path) == hashes[str(path)], f'Input changed during run: {path}'
        result['passed'] = True
    except BaseException as error:
        result['failure'] = str(error) or type(error).__name__
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - start
        result['evidence_sha256'] = {path.name:digest(path) for path in (stdout_path,stderr_path,oracle_path) if path.exists()}
        with result_path.open('x') as output:
            json.dump(result, output, indent=2)
    print('207 raw Q4 cases passed with zero',tool,'errors')


if __name__ == '__main__':
    main()
