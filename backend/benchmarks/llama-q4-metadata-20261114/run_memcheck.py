"""Run the raw Q4 sweep under Compute Sanitizer and retain terminal evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def validate(exit_code, stdout, stderr, oracle):
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError('Sanitizer process failed')
    if re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$', stdout+'\n'+stderr, re.M) != ['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    cases = oracle.get('cases', [])
    if oracle.get('passed') is not True or len(cases) != 207:
        raise ValueError('Incomplete raw correctness sweep')
    if any(case.get('byte_exact') is not True or case.get('guards_clear') is not True for case in cases):
        raise ValueError('Raw parity or guards failed')
    shapes = [(256,1),(256,7),(256,9),(3840,512),(3840,2048),(3840,4096),
              (3840,8192),(3840,15360),(4096,3840),(8192,3840),(15360,3840)]
    bits = [0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
            0x3c00,0xbc00,0x7bff,0xfbff,0x27ae,0xa75a]
    expected = [(k,r,None) for k,r in shapes] + [(256,9,[d,m]) for d in bits for m in bits]
    actual = [(case.get('columns'),case.get('rows'),case.get('half_bits')) for case in cases]
    if actual != expected:
        raise ValueError('Raw shape/coefficient inventory differs')
    for case in cases:
        if case['columns'] == 256:
            bound = case.get('maximum_fraction_of_float64_bound')
            if (case.get('float64_checked') is not True or type(bound) not in (float,int)
                    or not 0 <= bound <= 1):
                raise ValueError('Missing or failed independent float64 bound')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    audit_path = out.parent / 'logit-readback-20261114/model-memcheck-short/result.json'
    if not audit_path.exists():
        raise SystemExit('Wait for the completed integrated sanitizer audit')
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
              ('small_batch.cu', 'small_batch_reuse.cu', 'interleaved_reuse.cu', 'interleaved_integer.cu', 'integer_launch.cu', 'llama_q4_metadata.cu')]
    assert hashlib.sha256(b'\n'.join(path.read_bytes() for path in paths)).hexdigest() == raw['source_sha256']
    for name, expected in raw['harness_sha256'].items():
        assert digest(out / name) == expected, name
    stdout_path, stderr_path = out / 'memcheck.log', out / 'memcheck.stderr.log'
    oracle_path, result_path = out / 'memcheck-oracle.json', out / 'memcheck-run.json'
    if any(path.exists() for path in (stdout_path, stderr_path, oracle_path, result_path)):
        raise SystemExit('Refusing to overwrite sanitizer evidence')
    inputs = paths + [out / 'check_kernel.py', out / 'common.py', Path(__file__), raw_path, audit_path, sanitizer]
    hashes = {str(path): digest(path) for path in inputs}
    command = [str(sanitizer), '--tool', 'memcheck', '--error-exitcode', '99',
               sys.executable, str(out / 'check_kernel.py'), '--memcheck']
    result = dict(accepted_benchmark=False, passed=False, command=command, input_sha256=hashes)
    start = time.perf_counter()
    try:
        with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
            process = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
        result['exit_code'] = process.returncode
        oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else {}
        validate(process.returncode, stdout_path.read_text(), stderr_path.read_text(), oracle)
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
    print('207 raw Q4 cases passed with zero Compute Sanitizer errors')


if __name__ == '__main__':
    main()
