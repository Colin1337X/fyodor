"""Sanitize the complete numerical sweep without accepting performance."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def validate_oracle(proof):
    shapes = [(256, r) for r in (1, 7, 9, 15, 16, 17, 63, 64, 65)]
    shapes += [(3840, r) for r in (512, 2048, 4096, 8192, 15360)]
    shapes += [(4096, 3840), (8192, 3840), (15360, 3840)]
    bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
            0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x27ae, 0xa75a]
    expected = [(k, r, None, 'random') for k, r in shapes]
    expected += [(256, 17, [d, m], 'random') for d in bits for m in bits]
    expected += [(256, 17, None, p) for p in ('zero', 'one', 'alternating', 'subnormal', 'large', 'impulse')]
    cases = proof.get('cases', [])
    observed = [(c.get('columns'), c.get('rows'), c.get('half_bits'), c.get('pattern')) for c in cases]
    if proof.get('passed') is not True or observed != expected:
        raise ValueError('Incomplete, changed or reordered numerical inventory')
    for case in cases:
        fractions = case.get('maximum_fraction_of_float64_bound', {})
        if case.get('guards_clear') is not True or set(fractions) != {'nya_small_batch_12', 'nya_q4_tf32_three', 'nya_q4_tf32_three_hoisted'}:
            raise ValueError('Missing guards or kernel comparisons')
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in fractions.values()):
            raise ValueError('Independent numerical bound failed')
    rejections = proof.get('rejection_cases', [])
    if [(c.get('columns'), c.get('batch')) for c in rejections] != [
            (0, 4), (1, 4), (255, 4), (257, 4), (256, 0), (256, 1), (256, 2), (256, 8)]:
        raise ValueError('Unsupported-shape inventory changed')
    if any(c.get('output_unchanged') is not True for c in rejections):
        raise ValueError('Unsupported shape touched output')


def validate(code, stdout, stderr, proof, tool):
    if type(code) is not int or code != 0:
        raise ValueError('Sanitizer child failed')
    pattern = (r'^========= RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)\s*$'
               if tool == 'racecheck' else r'^========= ERROR SUMMARY: (\d+) errors\s*$')
    expected = [('0', '0', '0')] if tool == 'racecheck' else ['0']
    if re.findall(pattern, stdout + '\n' + stderr, re.M) != expected:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    validate_oracle(proof)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('memcheck', 'racecheck', 'synccheck'):
        raise SystemExit('Choose memcheck, racecheck or synccheck')
    tool = sys.argv[1]
    root, out = Path.cwd(), Path(__file__).resolve().parent
    raw_path = out / 'oracle.json'
    if not raw_path.exists():
        raise SystemExit('Wait for completed raw numerical validation')
    raw = json.loads(raw_path.read_text()); validate_oracle(raw)
    inputs = {root / name for name in raw['input_sha256']}
    for name, expected in raw['input_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError('Changed raw input: ' + name)
    sanitizer = root / '.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    inputs.update((Path(__file__).resolve(), raw_path, sanitizer, Path(sys.executable)))
    hashes = {str(p): digest(p) for p in inputs}
    stdout_path, stderr_path = out / (tool + '.log'), out / (tool + '.stderr.log')
    oracle_path, result_path = out / (tool + '-oracle.json'), out / (tool + '-run.json')
    running_path = out / (tool + '-running.json')
    if any(p.exists() for p in (stdout_path, stderr_path, oracle_path, result_path, running_path)):
        raise SystemExit('Refusing sanitizer evidence overwrite')
    command = [str(sanitizer), '--tool', tool, '--error-exitcode', '99',
               sys.executable, str(out / 'check_kernel.py'), '--' + tool]
    result = dict(passed=False, accepted_benchmark=False, command=command, input_sha256=hashes)
    start = time.perf_counter()
    try:
        with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
            child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            result['process_id'] = child.pid
            running_path.write_text(json.dumps(result, indent=2))
            try:
                result['exit_code'] = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate(); child.wait(timeout=15)
        proof = json.loads(oracle_path.read_text()) if oracle_path.exists() else {}
        validate(result['exit_code'], stdout_path.read_text(), stderr_path.read_text(), proof, tool)
        if proof['source_sha256'] != raw['source_sha256'] or proof['input_sha256'] != raw['input_sha256']:
            raise ValueError('Sanitized source or input identity changed')
        for path in inputs:
            if digest(path) != hashes[str(path)]:
                raise ValueError('Input changed during sanitizer: ' + str(path))
        result['passed'] = True
    except BaseException as error:
        result['failure'] = str(error) or type(error).__name__
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - start
        result['evidence_sha256'] = {p.name: digest(p) for p in (stdout_path, stderr_path, oracle_path) if p.exists()}
        result_path.write_text(json.dumps(result, indent=2))
    print('219 numerical cases and eight rejections passed under', tool)


if __name__ == '__main__':
    main()
