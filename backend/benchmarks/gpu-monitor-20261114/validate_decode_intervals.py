"""Run native CPU generation assertions and cross-check emitted QPC intervals."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from decode_intervals import INTERVAL, PROFILE, parse

root = Path.cwd()
out = Path(__file__).resolve().parent
label = sys.argv[1] if len(sys.argv) == 2 else ''
if len(sys.argv) > 2 or (label and (not label.isascii() or not label.isalnum())):
    raise SystemExit('Usage: validate_decode_intervals.py [ALPHANUMERIC_LABEL]')
suffix = '-' + label if label else ''
destination = out / f'decode-interval-validation{suffix}.json'
paths = [out / f'decode-interval-{standard}-{enabled}{suffix}.{stream}'
         for standard in ('c17', 'c23') for enabled in (0, 1) for stream in ('stdout', 'stderr')]
if destination.exists() or any(path.exists() for path in paths):
    raise SystemExit('Refusing to overwrite evidence')
env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
env['NYA_COMPUTE'] = 'cpu'
meta = dict(accepted_benchmark=False, cases=[],
            note='Tiny CPU correctness tests while GPU sanitizer runs; not a performance benchmark.',
            source_sha256=hashlib.sha256((root / 'backend/core/llm_cpu.c').read_bytes()).hexdigest())
for standard, build in (('c17', 'build-cpu'), ('c23', 'build-cuda-c23')):
    exe = root / build / 'nya-test-generation.exe'
    for enabled in (0, 1):
        env['NYA_GENERATION_PROFILE'] = str(enabled)
        prefix = out / f'decode-interval-{standard}-{enabled}{suffix}'
        command = [str(exe), str(root / build / f'decode-interval-{enabled}{suffix}.gguf')]
        with prefix.with_suffix('.stdout').open('x') as stdout, prefix.with_suffix('.stderr').open('x') as stderr:
            before = time.perf_counter()
            result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
            after = time.perf_counter()
        text = prefix.with_suffix('.stderr').read_text()
        rows, zero_duration, parse_error = [], [], None
        if enabled and result.returncode == 0:
            try:
                lines = [line for line in text.splitlines() if line.startswith(('Generation profile:', 'Generation interval:'))]
                assert len(lines) % 2 == 0
                for index in range(0, len(lines), 2):
                    pair = '\n'.join(lines[index:index + 2])
                    interval = INTERVAL.fullmatch(lines[index + 1])
                    if interval and interval[7] == '0':
                        duration = PROFILE.fullmatch(lines[index])
                        assert duration and float(duration[4]) == 0 and 0 < float(interval[5]) == float(interval[6])
                        assert duration.group(1, 2, 3) == interval.group(1, 2, 3) and interval[4] == 'qpc'
                        # One-token synthetic calls can finish within a clock
                        # tick. They must be explicitly invalid for benchmarking.
                        try:
                            parse(pair, clock='qpc')
                        except ValueError:
                            zero_duration.append(pair)
                        else:
                            raise AssertionError('Zero duration was accepted')
                    else:
                        rows.extend(parse(pair, clock='qpc'))
            except (AssertionError, ValueError) as error:
                parse_error = str(error) or type(error).__name__
        valid = result.returncode == 0 and (bool(rows) if enabled else 'Generation interval:' not in text)
        valid &= parse_error is None
        if rows:
            valid &= all(before <= row['start'] < row['end'] <= after for row in rows)
            valid &= {'ordinary', 'speculative'} <= {row['mode'] for row in rows}
            valid &= all(a['end'] <= b['start'] for a, b in zip(rows, rows[1:]))
        case = dict(standard=standard, profile_enabled=enabled, command=command,
                    binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                    exit_code=result.returncode, parent_start=before, parent_end=after,
                    intervals=rows, rejected_zero_duration=zero_duration, parse_error=parse_error, valid=valid)
        meta['cases'].append(case)
        destination.write_text(json.dumps(meta, indent=2))
        if not valid:
            raise SystemExit('Native interval validation failed')
        print(standard, enabled, len(rows), 'intervals validated', flush=True)
