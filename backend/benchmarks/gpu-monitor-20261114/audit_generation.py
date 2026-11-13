"""Join completed MTP-cost decode intervals to sampled GPU evidence.

This checks evidence coverage, not benchmark acceptance or workload equivalence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from coverage import audit
from decode_intervals import parse

MODES = ('ordinary', 'mtp-serial2', 'mtp-batch2', 'mtp-batch4')
ORDER = [(round_, mode) for round_ in range(3)
         for mode in (reversed(MODES) if round_ == 1 else MODES)]
CASE = re.compile(r'CASE round=(\d+) mode=(\S+)')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def lifetime(run):
    if not isinstance(run, dict):
        raise ValueError('Malformed child process record')
    pid = run.get('process_id')
    before, after, end = (run.get(key) for key in
                          ('spawn_before_seconds', 'spawn_after_seconds', 'exit_observed_seconds'))
    if (run.get('exit_code') != 0 or isinstance(run.get('exit_code'), bool) or
        run.get('observer_clock') != 'qpc' or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or
        not all(finite(value) for value in (before, after, end)) or not 0 <= before <= after < end):
        raise ValueError('Invalid or incomplete child lifetime evidence')
    return pid, (before, end)


def audit_run(run, stdout, stderr, samples, *, luid, lifetimes):
    pid, bounds = lifetime(run)
    if lifetimes.get(pid) != bounds:
        raise ValueError('Child lifetime mismatch')
    outputs = [json.loads(line) for line in stdout.splitlines() if line.startswith('{"round":')]
    if [(row.get('round'), row.get('mode')) for row in outputs] != ORDER:
        raise ValueError('Incomplete or reordered generation outputs')
    if any(type(row.get('round')) is not int or row.get('parity') is not True or type(row.get('generated')) is not int or row['generated'] != 32 or
           type(row.get('prompt')) is not int or row['prompt'] <= 0 for row in outputs):
        raise ValueError('Output correctness or workload contract failed')
    sections, current = [], None
    for line in stderr.splitlines():
        if line.startswith('CASE '):
            match = CASE.fullmatch(line)
            if not match:
                raise ValueError('Malformed generation case')
            current = dict(key=(int(match[1]), match[2]), lines=[])
            sections.append(current)
        elif line.startswith(('Generation profile:', 'Generation interval:')):
            if current is None:
                raise ValueError('Timing without a generation case')
            current['lines'].append(line)
    if [section['key'] for section in sections] != ORDER:
        raise ValueError('Incomplete or reordered case labels')
    observations = []
    previous_end = bounds[0]
    for output, section in zip(outputs, sections):
        intervals = parse('\n'.join(section['lines']), clock='qpc')
        if len(intervals) != 1:
            raise ValueError('A case must have exactly one decode interval')
        interval = intervals[0]
        expected_mode = 'ordinary' if output['mode'] == 'ordinary' else 'mtp'
        if (interval['mode'] != expected_mode or interval['prompt'] != output['prompt'] or
            interval['generated'] != output['generated'] or
            not previous_end <= interval['start'] < interval['end'] <= bounds[1]):
            raise ValueError('Native decode interval does not match its completed case')
        previous_end = interval['end']
        coverage = audit(samples, luid=luid, start=interval['start'], end=interval['end'], lifetimes=lifetimes)
        observations.append(dict(round=output['round'], mode=output['mode'], interval=interval, coverage=coverage))
    return dict(accepted_benchmark=False, context=run['context'], process_id=pid,
                sampled_coverage_clear=all(row['coverage']['sampled_coverage_clear'] for row in observations),
                observations=observations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--luid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    result = dict(accepted_benchmark=False, sampled_coverage_clear=False, runs=[], errors=[], sha256={})
    def read(path):
        data = path.read_bytes()
        result['sha256'][str(path)] = hashlib.sha256(data).hexdigest()
        return data.decode('utf-8')
    try:
        metadata = json.loads(read(args.folder / 'timing.json'))
        if not isinstance(metadata, dict):
            raise ValueError('Malformed timing metadata')
        result['legacy_environment_gate_passed'] = metadata.get('gate_passed')
        samples = [json.loads(line) for line in read(args.samples).splitlines()]
        if not all(isinstance(sample, dict) for sample in samples):
            raise ValueError('Malformed monitor sample')
        runs = metadata['runs']
        if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
            raise ValueError('Malformed benchmark runs')
        contexts = [run['context'] for run in runs]
        if not runs or len(set(contexts)) != len(contexts) or any(c not in ('short', 'long') for c in contexts):
            raise ValueError('Invalid context inventory')
        spans = [lifetime(run) for run in runs]
        if len({pid for pid, _ in spans}) != len(spans):
            raise ValueError('PID reuse requires independent lifetime evidence')
        for run in runs:
            result['runs'].append(audit_run(run, read(args.folder / (run['context'] + '.stdout')),
                read(args.folder / (run['context'] + '.stderr')), samples,
                luid=args.luid.lower(), lifetimes=dict(spans)))
        result['sampled_coverage_clear'] = all(run['sampled_coverage_clear'] for run in result['runs'])
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        result['errors'].append(str(error))
    result['limitations'] = ['Sampled GPU coverage is not performance acceptance.',
        'CPU activity, thermal state, cross-engine equivalence and variance require separate checks.',
        'Spawn and observed-exit times bound a process lifetime; short-lived processes and PID reuse remain sampling limitations.']
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    if not result['sampled_coverage_clear']:
        raise SystemExit('GPU coverage evidence incomplete or contaminated; see output JSON')


if __name__ == '__main__':
    main()
