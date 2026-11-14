"""Describe all audited rounds without certifying performance or discarding failures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from audit_joint_one import base


def summarize(audit):
    if audit.get('errors'):
        raise ValueError('Cannot summarize an incomplete audit')
    order = base.case_order(audit['mode_set'])
    runs = audit['runs']
    contexts = [run['context'] for run in runs]
    if not runs or len(set(contexts)) != len(contexts):
        raise ValueError('Missing or duplicate contexts')
    result = dict(accepted_benchmark=False, diagnostic_only=True, runs=[])
    for run in runs:
        if run['context'] not in ('short', 'long'):
            raise ValueError('Unknown context')
        observations = run['observations']
        if [(r['round'], r['mode']) for r in observations] != order:
            raise ValueError('Incomplete or reordered rounds')
        cases, groups = [], {}
        for row in observations:
            interval = row['interval']
            duration = interval['milliseconds']
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0
                    or type(interval['generated']) is not int or interval['generated'] != 32):
                raise ValueError('Invalid duration or output count')
            coverage = row['coverage']
            if type(coverage.get('sampled_coverage_clear')) is not bool:
                raise ValueError('Missing coverage outcome')
            cases.append(dict(round=row['round'], mode=row['mode'],
                              milliseconds=duration, coverage=coverage))
            groups.setdefault(row['mode'], []).append(duration)
        result['runs'].append(dict(context=run['context'], cases=cases,
            modes={mode: dict(rounds=len(values), all_rounds_mean_ms=statistics.mean(values),
                             sample_stdev_ms=statistics.stdev(values))
                   for mode, values in groups.items()}))
    result['limitations'] = ['All three rounds retained, including warmup and failed coverage.',
        'Descriptive timings only; no clean-run selection, speedup ratio or score acceptance.',
        'Coverage clearance alone does not establish CPU, thermal or workload equivalence.']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    audit = json.loads(args.audit.read_text())
    if not audit.get('sha256'):
        raise ValueError('Missing audit input hashes')
    checked = {}
    for name, expected in audit['sha256'].items():
        path = Path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Changed audit input: ' + name)
        checked[str(path.resolve())] = actual
    result = summarize(audit)
    for path in (args.audit, Path(__file__)):
        checked[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['sha256'] = checked
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print('Diagnostic summary written; no performance acceptance')


if __name__ == '__main__':
    main()
