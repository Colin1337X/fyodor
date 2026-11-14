"""Summarize raw-counter transitions without deriving rates or accepting timings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(rows):
    if not rows or rows[0].get('kind') != 'header':
        raise ValueError('Missing capture header')
    if rows[-1] != dict(kind='end', failed=False, changed_inputs=[]):
        raise ValueError('Capture did not complete successfully')
    samples = [row for row in rows if row.get('kind') == 'sample']
    if len(samples) != rows[0]['samples'] or [r['index'] for r in samples] != list(range(len(samples))):
        raise ValueError('Incomplete or reordered capture')
    previous = {}
    invalid, decreases = [], []
    positive = Counter()
    for sample in samples:
        current = {}
        for item in sample['counters']:
            formatted, raw = item['formatted'], item['raw']
            name = formatted['instance']
            old = previous.get(name)
            record = dict(index=sample['index'], before=sample['before'], after=sample['after'],
                          instance=name, formatted=formatted, raw=raw, previous_raw=old)
            if formatted['status'] not in (0, 1):
                invalid.append(record)
            if (raw and old and raw['status'] in (0, 1) and old['status'] in (0, 1)
                    and (raw['first'] < old['first'] or raw['second'] < old['second'])):
                decreases.append(record)
            if formatted['engine_percent'] is not None and formatted['engine_percent'] > 0:
                positive[name] += 1
            if raw:
                current[name] = raw
        previous = current  # Do not compare across missing instances.
    return dict(accepted_benchmark=False, samples=len(samples), invalid_formatted=invalid,
                raw_decreases=decreases, positive_samples=dict(positive),
                limitations=['Raw decreases are not causally attributed.',
                             'No rate reconstruction or benchmark acceptance.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite analysis')
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    for name, digest in rows[0]['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Capture source changed: {name}')
    result = analyze(rows)
    result['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (args.input, Path(__file__))}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print('Samples:', result['samples'], 'invalid:', len(result['invalid_formatted']),
          'raw decreases:', len(result['raw_decreases']))


if __name__ == '__main__':
    main()
