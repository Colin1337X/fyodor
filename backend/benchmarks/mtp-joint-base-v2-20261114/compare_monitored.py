"""Compare one completed context only when both sampled coverage audits clear."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

reference = Path(__file__).resolve().parent.parent / 'aligned-prefix-runtime-20261114'
sys.path.insert(0, str(reference))
from compare_generation import compare, native_text


def load_audit(folder):
    audit = json.loads((folder / 'gpu-coverage.json').read_text())
    for name, expected in audit['sha256'].items():
        path = Path(name) if Path(name).is_absolute() else folder / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed audit input: {path}')
    if audit.get('errors') or audit.get('sampled_coverage_clear') is not True:
        raise ValueError(f'Incomplete or contaminated coverage: {folder}')
    timing = json.loads((folder / 'timing.json').read_text())
    if timing.get('gate_passed') is not True:
        raise ValueError(f'Idle gate did not pass: {folder}')
    runs = audit['runs']
    if len(runs) != 1 or runs[0]['context'] not in ('short', 'long'):
        raise ValueError('Expected exactly one context per audit')
    if not runs[0]['observations'] or any(row['coverage']['sampled_coverage_clear'] is not True
                                       for row in runs[0]['observations']):
        raise ValueError('Per-case coverage did not clear')
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('native', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite comparison evidence')
    result = dict(accepted_benchmark=False, errors=[], sha256={},
        limitations=['Diagnostic generation comparison, not fixed-forward score acceptance.',
                     'Two warmed observations; reversed cross-engine order and CPU/thermal equivalence still required.'])
    try:
        native, llama = load_audit(args.native), load_audit(args.reference)
        if native.get('mode_set') != 'joint-base':
            raise ValueError('Native audit is not joint-base')
        n, l = native['runs'][0], llama['runs'][0]
        if n['context'] != l['context']:
            raise ValueError('Cross-engine contexts differ')
        text_path = args.native / (n['context'] + '.stdout')
        result['comparison'] = compare(n['observations'], l['observations'],
            native_text(text_path.read_text()), mode_set='joint-base')
        for path in (args.native / 'gpu-coverage.json', args.reference / 'gpu-coverage.json',
                     text_path, Path(__file__), reference / 'compare_generation.py'):
            result['sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, KeyError, TypeError, ValueError) as error:
        result['errors'].append(str(error))
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    if result['errors']:
        raise SystemExit('Comparison rejected; no ratios emitted')


if __name__ == '__main__':
    main()
