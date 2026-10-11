"""Require complete monitored comparisons and exact outputs before pooling."""
import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
data = json.loads((args.directory / 'results.json').read_text())
environment = json.loads((args.directory / 'environment.json').read_text())
config = data['configuration']
expected = len(config['contexts']) * config['repeats'] * 2
assert data['all_outputs_exact'] and config['monitor_environment']
assert len(data['runs']) == expected == len(environment['activities'])
assert 'failure' not in environment
assert all(a['passed'] and a['samples'] and
           all(not s['competitors'] for s in a['samples'])
           for a in environment['activities'])
summary = {'method': 'All process means pooled by context; exact checkpoint/export hashes required',
           'hardware': environment['hardware'], 'contexts': []}
for context in config['contexts']:
    runs = [r for r in data['runs'] if r['context'] == context]
    assert len({r['checkpoint_sha256'] for r in runs}) == 1
    assert len({r['export_sha256'] for r in runs}) == 1
    result = {'context': context, 'outputs_exact': True}
    for version in ('before', 'after'):
        selected = [r for r in runs if r['version'] == version]
        assert {r['repeat'] for r in selected} == set(range(config['repeats']))
        means = [r['mean_step_ms'] for r in selected]
        result[version] = {'mean_step_ms': statistics.mean(means),
                           'process_mean_sample_stdev': statistics.stdev(means),
                           'process_repetitions': len(means)}
    result['throughput_improvement_percent'] = (
        result['before']['mean_step_ms'] / result['after']['mean_step_ms'] - 1) * 100
    summary['contexts'].append(result)
(args.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
