"""Pool both diagnostic matrix orders, retaining every forward sample."""
import argparse
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--shapes', type=int, default=48)
args = parser.parse_args()
groups = {}
for version in ('baseline', 'candidate'):
    for order in ('forward', 'reverse'):
        raw = (args.directory / f'{version}-{order}.log').read_bytes()
        text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
        assert len(rows) == args.shapes * 3
        assert all(len(r['ms']) == 5 and all(math.isfinite(v) and v > 0 for v in r['ms']) for r in rows)
        selected = [r for r in rows if r['operation'] == 0]
        assert len(selected) == args.shapes
        keys = {(r['outputs'], r['inputs'], r['tokens']) for r in selected}
        assert len(keys) == args.shapes
        for row in selected:
            key = (row['outputs'], row['inputs'], row['tokens'])
            groups.setdefault(key, {}).setdefault(version, []).extend(row['ms'])
summary = []
for (rank, width, tokens), versions in sorted(groups.items()):
    result = {'rank': rank, 'width': width, 'tokens': tokens}
    for version in ('baseline', 'candidate'):
        values = versions[version]
        assert len(values) == 10
        result[version] = {'mean_ms': statistics.mean(values), 'sample_stdev_ms': statistics.stdev(values)}
    result['throughput_change_percent'] = (result['baseline']['mean_ms'] / result['candidate']['mean_ms'] - 1) * 100
    summary.append(result)
(args.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
