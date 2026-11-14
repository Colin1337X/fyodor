"""Summarize a completed diagnostic without certifying GPU activity or timing."""
import hashlib
import json
import math
import re
from pathlib import Path
import struct
import sys


def summarize(rows, luid):
    if not re.fullmatch(r"luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}", luid):
        raise ValueError("Invalid adapter LUID")
    previous = {}
    prior_names = None
    differences = []
    over = []
    changes = []
    failures = []
    invalid = {'raw': 0, 'formatted': 0}
    for row in rows:
        index = row['index']
        for field in ('raw', 'formatted'):
            entries = row.get(field, {}).get('rows')
            if entries is None:
                failures.append({'index': index, 'field': field,
                                 'calls': row.get(field, {}).get('calls', [])})
            else:
                invalid[field] += sum(v['status'] not in (0, 1) for v in entries)
        raw = {v['instance']: v for v in row.get('raw', {}).get('rows') or []}
        names = set(raw)
        if prior_names is not None and names != prior_names:
            changes.append(dict(index=index, added=sorted(names-prior_names),
                                removed=sorted(prior_names-names)))
        prior_names = names
        for entry in row.get('formatted', {}).get('rows') or []:
            name = entry['instance']
            if luid not in name:
                continue
            now, before = raw.get(name), previous.get(name)
            if not now or not before or any(v['status'] not in (0, 1) for v in (entry, now, before)):
                continue
            delta = now['second'] - before['second']
            if delta <= 0:
                continue
            value = struct.unpack('<d', bytes.fromhex(entry['value_bits']))[0]
            if not math.isfinite(value):
                continue
            ratio = 100 * (now['first'] - before['first']) / delta
            differences.append(abs(value-ratio))
            if value > 100:
                over.append(dict(index=index, instance=name, formatted=value, raw_delta_ratio=ratio))
        previous = raw
    return dict(accepted_benchmark=False, samples=len(rows), paired_counters=len(differences),
                maximum_difference=max(differences, default=None), above_100=over,
                instance_changes=changes, api_failures=failures, invalid_status_counts=invalid,
                limitation='Raw-delta agreement is diagnostic, not proof of actual utilization or causal PID attribution.')


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: analyze.py COMPLETED_FOLDER TARGET_LUID')
    folder = Path(sys.argv[1])
    proof_path = folder/'result.json'
    if not proof_path.exists():
        raise SystemExit('Wait for diagnostic completion')
    proof = json.loads(proof_path.read_text())
    if proof.get('inputs_unchanged') is not True or proof.get('errors'):
        raise SystemExit('Diagnostic failed or changed inputs')
    for name, expected in proof['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise SystemExit('Changed diagnostic input/output: '+name)
    log = folder/'samples.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    if len(rows) != proof['samples'] or [r['index'] for r in rows] != list(range(len(rows))):
        raise SystemExit('Missing or reordered samples')
    result = summarize(rows, sys.argv[2].lower())
    result['sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), proof_path, log)}
    with (folder/'analysis.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sha256','above_100','instance_changes')}, allow_nan=False))

if __name__ == '__main__':
    main()
