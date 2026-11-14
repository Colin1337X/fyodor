"""Classify invalid PDH entries by observed instance history, without accepting them."""
import hashlib
import json
import re
import sys
from pathlib import Path


def classify(rows, luid):
    if not re.fullmatch(r'luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}', luid):
        raise ValueError('Invalid adapter identity')
    previous = None
    invalid = []
    failures = []
    for index, row in enumerate(rows):
        if row['index'] != index:
            raise ValueError('Missing or reordered samples')
        raw_rows = row.get('raw', {}).get('rows')
        formatted = row.get('formatted', {}).get('rows')
        if row.get('collect_status', 0) or raw_rows is None or formatted is None:
            failures.append(index)
        raw = {e['instance']: e for e in raw_rows or []}
        for entry in formatted or []:
            name = entry['instance']
            if not re.match(r'pid_\d+_' + re.escape(luid) + r'_', name):
                continue
            if entry['status'] in (0, 1):
                continue
            prior = previous.get(name) if previous is not None else None
            current = raw.get(name)
            # No preceding array means unknown history, not a new instance.
            history = ('unknown' if previous is None else
                       'present' if prior is not None else 'new')
            invalid.append(dict(index=index, instance=name, status=entry['status'],
                                history=history,
                                raw_status=current['status'] if current else None,
                                prior_raw_status=prior['status'] if prior else None))
        previous = raw if raw_rows is not None and not row.get('collect_status', 0) else None
    return dict(accepted_benchmark=False, samples=len(rows), api_failure_indices=failures,
                invalid=invalid, history_counts={h: sum(e['history'] == h for e in invalid)
                                                for h in ('new', 'present', 'unknown')},
                limitation='Observed instance history only; new does not establish the cause of invalid data. All invalid entries remain invalid.')


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: classify_invalid.py COMPLETED_FOLDER TARGET_LUID')
    folder = Path(sys.argv[1])
    proof_path = folder / 'result.json'
    proof = json.loads(proof_path.read_text())
    if proof.get('inputs_unchanged') is not True or proof.get('errors'):
        raise ValueError('Failed or changed capture')
    for name, expected in proof['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed capture input/output: ' + name)
    log = folder / 'samples.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    if len(rows) != proof['samples']:
        raise ValueError('Incomplete sample inventory')
    result = classify(rows, sys.argv[2])
    result['sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), proof_path, log)}
    with (folder / 'invalid-history.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('invalid','sha256')}))


if __name__ == '__main__':
    main()
