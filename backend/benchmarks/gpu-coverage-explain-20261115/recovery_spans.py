"""Locate decode intervals near failed reads, including following sample pairs."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
out = Path(__file__).resolve().parent
base = out.parent / 'pdh-recovery-20261115'
reports = []
hashes = {}
for label in ('DisabledRecoveryA', 'EnabledRecoveryA'):
    folder = base / ('monitored-long-' + label)
    capture = folder / 'gpu-pdh.jsonl'
    audit_path = folder / 'gpu-coverage.json'
    audit = json.loads(audit_path.read_text())
    observations = [o for run in audit['runs'] for o in run['observations']]
    previous = None
    spans = []
    pending = []
    digest = hashlib.sha256()
    with capture.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row['kind'] == 'sample':
                now = row['qpc_observed_seconds']
                for span in pending:
                    span['following_samples'].append(now)
                    if len(span['following_samples']) == 2:
                        span['end'] = now
                pending = [span for span in pending if len(span['following_samples']) < 2]
                previous = now
            elif row['kind'] == 'error':
                span = dict(start=previous, end=None, error=row, following_samples=[])
                pending.append(span)
                spans.append(span)
    for span in spans:
        span['overlapping_decode_cases'] = [dict(round=o['round'], mode=o['mode']) for o in observations
            if span['start'] is None or span['end'] is None or
            (span['start'] <= o['interval']['end'] and span['end'] >= o['interval']['start'])]
    hashes[str(capture.relative_to(root))] = digest.hexdigest()
    hashes[str(audit_path.relative_to(root))] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    reports.append(dict(label=label, error_spans=spans))
    print(label, len(spans), 'failed reads;', sum(bool(s['overlapping_decode_cases']) for s in spans),
          'expanded recovery spans overlap decode')
hashes[str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(out / 'recovery-spans.json').write_text(json.dumps(dict(accepted_benchmark=False, reports=reports,
    sha256=hashes, scope='Temporal diagnostic only. Spans cover the preceding emitted sample through two following emitted samples; this is not proof of provider recovery or complete coverage.'), indent=2))
