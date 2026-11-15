"""Explain retained coverage rejections without changing acceptance decisions."""
import hashlib
import json
from pathlib import Path


def explain(observation, collector_errors, collector_end):
    coverage = observation['coverage']
    interval = observation['interval']
    start, end = interval['start'], interval['end']
    stop = collector_end.get('kind') == 'end' and collector_end.get('requested_stop') is True
    telemetry = [error for error in coverage['errors']
                 if error not in ('unattributed GPU activity', 'missing successful cooperative termination')]
    if 'missing successful cooperative termination' in coverage['errors']:
        if not stop:
            telemetry.append('missing requested-stop end record')
        elif collector_end.get('failed') is True:
            telemetry.append('collector stopped as requested but retained a failure flag')
        else:
            telemetry.append('collector end record did not satisfy the original audit')
    overlapping = []
    unknown_time = 0
    for error in collector_errors:
        before = error.get('qpc_before_collect')
        after = error.get('qpc_observed_seconds')
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            unknown_time += 1
        elif before <= end and after >= start:
            overlapping.append(error)
    pids = sorted({row['process_id'] for row in coverage['positive_engines'] if not row['known_workload']})
    return dict(round=observation['round'], mode=observation['mode'], interval=interval,
        accepted_benchmark=False, sampled_coverage_clear=coverage['sampled_coverage_clear'],
        original_errors=coverage['errors'], telemetry_reasons=telemetry,
        requested_stop_record_present=stop, capture_wide_error_count=len(collector_errors),
        collector_errors_overlapping_interval=overlapping, collector_errors_without_time=unknown_time,
        unattributed_process_ids=pids,
        contamination_status='unresolved attribution' if pids else 'not established by this report',
        note='Error-call overlap is not sample coverage. No overlap does not prove a clean interval. System PID 4 may include benchmark-related driver work.')


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    base = out.parent / 'pdh-recovery-20261115'
    comparison = base / 'long-comparison-diagnostic.json'
    summary = json.loads(comparison.read_text())
    inputs = [Path(__file__).resolve(), comparison]
    reports = []
    for record in summary['records']:
        path = base / ('monitored-long-' + record['label']) / 'gpu-coverage.json'
        inputs.append(path)
        original = json.loads(path.read_text())
        reports.append(dict(label=record['label'], accepted_benchmark=False,
            observations=[explain(obs, record['collector_errors'], record['collector_end'])
                for run in original['runs'] for obs in run['observations']]))
    result = dict(accepted_benchmark=False, acceptance_policy_changed=False, reports=reports,
        sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out / 'report.json').write_text(json.dumps(result, indent=2))
    for report in reports:
        cases = report['observations']
        print(report['label'], len(cases), 'intervals;',
              sum(bool(c['collector_errors_overlapping_interval']) for c in cases),
              'with overlapping error calls; unattributed PIDs:',
              sorted({pid for c in cases for pid in c['unattributed_process_ids']}))


if __name__ == '__main__':
    main()
