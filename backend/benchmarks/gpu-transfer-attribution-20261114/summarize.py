"""Describe whole PDH intervals inside transfer controls; never waive activity."""
import json,math
from pathlib import Path

PHASES=('idle-before','pageable-upload','pageable-download','pinned-upload','pinned-download','device-copy','idle-after')
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def summarize(rows,phases,worker):
    if [p.get('phase') for p in phases]!=list(PHASES):raise ValueError('Incomplete phase inventory')
    if not rows or rows[0].get('kind')!='header' or rows[-1].get('kind')!='end':raise ValueError('Incomplete capture')
    if rows[-1].get('requested_stop') is not True or rows[-1].get('failed') is not False or any(r.get('kind')=='error' for r in rows):
        raise ValueError('Collector failed')
    if type(worker) is not int or worker<=4:raise ValueError('Invalid worker PID')
    prior=0
    for phase in phases:
        if not all(finite(phase.get(k)) for k in ('start','end')) or not prior<=phase['start']<phase['end']:
            raise ValueError('Invalid phase bounds')
        prior=phase['end']
    samples=[r for r in rows if r.get('kind')=='sample']
    times=[r.get('qpc_observed_seconds') for r in samples]
    if any(not finite(t) for t in times) or any(a>=b for a,b in zip(times,times[1:])):raise ValueError('Invalid sample clock')
    result=dict(accepted_benchmark=False,system_attribution_proven=False,phases=[])
    for phase in phases:
        intervals=[];errors=[];activity=[]
        for left,right in zip(samples,samples[1:]):
            start,end=left['qpc_observed_seconds'],right['qpc_observed_seconds']
            if not phase['start']<=start<end<=phase['end']:continue
            intervals.append([start,end])
            if end-start>1.5:errors.append('sample gap')
            for counter in right['target_counters']:
                value=counter.get('engine_percent')
                if counter.get('status') not in (0,1) or not finite(value) or not 0<=value<=100:
                    errors.append('invalid target counter');continue
                if value>0:
                    pid=counter.get('process_id')
                    activity.append(dict(interval=[start,end],process_id=pid,instance=counter.get('instance'),engine_percent=value,
                        category='worker' if pid==worker else 'System-unattributed' if pid==4 else 'other-unattributed'))
        if len(intervals)<2:errors.append('insufficient whole intervals')
        result['phases'].append(dict(phase=phase['phase'],intervals=intervals,errors=errors,positive_engines=activity))
    return result

def main():
    import hashlib
    folder=Path(__file__).resolve().parent/'controls';destination=folder/'summary.json'
    if destination.exists():raise SystemExit('Refusing to overwrite summary')
    run=json.loads((folder/'result.json').read_text())
    if run['errors'] or run['exit_code']!=0:raise ValueError('Control workload failed')
    for name,expected in run['evidence_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed evidence: '+name)
    rows=[json.loads(line) for line in (folder/'gpu-pdh.jsonl').read_text().splitlines()]
    phases=[json.loads(line) for line in (folder/'worker.stdout').read_text().splitlines()]
    result=summarize(rows,phases,run['process_id'])
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
