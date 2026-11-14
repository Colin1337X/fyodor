"""Join transfer-only control phases to completed PDH captures, without acceptance."""
import hashlib,json,sys
from pathlib import Path
from correlate_decode import correlate

if len(sys.argv)!=4:raise SystemExit('Usage: correlate_transfer.py CAPTURE CONTROL LUID')
capture,control=map(Path,sys.argv[1:3]);hashes={}
for folder in (capture,control):
    path=folder/'result.json';proof=json.loads(path.read_text())
    hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    if folder==capture:
        if proof.get('inputs_unchanged') is not True or proof.get('errors'):raise ValueError('Changed or failed capture')
        count=proof['samples']
    else:
        if proof.get('passed') is not True or proof.get('source_unchanged') is not True or proof.get('cleanup_errors'):raise ValueError('Failed transfer control')
        pid=proof['process_id'];phases=proof['phases']
    for name,expected in proof['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed input: '+name)
samples=[json.loads(l) for l in (capture/'samples.jsonl').read_text().splitlines()]
if len(samples)!=count:raise ValueError('Incomplete capture')
observations=correlate(samples,phases,sys.argv[3],pid)
for observation in observations:
    phase=observation['interval']
    interior=[b['index'] for a,b in zip(samples,samples[1:]) if a['qpc_start']>=phase['start'] and b['qpc_end']<=phase['end']]
    observation['interior_sample_indices']=interior
    observation['interior_events']=[e for e in observation['events'] if e['index'] in interior]
for path in (Path(__file__),Path(__file__).with_name('correlate_decode.py')):
    hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
result=dict(accepted_benchmark=False,process_id=pid,observations=observations,sha256=hashes,
            limitation='Diagnostic correlation only. Boundary-overlapping and wholly interior samples are distinguished; neither establishes causality or model performance.')
with (capture/('transfer-overlap-'+control.name+'.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print([(o['interval']['round'],o['interval']['mode'],len(o['interior_sample_indices']),len(o['interior_events']),o['bracketed']) for o in observations])
