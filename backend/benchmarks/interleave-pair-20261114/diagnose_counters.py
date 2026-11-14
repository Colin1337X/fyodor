"""Explain recorded counter rejection without changing coverage or accepting timing."""
import hashlib,json,math
from pathlib import Path

out=Path(__file__).resolve().parent
folder=out/'monitored-short-A'
destination=out/'monitored-short-A-counter-diagnosis.json'
if destination.exists():raise SystemExit('Refusing to overwrite diagnostic evidence')
audit_path=folder/'gpu-coverage.json'
raw_path=folder/'gpu-pdh.jsonl'
audit=json.loads(audit_path.read_text())
for name,digest in audit['sha256'].items():
 path=Path(name) if Path(name).is_absolute() else folder/name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
samples=[r for r in map(json.loads,raw_path.read_text().splitlines()) if r.get('kind')=='sample']
result={'accepted_benchmark':False,'changes_acceptance':False,'cases':[],
 'limitations':['Recorded sampled intervals overlap decode but may include activity outside it.',
 'System engine activity is not causally attributed by this report.']}
for run in audit['runs']:
 for observation in run['observations']:
  span=observation['interval']; entries=[]
  for previous,sample in zip(samples,samples[1:]):
   start,end=previous['qpc_observed_seconds'],sample['qpc_observed_seconds']
   if end<=span['start'] or start>=span['end']:continue
   unknown={a['process_id'] for a in observation['coverage']['positive_engines']
            if not a['known_workload'] and a['interval']==[start,end]}
   for counter in sample['target_counters']:
    value=counter.get('engine_percent')
    finite=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    invalid=counter.get('status') not in (0,1) or not finite or not 0<=value<=100
    unknown_positive=not invalid and value>0 and counter['process_id'] in unknown
    if invalid or unknown_positive:
     entries.append(dict(counter=counter,interval=[start,end],invalid=invalid,
                         unknown_positive=unknown_positive))
  result['cases'].append(dict(round=observation['round'],mode=observation['mode'],
    decode_interval=span,original_errors=observation['coverage']['errors'],counter_details=entries))
result['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),audit_path,raw_path)}
with destination.open('x') as f:json.dump(result,f,indent=2)
print('Retained counter identities for',len(result['cases']),'cases; no acceptance change')
