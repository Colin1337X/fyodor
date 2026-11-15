"""Validate and summarize the sequential, diagnostic-only overhead comparison."""
import hashlib,json
from pathlib import Path
root=Path.cwd(); folder=Path(__file__).resolve().parent
labels=['OverheadBase5','OverheadNoMatch5','OverheadCandidate5','OverheadNoStreamRaces5']
records=[]
for label in labels:
 p=folder/label/'result.json'; r=json.loads(p.read_text())
 assert r['passed'] and not r['accepted_benchmark'] and not r['changed_inputs'],label
 for name,h in r['sha256'].items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
 records.append(dict(label=label,elapsed_s=r['elapsed_s'],memcheck=r['memcheck'],kernel_filter=r['kernel_filter'],stream_races=r.get('stream_races','all') if r['memcheck'] else None,prefill_summaries=r['prefill_summaries'],result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
base=records[0]['elapsed_s']
for r in records:r['elapsed_ratio_to_baseline']=r['elapsed_s']/base
out=folder/'overhead-comparison.json'
assert not out.exists(),'Refusing to overwrite evidence'
out.write_text(json.dumps(dict(passed=True,accepted_benchmark=False,prefix=5,standard='c23',repetitions=1,records=records,conclusion='Short fallback diagnostic only. Neither no-match filtering nor disabling stream-ordered race checks constitutes candidate safety evidence. Exact overhead mechanism remains unresolved.'),indent=2))
print(json.dumps([ {k:v for k,v in r.items() if k!='prefill_summaries'} for r in records],indent=2))
