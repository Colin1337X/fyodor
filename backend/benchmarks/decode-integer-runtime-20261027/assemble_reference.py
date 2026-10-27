from pathlib import Path
import json,shutil
p=Path(__file__).resolve().parent
out=p/'reference-combined';out.mkdir(exist_ok=False)
meta=None;environment=None
for directory in ('reference','reference-continuation-clean'):
 source=p/directory
 m=json.loads((source/'metadata.json').read_text());e=json.loads((source/'environment.json').read_text())
 assert 'failure' not in e
 if meta is None:
  meta={**m,'runs':[],'segments':[]};environment={**e,'activities':[],'segments':[]}
 else:
  for key in ('binary_sha256','model_sha256','dependency_sha256','monitor_sha256'):assert meta[key]==m[key]
 meta['segments'].append(directory);environment['segments'].append(directory)
 for run,activity in zip(m['runs'],e['activities']):
  assert run['exit_code']==0 and activity['passed']
  meta['runs'].append(run);environment['activities'].append(activity)
  for suffix in ('.json','.log'):shutil.copy2(source/(run['name']+suffix),out/(run['name']+suffix))
assert len(meta['runs'])==12 and len({r['name'] for r in meta['runs']})==12
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
