"""Pool both nine-sample orders without selecting or dropping observations."""
import json
import math
import sys
from pathlib import Path

out=Path(__file__).resolve().parent
directory=out/(sys.argv[1] if len(sys.argv)>1 else 'accepted-clean3')
meta=json.loads((directory/'metadata.json').read_text())
assert len(meta['runs'])==24 and all(r['exit_code']==0 and 'gpu_idle_after' in r for r in meta['runs'])
groups={}
for row in meta['runs']:
    prefix,engine,phase=row['name'].split('-')
    data=json.loads((directory/(row['name']+'.json')).read_text())
    for result in data['results']:
        key=(int(prefix),engine,result['test'])
        groups.setdefault(key,[]).append((data['repetitions'],result['tokens_per_second'],result['stddev']))
summary=[]
for key,parts in groups.items():
    n=sum(p[0] for p in parts)
    mean=sum(p[0]*p[1] for p in parts)/n
    sd=math.sqrt(sum((p[0]-1)*p[2]**2+p[0]*(p[1]-mean)**2 for p in parts)/(n-1))
    summary.append({'prefix':key[0],'engine':key[1],'test':key[2],'samples':n,'mean':mean,'sample_sd':sd})
print('| Prefix | Test | Before | After | Change | llama C API |')
print('|---:|---|---:|---:|---:|---:|')
for prefix,test in sorted({(r['prefix'],r['test']) for r in summary}):
    rows={r['engine']:r for r in summary if r['prefix']==prefix and r['test']==test}
    def display(engine):
        if engine not in rows:return '—'
        return f"{rows[engine]['mean']:.2f} ± {rows[engine]['sample_sd']:.2f}"
    change=(rows['after']['mean']/rows['before']['mean']-1)*100
    print(f'| {prefix} | {test} | {display("before")} | {display("after")} | {change:+.2f}% | {display("llama")} |')
(out/'accepted-summary.json').write_text(json.dumps(summary,indent=2))
resources=[]
for prefix in (0,256,512,1024,1536):
    for engine in ('before','after','llama'):
        samples=[];data=None
        for phase in ('a','b'):
            path=directory/f'{prefix}-{engine}-{phase}.json'
            if not path.exists():continue
            data=json.loads(path.read_text())
            samples+=json.loads(path.with_name(path.stem+'-resources.json').read_text())
        if data is None:continue
        resources.append({'prefix':prefix,'engine':engine,
            'maximum_observed_peak_working_set_bytes':max(s.get('peak_working_set',0) for s in samples),
            'maximum_sampled_private_commit_bytes':max(s.get('private_bytes',0) for s in samples),
            'maximum_sampled_whole_gpu_mib':max(int(s['whole_gpu'].split(',')[1].split()[0]) for s in samples if s.get('whole_gpu')),
            'explicit_device_bytes':data.get('device_memory_after')})
(out/'resource-summary.json').write_text(json.dumps(resources,indent=2))
