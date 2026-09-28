import json, math, sys
from pathlib import Path
out=Path(__file__).resolve().parent; directory=out/(sys.argv[1] if len(sys.argv)>1 else 'accepted-remote')
meta=json.loads((directory/'metadata.json').read_text())
assert len(meta['runs'])==20
summary={}
for run in meta['runs']:
    assert run['exit_code']==0 and 'gpu_idle_after' in run and not run.get('contaminated')
    mode,prefix,engine,order=run['name'].split('-')
    data=json.loads((directory/(run['name']+'.json')).read_text())
    samples=json.loads((directory/(run['name']+'-resources.json')).read_text())
    assert samples and all(not s.get('competing_compute_processes') for s in samples)
    for result in data['results']:
        key=(mode,int(prefix),engine,result['test'])
        summary.setdefault(key,[]).append((data['repetitions'],result['tokens_per_second'],result['stddev']))
rows=[]
for key,parts in sorted(summary.items()):
    n=sum(p[0] for p in parts); mean=sum(p[0]*p[1] for p in parts)/n
    sd=math.sqrt(sum((p[0]-1)*p[2]**2+p[0]*(p[1]-mean)**2 for p in parts)/(n-1))
    rows.append(dict(zip(('mode','prefix','engine','test','samples','mean','sample_sd'),(*key,n,mean,sd))))
(out/'summary.json').write_text(json.dumps(rows,indent=2))
for mode,prefix,test in sorted({(r['mode'],r['prefix'],r['test']) for r in rows}):
    values={r['engine']:r for r in rows if (r['mode'],r['prefix'],r['test'])==(mode,prefix,test)}
    print(mode,prefix,test,[(e,round(r['mean'],2),round(r['sample_sd'],2)) for e,r in values.items()], 'change', round(100*(values['after']['mean']/values['before']['mean']-1),2))
