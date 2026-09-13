import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'backend/benchmarks'
for folder in ['stage8-final']:
    print(folder)
    for version in ['before','after','llama']:
        for prefix in [0,1024]:
            paths=[root/folder/f'{prefix}-{version}-{phase}.json' for phase in ['a','b']]
            if not all(p.exists() for p in paths):continue
            values=[json.loads(p.read_text()) for p in paths]
            for i,r in enumerate(values[0]['results']):
                pair=[v['results'][i] for v in values];mean=sum(x['tokens_per_second'] for x in pair)/2
                sd=math.sqrt(sum(8*x['stddev']**2+9*(x['tokens_per_second']-mean)**2 for x in pair)/17)
                print(version,prefix,r['test'],round(mean,3),round(sd,3),'latency',round(sum(x['mean_latency_ms'] for x in pair)/2,3))
            if 'device_scratch_bytes' in values[0]:print('scratch MiB',values[0]['device_scratch_bytes']/1048576)

