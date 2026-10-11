from pathlib import Path
import json,statistics
directory=Path(__file__).resolve().parent
groups={}
for version in ('baseline','candidate'):
    for order in ('forward','reverse'):
        raw=(directory/f'forward64-{version}-checked-{order}-20261012.log').read_bytes()
        text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
        for line in text.splitlines():
            if not line.startswith('{'):continue
            row=json.loads(line)
            if row['operation']!=0:continue
            key=(row['tensor'],row['type'],row['tokens'])
            groups.setdefault(key,{}).setdefault(version,[]).extend(row['ms'])
result=[]
for (tensor,dtype,tokens),versions in sorted(groups.items()):
    row={'tensor':tensor,'type':dtype,'tokens':tokens}
    for version,values in versions.items():
        assert len(values)==10
        row[version]={'mean_ms':statistics.mean(values),'sample_stdev_ms':statistics.stdev(values)}
    row['throughput_change_percent']=(row['baseline']['mean_ms']/row['candidate']['mean_ms']-1)*100
    result.append(row)
(directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
