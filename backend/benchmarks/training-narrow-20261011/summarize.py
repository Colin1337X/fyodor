from pathlib import Path
import json,statistics
directory=Path(__file__).resolve().parent
record=json.loads((directory/'results.json').read_text())
assert record['all_outputs_exact']
summary=[]
for context in (8,32,128):
    row={'context':context}
    for version in ('before','after'):
        runs=[r for r in record['runs'] if r['context']==context and r['version']==version]
        values=[r['mean_step_ms'] for r in runs]
        assert len(values)==2
        row[version]={'step_ms_mean':statistics.mean(values),'step_ms_sample_stdev':statistics.stdev(values),
                      'tokens_per_second':context*1000/statistics.mean(values)}
    row['throughput_improvement_percent']=(row['before']['step_ms_mean']/row['after']['step_ms_mean']-1)*100
    summary.append(row)
(directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
