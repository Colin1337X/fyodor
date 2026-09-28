"""Summarize resources without conflating whole-device and tensor allocations."""
import json
from pathlib import Path

out = Path(__file__).resolve().parent
directory = out/'accepted-remote'
meta = json.loads((directory/'metadata.json').read_text())
assert len(meta['runs']) == 20 and all('gpu_idle_after' in r for r in meta['runs'])
groups = {}
for run in meta['runs']:
    mode,prefix,engine,phase = run['name'].split('-')
    key = (mode,int(prefix),engine)
    group = groups.setdefault(key,{'samples':[],'results':[]})
    group['samples'] += json.loads((directory/(run['name']+'-resources.json')).read_text())
    group['results'].append(json.loads((directory/(run['name']+'.json')).read_text()))
rows = []
for (mode,prefix,engine),group in sorted(groups.items()):
    samples = group['samples']
    data = group['results']
    rows.append({'mode':mode,'prefix':prefix,'engine':engine,
        'maximum_observed_peak_working_set_bytes':max(s.get('peak_working_set',0) for s in samples),
        'maximum_sampled_private_commit_bytes':max(s.get('private_bytes',0) for s in samples),
        'maximum_sampled_whole_gpu_mib':max(int(s['whole_gpu'].split(',')[1].split()[0]) for s in samples if s.get('whole_gpu')),
        'explicit_device_memory_by_order':[d.get('device_memory_after') for d in data],
        'measured_dispatch_counters_by_order':[[{k:r[k] for k in ('test','kernel_launches','uploads','downloads','synchronizations','graph_captures','graph_replays') if k in r} for r in d['results']] for d in data]})
(out/'resources.json').write_text(json.dumps(rows,indent=2))
print('Recorded',len(rows),'engine/workload resource groups.')
