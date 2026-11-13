"""Summarize profiling categories without treating graph-disabled time as throughput."""
import json,re
from pathlib import Path

root=Path(__file__).resolve().parent
result={}
for context in ('short','long'):
    modes={};current=None
    for line in (root/(context+'.stderr')).read_text().splitlines():
        case=re.fullmatch(r'CASE round=0 mode=(\S+)',line)
        if case:
            current={'matrix_ms':{},'target_attention_ms':0.0}
            modes[case[1]]=current
        if line.startswith('CUDA kernel GPU milliseconds:'):
            current['gpu_ms']={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
        matrix=re.fullmatch(r'CUDA matrix profile: path=(\S+) type=(\d+) rows=(\d+) columns=(\d+) batch=(\d+) calls=(\d+) milliseconds=([\d.]+)',line)
        if matrix:
            key=matrix[1]+'/batch'+matrix[5]
            current['matrix_ms'][key]=current['matrix_ms'].get(key,0)+float(matrix[7])
        attention=re.search(r'CUDA attention profile: .* milliseconds=([\d.]+)',line)
        if attention:current['target_attention_ms']+=float(attention[1])
    assert len(modes)==4 and all('gpu_ms' in row for row in modes.values())
    result[context]=modes
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
