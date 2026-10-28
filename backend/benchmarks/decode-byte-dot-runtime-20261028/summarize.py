from pathlib import Path
import json,math,hashlib,subprocess
p=Path(__file__).resolve().parent;r=p.parents[2];d=p/'reference-clean'
env=json.loads((d/'environment.json').read_text());meta=json.loads((d/'metadata.json').read_text())
assert len(meta['runs'])==len(env['activities'])==12
assert all(a['passed'] for a in env['activities']) and all(a['exit_code']==0 for a in meta['runs'])
summary={'convention':'Fyodor throughput / llama.cpp throughput * 100; 100 is parity','policy':'Default-off aligned Q4 projection packing; both Fyodor variants include accepted gate packing and selective FDPP','comparisons':[]}
for prefix in (0,1024):
 row={'prefix':prefix}
 for engine in ('baseline','fixed','llama'):
  samples=[json.loads((d/f'{prefix}-{engine}-{phase}.json').read_text()) for phase in ('a','b')]
  n=sum(s['repetitions'] for s in samples);mean=sum(s['repetitions']*s['results'][0]['tokens_per_second'] for s in samples)/n
  variance=sum((s['repetitions']-1)*s['results'][0]['stddev']**2+s['repetitions']*(s['results'][0]['tokens_per_second']-mean)**2 for s in samples)/(n-1)
  row[engine]={'samples':n,'tokens_per_second':mean,'sample_stdev':math.sqrt(variance)}
 row['baseline_llama_percent']=100*row['baseline']['tokens_per_second']/row['llama']['tokens_per_second']
 row['fixed_llama_percent']=100*row['fixed']['tokens_per_second']/row['llama']['tokens_per_second']
 row['fixed_baseline_gain_percent']=100*(row['fixed']['tokens_per_second']/row['baseline']['tokens_per_second']-1)
 summary['comparisons'].append(row)
(p/'reference-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files=['CUDA/cuda.c','CUDA/resident.inc','CUDA/CMakeLists.txt','CUDA/q4_byte_pack.h','optcpp/matvec/byte_scale_dot.cu']
manifest={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in files}
(p/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
