"""Validate environmental gates and pool all samples; never select best runs."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('inference',type=Path)
parser.add_argument('--training',type=Path)
parser.add_argument('--output',type=Path,help='Write a new scorecard without replacing historical evidence')
args=parser.parse_args();directory=Path(__file__).resolve().parent
meta=json.loads((args.inference/'metadata.json').read_text())
assert len(meta['runs'])==8
groups={}
for run in meta['runs']:
    assert run['exit_code']==0 and 'gpu_idle_after' in run and not run.get('contaminated') and not run.get('idle_failure')
    samples=json.loads((args.inference/(run['name']+'-resources.json')).read_text())
    assert samples and all(s.get('whole_gpu') and 'competing_compute_processes' in s and not s['competing_compute_processes'] for s in samples)
    mode,prefix,engine,order=run['name'].split('-')
    data=json.loads((args.inference/(run['name']+'.json')).read_text())
    assert data['kv_type']=='f32' and data['warmup']==2 and data['repetitions']==5
    for result in data['results']:
        groups.setdefault((int(prefix),engine,result['test']),[]).append((5,result['tokens_per_second'],result['stddev']))
pooled={}
for key,parts in groups.items():
    assert len(parts)==2
    count=sum(p[0] for p in parts);mean=sum(p[0]*p[1] for p in parts)/count
    variance=sum((p[0]-1)*p[2]**2+p[0]*(p[1]-mean)**2 for p in parts)/(count-1)
    pooled[key]={'samples':count,'tokens_per_second':mean,'sample_stdev':math.sqrt(variance)}
record={'convention':'Fyodor throughput / reference throughput * 100; 100 is parity',
        'measurement':'Monitored interactive desktop, not headless; no observed competing GPU processes; OS/development rendering remains',
        'hardware':meta['hardware'],'inference_reference':'llama.cpp b10809, F32 KV, FA off; pinned comparable reference, not latest/fastest configuration',
        'inference':[]}
for label,prefix,test in [('prefill',0,'pp512'),('short_decode',0,'tg128'),('long_decode',1024,'tg128')]:
    fyodor=pooled[(prefix,'after',test)];reference=pooled[(prefix,'llama',test)]
    record['inference'].append({'workload':label,'prefix':prefix,'fyodor':fyodor,'reference':reference,
                                'percentage':fyodor['tokens_per_second']/reference['tokens_per_second']*100})
if args.training:
    data=json.loads((args.training/'results.json').read_text())
    environment=json.loads((args.training/'environment.json').read_text())
    assert 'failure' not in environment and len(environment['activities'])==9
    assert all(a['passed'] for a in environment['activities'])
    assert len(data['pairs'])==4 and all(p['correctness_passed'] for p in data['pairs'])
    native=[p['native_mean_ms'] for p in data['pairs']]
    reference=[p['torch_mean_ms'] for p in data['pairs']]
    record['training']={'workload':'CUDA width64 FF176 layers2 context32 full-rank decoder, eager F32 PyTorch with double RMS/RoPE/clipping and fused AdamW',
                        'torch_version':data['torch_version'],'process_repetitions':4,
                        'native_ms_mean':statistics.mean(native),'native_ms_sample_stdev':statistics.stdev(native),
                        'torch_ms_mean':statistics.mean(reference),'torch_ms_sample_stdev':statistics.stdev(reference),
                        'percentage':statistics.mean(reference)/statistics.mean(native)*100,
                        'tolerance':data['tolerance'],'correctness_passed':True,
                        'limitation':'Small eager workload only; no claim about compiled PyTorch, larger models or LoRA parity'}
(args.output or directory/'scorecard.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
