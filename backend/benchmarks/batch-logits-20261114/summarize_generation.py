"""Summarize only complete, successful diagnostic runs; never promote scores."""
import json,re,statistics,sys
from pathlib import Path
root=Path(__file__).resolve().parent
modes=('ordinary','mtp-serial2','mtp-batch2','mtp-batch4')

def collect(folder):
    metadata=json.loads((folder/'timing.json').read_text())
    assert len(metadata['runs'])==2 and all(r.get('exit_code')==0 for r in metadata['runs']), 'Timing run incomplete or failed'
    result={}
    for context in ('short','long'):
        stdout=(folder/(context+'.stdout')).read_text()
        rows=[json.loads(line) for line in stdout.splitlines() if line.startswith('{"round":')]
        assert len(rows)==12 and all(row['parity'] and row['generated']==32 for row in rows), 'Incomplete output or parity failure'
        keyed={(row['round'],row['mode']):row for row in rows}
        assert len(keyed)==12 and set(keyed)=={(r,m) for r in range(3) for m in modes}
        case=None; timed=set()
        for line in (folder/(context+'.stderr')).read_text().splitlines():
            match=re.fullmatch(r'CASE round=(\d+) mode=(\S+)',line)
            if match:case=(int(match[1]),match[2])
            match=re.fullmatch(r'Generation profile: mode=(\S+) prompt=(\d+) generated=(\d+) decode_loop_ms=([0-9]+(?:\.[0-9]+)?)',line)
            if match:
                assert case in keyed and case not in timed
                row=keyed[case]
                assert int(match[2])==row['prompt'] and int(match[3])==row['generated']
                row['decode_ms']=float(match[4]);timed.add(case)
            match=re.fullmatch(r'Generation assistant profile: calls=(\d+) forward_ms=([0-9]+(?:\.[0-9]+)?)',line)
            if match:
                assert case in timed
                keyed[case].update(assistant_calls=int(match[1]),assistant_ms=float(match[2]))
        assert len(timed)==12
        text=stdout.split('TEXT ',1)[1].split('\n{"round":',1)[0]
        cells={}
        for mode in modes:
            observations=[keyed[(r,mode)] for r in (1,2)]
            ms=[row['decode_ms'] for row in observations]
            cells[mode]={'observations':observations,'mean_decode_ms':statistics.mean(ms),'stdev_decode_ms':statistics.stdev(ms),
                         'mean_call_seconds':statistics.mean(row['seconds'] for row in observations)}
        result[context]={'cells':cells,'reference_text':text}
    return result

off_meta=json.loads((root/'runtime-off/timing.json').read_text())
on_meta=json.loads((root/'runtime-on/timing.json').read_text())
assert off_meta['binary_sha256']==on_meta['binary_sha256'], 'Binary changed between runs'
off_env=off_meta['environment'].copy();on_env=on_meta['environment'].copy()
assert off_env.pop('NYA_CUDA_BATCH_LOGITS')=='0'
assert on_env.pop('NYA_CUDA_BATCH_LOGITS')=='1'
assert off_env==on_env, 'Unmatched environment'
for label in ('runtime-off','runtime-on'):
    for context in ('short','long'):
        trace=(root/label/(context+'.stderr')).read_text()
        assert 'CUDA kernel GPU milliseconds:' not in trace, 'Event profiling invalidates this timing contract'
        calls=[int(n) for n in re.findall(r'CUDA batched output projection calls: (\d+)',trace)]
        if label=='runtime-on':
            assert len(calls)==6 and all(n>0 for n in calls), 'Missing batched-output dispatch'
        else:
            assert not calls, 'Unexpected batched-output dispatch in control'
baseline=collect(root/'runtime-off');candidate=collect(root/'runtime-on')
comparisons={}
for context in baseline:
    assert baseline[context]['reference_text']==candidate[context]['reference_text'], 'Cross-revision greedy output changed'
    comparisons[context]={mode:100*baseline[context]['cells'][mode]['mean_decode_ms']/candidate[context]['cells'][mode]['mean_decode_ms'] for mode in modes}
summary={'accepted_llama_score':False,'clock_note':'Same binary, QPC, B512, output projection batching off/on. Weight reuse, target wide batches, borrowed wide attention and greedy early rejection enabled in both.','contract':'32 output tokens including initial prefill-logit sample; decode-loop wall time; first round excluded',
         'baseline':baseline,'candidate':candidate,'candidate_speed_percent_of_baseline':comparisons}
summary['candidate_dispatch_contexts']=['short','long']
summary['idle_gate_passed']={'baseline':off_meta['gate_passed'],'candidate':on_meta['gate_passed']}
summary['environment_note']='Idle gates and one-second resource/process samples are diagnostic evidence, not continuous clean-run acceptance. accepted_llama_score remains false regardless of gate flags.'
(root/'runtime-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
