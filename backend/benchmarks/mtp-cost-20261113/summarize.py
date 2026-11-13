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

baseline=collect(root/'b512-baseline');candidate=collect(root/'b512-candidate')
comparisons={}
for context in baseline:
    assert baseline[context]['reference_text']==candidate[context]['reference_text'], 'Cross-revision greedy output changed'
    comparisons[context]={mode:100*baseline[context]['cells'][mode]['mean_decode_ms']/candidate[context]['cells'][mode]['mean_decode_ms'] for mode in modes}
summary={'accepted_llama_score':False,'clock_note':'Both B512 standalone probe variants use QPC. Incomplete older B8 coarse-clock diagnostics are excluded.','contract':'32 output tokens including initial prefill-logit sample; decode-loop wall time; first round excluded',
         'baseline':baseline,'candidate':candidate,'candidate_speed_percent_of_baseline':comparisons}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
