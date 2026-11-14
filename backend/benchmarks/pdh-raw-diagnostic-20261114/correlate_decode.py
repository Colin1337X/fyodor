"""Correlate completed diagnostic counters with native decode intervals; no acceptance."""
import hashlib,json,math,re,struct,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'gpu-monitor-20261114'))
from decode_intervals import parse


def correlate(samples, intervals, luid, pid):
    if not re.fullmatch(r'luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}',luid):
        raise ValueError('Invalid adapter identity')
    for index,s in enumerate(samples):
        if s['index']!=index or not all(math.isfinite(s[k]) for k in ('qpc_start','qpc_end')) or s['qpc_end']<s['qpc_start']:
            raise ValueError('Invalid sample sequence')
        if index and s['qpc_start']<=samples[index-1]['qpc_end']:
            raise ValueError('Overlapping collection calls')
    output=[]
    for interval in intervals:
        start,end=interval['start'],interval['end']
        if not all(math.isfinite(x) for x in (start,end)) or end<=start:
            raise ValueError('Invalid decode interval')
        events=[];indices=[];gaps=[]
        for previous,current in zip(samples,samples[1:]):
            # Conservative bounds: collection may occur anywhere within its call.
            a,b=previous['qpc_start'],current['qpc_end']
            if b<=start or a>=end:continue
            indices.append(current['index'])
            if b-a>1.5:gaps.append(current['index'])
            entries=current.get('formatted',{}).get('rows')
            if current.get('collect_status',0) or entries is None:
                events.append(dict(index=current['index'],kind='api_failure'));continue
            for e in entries:
                match=re.match(r'^pid_(\d+)_'+re.escape(luid)+r'_',e['instance'])
                if not match:continue
                owner=int(match[1]);value=struct.unpack('<d',bytes.fromhex(e['value_bits']))[0]
                kind=('invalid' if e['status'] not in (0,1) or not math.isfinite(value) or value<0 else
                      'above_100' if value>100 else
                      'system_positive' if owner==4 and value>0 else
                      'other_positive' if owner!=pid and value>0 else None)
                if kind:events.append(dict(index=current['index'],kind=kind,instance=e['instance'],status=e['status'],value=value if math.isfinite(value) else None))
        output.append(dict(interval=interval,sample_indices=indices,gap_indices=gaps,
                           bracketed=bool(samples and samples[0]['qpc_end']<=start and samples[-1]['qpc_start']>=end),events=events))
    return output


def main():
    if len(sys.argv)!=4:raise SystemExit('Usage: correlate_decode.py CAPTURE_FOLDER GENERATION_FOLDER LUID')
    capture,generation=map(Path,sys.argv[1:3]);hashes={}
    for folder in (capture,generation):
        path=folder/'result.json';proof=json.loads(path.read_text());hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        if folder==capture:
            if proof.get('inputs_unchanged') is not True or proof.get('errors'):raise ValueError('Invalid capture proof')
            count=proof['samples']
        else:
            if proof.get('passed') is not True or proof.get('exit_code')!=0:raise ValueError('Incomplete generation')
            pid=proof['process_id'];cases=proof['cases']
        for name,expected in proof['sha256'].items():
            source=Path(name) if Path(name).is_absolute() else ROOT/name
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Changed input: '+str(source))
    samples=[json.loads(line) for line in (capture/'samples.jsonl').read_text().splitlines()]
    if len(samples)!=count:raise ValueError('Incomplete capture')
    intervals=parse((generation/'stderr').read_text(),clock='qpc')
    if len(intervals)!=len(cases):raise ValueError('Case/interval inventory mismatch')
    for interval,case in zip(intervals,cases):
        if (interval['prompt'],interval['generated'],interval['mode'])!=(case['prompt'],case['generated'],'ordinary' if case['mode']=='ordinary' else 'mtp'):
            raise ValueError('Case/interval mismatch')
        interval.update(round=case['round'],case_mode=case['mode'])
    for path in (Path(__file__),HERE.parent/'gpu-monitor-20261114/decode_intervals.py'):
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(accepted_benchmark=False,process_id=pid,observations=correlate(samples,intervals,sys.argv[3],pid),sha256=hashes,
                limitation='Conservative collection-call overlap only; no causal System attribution or performance acceptance.')
    with (capture/('decode-overlap-'+generation.name+'.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print([(r['interval']['case_mode'],len(r['events']),r['bracketed']) for r in result['observations']])

if __name__=='__main__':main()
