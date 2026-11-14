"""Offline clock/engine diagnosis; never certifies coverage or performance."""
import hashlib,json,re,struct,sys
from pathlib import Path

def summarize(rows,luid):
    if not re.fullmatch(r'luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}',luid):
        raise ValueError('Invalid adapter identity')
    groups={}; previous=None
    for row in rows:
        if previous is not None:
            old={x['instance']:x for x in previous.get('raw',{}).get('rows') or []}
            formatted={x['instance']:x for x in row.get('formatted',{}).get('rows') or []}
            for now in row.get('raw',{}).get('rows') or []:
                name=now['instance']; match=re.fullmatch(r'pid_(\d+)_'+luid+r'_(.+)',name)
                before=old.get(name); value=formatted.get(name)
                if not match or not before or not value or any(x['status'] not in (0,1) for x in (now,before,value)):continue
                delta=now['second']-before['second']
                if delta<=0:continue
                qpc=row['qpc_start']-previous['qpc_start']
                if qpc<=0:raise ValueError('Nonmonotonic collection clock')
                percent=struct.unpack('<d',bytes.fromhex(value['value_bits']))[0]
                g=groups.setdefault(name,dict(process_id=int(match[1]),pairs=0,above_100=0,positive=0,maximum_percent=0,maximum_timestamp_delta_difference=0,maximum_qpc_relative_difference=0))
                g['pairs']+=1;g['above_100']+=percent>100;g['positive']+=percent>0
                g['maximum_percent']=max(g['maximum_percent'],percent)
                g['maximum_timestamp_delta_difference']=max(g['maximum_timestamp_delta_difference'],abs(delta-(now['timestamp']-before['timestamp'])))
                g['maximum_qpc_relative_difference']=max(g['maximum_qpc_relative_difference'],abs(delta/1e7/qpc-1))
        previous=row
    return dict(accepted_benchmark=False,engines=groups,limitation='Clock agreement does not prove correct GPU running-time accounting or causal System attribution.')

def main():
    folder=Path(sys.argv[1]);proof_path=folder/'result.json';proof=json.loads(proof_path.read_text())
    if proof.get('inputs_unchanged') is not True or proof.get('errors'):raise ValueError('Incomplete or failed diagnostic')
    for name,h in proof['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('Changed input: '+name)
    log=folder/'samples.jsonl'
    with log.open() as f:rows=[json.loads(line) for line in f]
    if len(rows)!=proof['samples'] or [r['index'] for r in rows]!=list(range(len(rows))):raise ValueError('Incomplete sample inventory')
    result=summarize(rows,sys.argv[2])
    result['sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),proof_path,log)}
    with (folder/'clock-audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({n:g for n,g in result['engines'].items() if g['positive']},indent=2))
if __name__=='__main__':main()
