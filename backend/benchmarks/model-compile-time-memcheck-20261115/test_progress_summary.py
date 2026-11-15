import hashlib,json
from pathlib import Path
from progress_summary import summarize
p=Path(__file__).resolve().parent
lines=[]
def event(name,done,total,position=0,layer=0,layers=0):return f'PROGRESS event={name} position={position} completed={done} total={total} layer={layer} layers={layers} elapsed_s=1 tokens_per_s=256\n'
lines=[event('prefill_begin',0,8192),event('batch_begin',0,8192),event('layer_begin',0,256,layer=1,layers=48)]
r=summarize(''.join(lines));assert r['active_prefill']['total_tokens']==8192 and r['active_prefill']['completed_tokens']==0
lines += [event('layer_complete',0,256,layer=48,layers=48),event('batch_complete',256,8192,position=256),event('batch_begin',256,8192,position=256),event('layer_begin',0,512,position=256,layer=1,layers=48)]
r=summarize(''.join(lines));assert r['active_prefill']['percent']==3.125 and r['active_prefill']['batch']['token_count']==512 and r['active_prefill']['batch']['layers_completed']==0
assert r['overall_percent'] is None and r['process_liveness']=='not_checked'
assert summarize(''.join(lines)+event('layer_begin',0,512,position=256,layer=2,layers=48).rstrip())==r
assert summarize(''.join(lines)+event('layer_complete',0,512,position=999,layer=2,layers=48))['malformed_records']==1
assert summarize(''.join(lines)+event('prefill_complete',256,8192))['malformed_records']==1
assert summarize(''.join(lines)+'TOKEN completed_position=1 session_capacity=8197 token_elapsed_s=nan instantaneous_tokens_per_s=1\n')['malformed_records']==1
r=summarize(''.join(lines)+event('batch_complete',8192,8192,position=8192)+event('prefill_complete',8192,8192)+'TOKEN completed_position=1 session_capacity=8197 token_elapsed_s=0.5 instantaneous_tokens_per_s=2\n')
assert r['active_prefill'] is None and r['prefills_completed']==1 and r['last_sampled_token']['session_capacity_is_not_work_total']
replays=[]
for label in ('CheckC17','CheckC23','CheckC23Prefix257'):
 data=(p/label/'stderr').read_text();r=summarize(data)
 assert r['prefills_started']==r['prefills_completed']==3 and r['active_prefill'] is None and r['malformed_records']==0,(label,r)
 replays.append(label)
result=dict(passed=True,completed_log_replays=replays,sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),p/'progress_summary.py',*[p/label/'stderr' for label in replays]]})
with (p/'progress-summary-tests.json').open('x') as f:json.dump(result,f,indent=2)
print('PASS: nested batch/prefill scope, partial writes, invalid records, sampled tokens and three completed real logs')
