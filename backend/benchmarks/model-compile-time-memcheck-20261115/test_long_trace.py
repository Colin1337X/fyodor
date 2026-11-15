"""Adversarial trace inventory checks; no CUDA context is initialized."""
import hashlib,json
from pathlib import Path
from long_trace import validate
p=Path(__file__).resolve().parent
stdout='reset=0 prefix=8192 tokens=4 max_scaled=0.00001 execution=resident\nreset=1 prefix=8192 tokens=4 max_scaled=0.00001 execution=resident\n'
lines=['CUDA Q4 integer batch-four calls: 1','CUDA QKV input reuse capture/direct hits: 1','CUDA paired rows capture/direct calls: 1','CUDA Q6 paired rows capture/direct calls: 1','CUDA cached query capture/direct calls: 8','CUDA batched graph captures: 1 replays: 2','CUDA native metadata Q4: selected','SANITIZER_MODULE compile_time_memcheck=1 exact_ptx_match=1 bytes=11261064','========= ERROR SUMMARY: 0 errors']
stderr='\n'.join(lines);cases=[]
def check(name,a,b,expected):
 actual=validate(a,b)['passed'];assert actual==expected,(name,actual);cases.append(dict(name=name,passed=True))
check('valid',stdout,stderr,True)
for i,line in enumerate(lines):
 check('missing_'+str(i),stdout,'\n'.join(lines[:i]+lines[i+1:]),False)
 check('duplicate_'+str(i),stdout,stderr+'\n'+line,False)
 check('malformed_extra_'+str(i),stdout,stderr+'\n'+line+' garbage',False)
for prefix in ('5','257','8191','8193'):check('wrong_prefix_'+prefix,stdout.replace('8192',prefix),stderr,False)
for value in ('-1','0.00050001','1e999','nan','1.2.3'):check('invalid_numeric_'+value,stdout.replace('0.00001',value),stderr,False)
for index in range(5):
 replacement=lines[index].rsplit(' ',1)[0]+' 0';check('zero_counter_'+str(index),stdout,stderr.replace(lines[index],replacement),False)
check('graph_no_replay',stdout,stderr.replace('replays: 2','replays: 1'),False)
check('reordered_resets','\n'.join(reversed(stdout.strip().splitlines())),stderr,False)
check('extra_numeric_row',stdout+stdout.splitlines()[0],stderr,False)
for label in ('CheckC17','CheckC23'):
 check('real_short_rejected_'+label,(p/label/'stdout').read_text(),(p/label/'stderr').read_text(),False)
result=dict(passed=True,cases=cases,sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),p/'long_trace.py')})
with (p/'long-trace-tests.json').open('x') as f:json.dump(result,f,indent=2)
print('PASS',len(cases),'strict trace cases')
