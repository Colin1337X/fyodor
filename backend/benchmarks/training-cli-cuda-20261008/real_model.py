"""Real TinyLlama public CLI parity and DPO smoke check; no performance claim."""
from pathlib import Path
import csv,hashlib,json,math,struct,subprocess,time
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
work=Path('D:/fyodor-validation/resident-training-cli-20261008/real-model')
work.mkdir(parents=True,exist_ok=False)
trainer=root/'build-cuda/fyodor-train.exe';base=root/'.tools/tinyllama-q4_k_m.gguf'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
record={'trainer_sha256':sha(trainer),'base_sha256':sha(base),'performance_acceptance':False,'runs':[]}
(work/'corpus.txt').write_text('The small cat sat on the mat. '*3,encoding='utf-8')
(work/'pairs.tsv').write_text('The cat\t sat.\t flew.\n',encoding='utf-8')
def save(): (out/'real-model.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
for compute,mode,steps in [('cpu','cpt',2),('cuda','cpt',2),('cuda','dpo',1)]:
    name=compute+'-'+mode
    command=[str(trainer),'--compute',compute,'--mode',mode,'--base',str(base),'--rank','2','--context','8','--threads','6','--memory-mib','512','--device-memory-mib','3072','--steps',str(steps),'--data',str(work/('pairs.tsv' if mode=='dpo' else 'corpus.txt')),'--output',str(work/(name+'.gguf')),'--checkpoint',str(work/(name+'.ckpt')),'--metrics',str(work/(name+'.csv'))]
    started=time.time()
    with (out/(name+'.log')).open('xb') as log:p=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    run={'command':command,'exit_code':p.returncode,'elapsed_seconds':time.time()-started};record['runs'].append(run);save()
    assert p.returncode==0,run
    run['export_bytes']=(work/(name+'.gguf')).stat().st_size
    run['export_sha256']=sha(work/(name+'.gguf'));run['checkpoint_sha256']=sha(work/(name+'.ckpt'))
    with (work/(name+'.csv')).open(newline='') as stream:run['metrics']=list(csv.DictReader(stream))
    assert len(run['metrics'])==steps
    assert all(math.isfinite(float(row['loss'])) and row['compute']==compute for row in run['metrics'])
    save();print(name,'passed',flush=True)
a=(work/'cpu-cpt.ckpt').read_bytes()[16:];b=(work/'cuda-cpt.ckpt').read_bytes()[16:]
assert a[:48]==b[:48]
count=struct.unpack_from('<Q',a,8)[0];offset=48;maximum=0
for parameter in range(count):
    rows,columns=struct.unpack_from('<QQ',a,offset);assert a[offset:offset+16]==b[offset:offset+16];offset+=16
    for state in range(4):
        for index in range(rows*columns):
            expected=struct.unpack_from('<f',a,offset)[0];actual=struct.unpack_from('<f',b,offset)[0];offset+=4
            error=abs(expected-actual)/(1+abs(expected));assert math.isfinite(actual) and error<=1e-5,(parameter,state,index,error)
            maximum=max(maximum,error)
assert offset+8==len(a)==len(b)
record['maximum_cpt_state_scaled_error']=maximum;record['passed']=True;save()
print('Real-model public training passed; maximum scaled state error',maximum,flush=True)
