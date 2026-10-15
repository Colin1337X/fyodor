"""Native public training baseline. Three repetitions, alternating CPU/CUDA order."""
from pathlib import Path
import ctypes,csv,datetime,hashlib,json,os,statistics,subprocess,threading,time
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
work=Path('D:/fyodor-validation/training-profile-20261010');work.mkdir(parents=True,exist_ok=False)
trainer=root/'build-cuda/fyodor-train.exe';base=root/'.tools/tinyllama-q4_k_m.gguf'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class Memory(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
query=ctypes.WinDLL('psapi').GetProcessMemoryInfo;query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),ctypes.c_ulong];query.restype=ctypes.c_int
record={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'trainer_sha256':sha(trainer),'model_sha256':sha(base),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':{'steps':5,'discarded_warmup_steps':2,'repetitions':3,'contexts':[8,32,128],'rank':2,'cpu_threads':6,'gpu_sampling_ms':1000,'memory_sampling_ms':200,'gpu_memory_scope':'whole device, not trainer-only','performance_acceptance':'pending environment and correctness review'},'runs':[]}
corpus=work/'corpus.txt';corpus.write_text('The small cat sat on the mat beside the quiet library. '*300,encoding='utf-8')
env=dict(os.environ,NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0')
def save(): (out/'baseline.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
save()
for repeat in range(3):
 for context in (8,32,128):
  for compute in (('cpu','cuda') if repeat%2==0 else ('cuda','cpu')):
   name=f'{repeat}-{context}-{compute}';model=work/(name+'.gguf');metrics=out/(name+'.csv');checkpoint=work/(name+'.ckpt')
   command=[str(trainer),'--mode','cpt','--base',str(base),'--rank','2','--context',str(context),'--threads','6','--compute',compute,'--memory-mib','4096','--device-memory-mib','6144','--steps','5','--data',str(corpus),'--output',str(model),'--checkpoint',str(checkpoint),'--metrics',str(metrics)]
   samples=[];start=time.perf_counter()
   with (out/(name+'.log')).open('xb') as log,(out/(name+'-gpu.csv')).open('xb') as gpu:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,nounits','-lms','1000'],stdout=gpu,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
     process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
     while process.poll() is None:
      m=Memory();m.cb=ctypes.sizeof(m)
      if query(int(process._handle),ctypes.byref(m),m.cb):samples.append({'seconds':time.perf_counter()-start,'working_bytes':m.working,'peak_working_bytes':m.peak_working,'private_bytes':m.private})
      time.sleep(.2)
     code=process.returncode
    finally:
     monitor.terminate();monitor.wait(timeout=10)
   run={'name':name,'command':command,'exit_code':code,'wall_seconds_including_startup_checkpoint_export':time.perf_counter()-start,'host_memory_samples':samples};record['runs'].append(run);save()
   assert code==0,run
   with metrics.open(newline='') as f:rows=list(csv.DictReader(f))
   assert len(rows)==5 and all(int(row['tokens'])==context for row in rows)
   run['metrics']=rows;run['checkpoint_sha256']=sha(checkpoint);run['export_sha256']=sha(model);run['export_bytes']=model.stat().st_size
   run['measured_mean']={k:statistics.mean(float(row[k]) for row in rows[2:]) for k in ('forward_ms','backward_ms','optimizer_ms','step_ms','tokens_per_second')}
   if compute=='cuda':run['measured_mean']['preparation_ms']=statistics.mean(float(row['preparation_ms']) for row in rows[2:])
   # Delete only the exact export produced by this run, after recording its hash.
   assert model.parent.resolve()==work.resolve() and model.is_file();model.unlink()
   save();print(name,run['measured_mean'],flush=True)
record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
