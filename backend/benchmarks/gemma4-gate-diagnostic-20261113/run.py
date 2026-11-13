"""Contaminated-desktop diagnostic only; NEVER an accepted performance score."""
from pathlib import Path
import os,sys,subprocess,json,hashlib,threading,time
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False,parents=True)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
exe=root/'build-cuda/fyodor-bench.exe'
model=Path('E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf')
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((root/'backend/benchmarks/gemma4-gate-20261113/model-oracle-exit.json').read_text())['environment'])
env.update(NYA_CUDA_BLAS='1',NYA_CUDA_GEMMA_GATE_PARTIAL='1')
meta={'accepted':False,'reason':'Diagnostic only: desktop GPU competitors are present; no performance acceptance gates bypassed or changed.', 'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'binary_sha256':sha(exe),'model_sha256':sha(model),'monitor_sha256':sha(Path(environment.__file__)),'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'runs':[]}
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
for prefix in (0,1024):
 for phase,order in [('a',('baseline','fused')),('b',('fused','baseline'))]:
  for variant in order:
   name=f'{prefix}-{variant}-{phase}';active=dict(env);active['NYA_CUDA_GEMMA_GATE']='1' if variant=='fused' else '0'
   samples=[];stop=threading.Event()
   def monitor():
    while not stop.is_set():
     try:samples.append(environment.snapshot())
     except Exception as e:samples.append({'error':str(e)})
     stop.wait(1)
   before=environment.snapshot()
   cmd=[str(exe),'-m',str(model),'-b','cuda','-p','0','-n','32','--context',str(prefix),'-r','3','--warmup','1','--json']
   thread=threading.Thread(target=monitor);thread.start()
   try:p=subprocess.run(cmd,env=active,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
   finally:stop.set();thread.join()
   (out/(name+'.json')).write_bytes(p.stdout);(out/(name+'.log')).write_bytes(p.stderr)
   (out/(name+'-environment.json')).write_text(json.dumps({'accepted':False,'before':before,'samples':samples,'after':environment.snapshot()},indent=2)+'\n')
   meta['runs'].append({'name':name,'command':cmd,'gate':active['NYA_CUDA_GEMMA_GATE'],'exit_code':p.returncode})
   (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
   assert p.returncode==0
   data=json.loads(p.stdout);assert data['kv_type']=='f32' and data['execution']=='resident'
   print(name,data['results'][0]['tokens_per_second'],flush=True)
