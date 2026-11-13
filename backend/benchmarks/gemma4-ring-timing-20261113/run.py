"""Post-cleanup diagnostic; an unchanged environment gate rejected acceptance."""
from pathlib import Path
import os,sys,subprocess,json,hashlib,threading,time,re
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
out=Path(__file__).resolve().parent/'raw';out.mkdir(exist_ok=False)
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
exe=root/'build-cuda/fyodor-bench.exe'
llama=root/'.tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe'
model=Path('E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf')
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((root/'backend/benchmarks/gemma4-gate-20261113/model-oracle-exit.json').read_text())['environment'])
env.update(NYA_CUDA_BLAS='1',NYA_CUDA_GEMMA_GATE='1',NYA_CUDA_GEMMA_GATE_PARTIAL='1')
meta={'accepted':False,'reason':'Acceptance gate rejected remaining desktop GPU identities after stopping verified game-launcher/music processes. Diagnostic only.',
'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
'binary_sha256':{'fyodor':sha(exe),'llama':sha(llama)},'model_sha256':sha(model),
'monitor_sha256':sha(Path(environment.__file__)),
'dependency_sha256':{p.name:sha(p) for p in [Path(env['NYA_CUDA_BLAS_LIBRARY']),*(llama.parent/n for n in ('llama.dll','ggml-base.dll','ggml-cuda.dll'))]},
'llama_commit':'5266f24da75dc449bd56cbed7addb9c8e4a6a73e',
'contract':'F32 KV; same deterministic tokens; host logits each token; llama FA off; full GPU; 32 decode tokens, 3 repetitions, 1 warmup, serial reversed orders.',
'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'runs':[]}
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
for prefix in (0,8192):
 for phase,order in [('a',('linear','ring','llama')),('b',('llama','ring','linear'))]:
  for variant in order:
   name=f'{prefix}-{variant}-{phase}';active=dict(env);active['NYA_CUDA_RING_KV']='1' if variant=='ring' else '0'
   samples=[];stop=threading.Event()
   def monitor():
    while not stop.is_set():
     try:samples.append(environment.snapshot())
     except Exception as e:samples.append({'error':str(e)})
     stop.wait(1)
   before=environment.snapshot()
   cmd=[str(llama if variant=='llama' else exe),'-m',str(model),'-b','cuda','-p','0','-n','32','--context',str(prefix),'-r','3','--warmup','1','--json']
   if variant=='llama':cmd+=['--batch','512','--threads','6']
   print('starting',name,flush=True)
   thread=threading.Thread(target=monitor);thread.start()
   try:p=subprocess.run(cmd,env=active,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
   finally:stop.set();thread.join()
   (out/(name+'.json')).write_bytes(p.stdout);(out/(name+'.log')).write_bytes(p.stderr)
   (out/(name+'-environment.json')).write_text(json.dumps({'accepted':False,'before':before,'samples':samples,'after':environment.snapshot()},indent=2)+'\n')
   meta['runs'].append({'name':name,'command':cmd,'ring':active['NYA_CUDA_RING_KV'],'exit_code':p.returncode})
   (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
   assert p.returncode==0
   data=json.loads(p.stdout);assert data['kv_type']=='f32'
   if variant=='llama':
    offload=re.search(rb'offloaded (\d+)/(\d+) layers to GPU',p.stderr)
    assert offload and offload[1]==offload[2] and data['flash_attention']==False
   else:assert data['execution']=='resident'
   print(name,data['results'][0]['tokens_per_second'],flush=True)
