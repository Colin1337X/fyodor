"""Serial fixed/integer scale-product comparison; retain every timed observation."""
from pathlib import Path
import os,subprocess,hashlib,json,re,sys
import environment
root=Path(__file__).resolve().parents[3];out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=False);environment.configure(out)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(NYA_CPU_THREADS='6',NYA_CUDA_BATCH='512',NYA_CUDA_BLAS='1',NYA_CUDA_CUTLASS='0',
           NYA_CUDA_BLAS_LIBRARY=str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'))
model=Path('E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf')
exe={'baseline':root/'build-cuda/fyodor-bench.exe','fixed':root/'build-cuda/fyodor-bench.exe',
     'llama':root/'.tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe'}
meta={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
      'model_sha256':sha(model),'binary_sha256':{k:sha(v) for k,v in exe.items()},
      'llama_commit':'5266f24da75dc449bd56cbed7addb9c8e4a6a73e',
      'dependency_sha256':{p.name:sha(p) for p in [Path(env['NYA_CUDA_BLAS_LIBRARY']),
       *(exe['llama'].parent/n for n in ('llama.dll','ggml-base.dll','ggml-cuda.dll'))]},
      'monitor_sha256':sha(Path(environment.__file__)),
      'contract':'F32 KV; full GPU; llama FA off; identical deterministic tokens; host logits each token; serial reverse orders; interactive desktop gates',
      'runs':[]}
completed=set()
if len(sys.argv)>2:
 previous=Path(sys.argv[2]);prior=json.loads((previous/'metadata.json').read_text())
 gates=json.loads((previous/'environment.json').read_text())
 for key in ('binary_sha256','model_sha256','dependency_sha256','monitor_sha256'):
  assert meta[key]==prior[key],key
 for run,gate in zip(prior['runs'],gates['activities']):
  assert run['exit_code']==0 and gate['passed']
  completed.add(run['name'])
 meta['continuation_of']=str(previous)
@environment.capture('decode')
def run(cmd):
 p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 environment.set_active_pid(p.pid)
 try:stdout,stderr=p.communicate(timeout=240)
 except BaseException:p.kill();p.communicate();raise
 return p.returncode,stdout,stderr
for prefix in (4096,8192):
 for phase,order in (('a',('baseline','fixed','llama')),('b',('llama','fixed','baseline'))):
  for engine in order:
   name=f'{prefix}-{engine}-{phase}'
   if name in completed:continue
   env['NYA_CUDA_FLASH_DECODING_PP']='short'
   env['NYA_CUDA_FIXED_GATE']='1'
   env['NYA_CUDA_INTEGER_GATE']='1'
   env['NYA_CUDA_INTEGER_DOT']='1'
   env['NYA_CUDA_INTEGER_Q6']='1'
   env['NYA_CUDA_BYTE_GATE']='1'
   env['NYA_CUDA_BYTE_DOT']='1'
   env['NYA_CUDA_WIDE_SPLIT']='1' if engine=='fixed' else '0'
   cmd=[str(exe[engine]),'-m',str(model),'-b','cuda','-p','0','-n','32','--context',str(prefix),'-r','3','--warmup','1','--json']
   if engine=='llama':cmd+=['--batch','512','--threads','6']
   code,stdout,stderr=run(cmd)
   (out/(name+'.json')).write_bytes(stdout);(out/(name+'.log')).write_bytes(stderr)
   meta['runs'].append({'name':name,'command':cmd,'fdpp':env['NYA_CUDA_FLASH_DECODING_PP'],'fixed':env['NYA_CUDA_FIXED_GATE'],'integer':env['NYA_CUDA_INTEGER_GATE'],'integer_dot':env['NYA_CUDA_INTEGER_DOT'],'integer_q6':env['NYA_CUDA_INTEGER_Q6'],'byte_gate':env['NYA_CUDA_BYTE_GATE'],'byte_dot':env['NYA_CUDA_BYTE_DOT'],'wide_split':env['NYA_CUDA_WIDE_SPLIT'],'exit_code':code})
   (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');assert code==0
   data=json.loads(stdout);assert data['kv_type']=='f32'
   if engine=='llama':
    offload=re.search(rb'offloaded (\d+)/(\d+) layers to GPU',stderr)
    assert offload and offload[1]==offload[2] and data['flash_attention']==False
   else:assert data['execution']=='resident'
   print(name,data['results'][0]['tokens_per_second'],flush=True)
