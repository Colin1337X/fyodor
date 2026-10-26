from pathlib import Path
import os,subprocess,json,sys
root=Path(__file__).resolve().parents[3];out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
out.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env['PATH']=str(root/'.tools/w64devkit/bin')+';'+env['PATH']
env['NYA_CUDA_CUTLASS_LIBRARY']=str(root/'build-cutlass/fyodor-cutlass.dll')
env['NYA_CUDA_BLAS_LIBRARY']=str(root/'.tools/llama-b10809-cuda/cublas64_13.dll')
env['NYA_CUDA_FLASH_DECODING_PP']='short'
env['NYA_CUDA_PINNED_LOGITS']='1'
results={}
for build in ('build-cuda','build-cuda-c23'):
 for label,cmd in [('build',[str(root/'.tools/w64devkit/bin/cmake.exe'),'--build',build,'-j','8']),('tests',[str(root/'.tools/w64devkit/bin/ctest.exe'),'--test-dir',build,'--output-on-failure','-R','inference-cuda-fdpp-short-|inference-cuda-prefix-replay|inference-cuda-without-graphs|inference-cuda-blas-unavailable'])]:
  name=build+'-'+label
  with (out/(name+'.log')).open('wb') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  results[name]=p.returncode;(out/'suite-exits.json').write_text(json.dumps(results,indent=2));print(name,p.returncode,flush=True);assert p.returncode==0
