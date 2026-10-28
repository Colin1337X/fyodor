from pathlib import Path
import subprocess,os,json
r=Path.cwd();d=r/'backend/benchmarks/decode-q6-runtime-20261028';env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CUDA_INTEGER_Q6='1',NYA_CUDA_INTEGER_DOT='1',NYA_CUDA_INTEGER_GATE='1',NYA_CUDA_FIXED_GATE='1',NYA_COMPUTE='cuda',NYA_TEST_CPU_OPTIMIZED='1',NYA_CUDA_FLASH_DECODING_PP='short',NYA_CPU_THREADS='6',NYA_CUDA_BLAS='1',NYA_CUDA_BLAS_LIBRARY=str(r/'.tools/llama-b10809-cuda/cublas64_13.dll'))
env['PATH']=str(r/'.tools/llama-b10809-cuda')+';'+env['PATH']
results={}
for capacity in (129,273):
 cmd=[str(r/'build-cuda/nya-test-inference.exe'),str(r/'.tools/tinyllama-q4_k_m.gguf'),'resident',str(capacity)]
 with (d/f'cpu-oracle-{capacity}.log').open('wb') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 results[str(capacity)]={'command':cmd,'exit_code':p.returncode};(d/'oracle-exits.json').write_text(json.dumps(results,indent=2));print(capacity,p.returncode,flush=True);assert p.returncode==0
