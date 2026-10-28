from pathlib import Path
import subprocess,os,json
r=Path.cwd();p=r/'backend/benchmarks/decode-byte-dot-runtime-20261028'
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CUDA_BYTE_DOT='1',NYA_CUDA_BYTE_GATE='0',NYA_CUDA_DEBUG='1',NYA_TEST_CPU_OPTIMIZED='1',NYA_COMPUTE='cuda',NYA_CPU_THREADS='6',NYA_CUDA_BLAS_LIBRARY=str(r/'.tools/llama-b10809-cuda/cublas64_13.dll'))
env['PATH']=str(r/'.tools/llama-b10809-cuda')+';'+env['PATH']
cmd=[str(r/'build-cuda/nya-test-inference.exe'),str(r/'.tools/tinyllama-q4_k_m.gguf'),'resident','129']
with (p/'projection-only.log').open('wb') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
(p/'projection-only.json').write_text(json.dumps({'command':cmd,'exit_code':result.returncode,'environment':{k:v for k,v in env.items() if k.startswith('NYA_')}},indent=2))
assert result.returncode==0
print('Projection-only CPU parity passed')
