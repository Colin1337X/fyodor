from pathlib import Path
import subprocess,os,json
r=Path.cwd();d=Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CPU_THREADS='6',NYA_CUDA_DEBUG='1',NYA_CUDA_BYTE_DOT='1',NYA_CUDA_BYTE_GATE='1',NYA_CUDA_INTEGER_DOT='1',NYA_CUDA_INTEGER_Q6='1',NYA_CUDA_INTEGER_GATE='1',NYA_CUDA_FLASH_DECODING_PP='short',NYA_CUDA_BLAS_LIBRARY=str(r/'.tools/llama-b10809-cuda/cublas64_13.dll'))
cmd=[str(r/'build-cuda/fyodor-bench.exe'),'-m','E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','-b','cuda','-p','0','-n','8','--context','0','-r','1','--warmup','0','--json']
with (d/'smoke.json').open('wb') as f,(d/'smoke.log').open('wb') as g:
 try:result=subprocess.run(cmd,env=env,stdout=f,stderr=g,timeout=180);code=result.returncode
 except subprocess.TimeoutExpired:code='timeout'
(d/'smoke-command.json').write_text(json.dumps({'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'exit_code':code},indent=2));print(code,flush=True)
