from pathlib import Path
import os,subprocess,json
r=Path(__file__).resolve().parents[3];b=Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cuda',NYA_CUDA_CUTLASS='0',NYA_CUDA_BLAS_LIBRARY=str(r/'.tools/llama-b10809-cuda/cublas64_13.dll'),NYA_CPU_THREADS='6')
results={}
def run(name,cmd,e=env):
 with (b/(name+'.log')).open('wb') as f:p=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=240)
 results[name]=p.returncode;(b/'validation-exits.json').write_text(json.dumps(results,indent=2));print(name,p.returncode,flush=True);assert p.returncode==0
for name,settings in [('cpu-oracle',{'NYA_TEST_CPU_OPTIMIZED':'1','NYA_TEST_DECODE':'1'}),('exact-graph',{'NYA_TEST_GATE_UP':'1','NYA_TEST_GRAPH':'1'})]:
 run(name,[str(r/'build-cuda/nya-test-inference.exe'),str(r/'.tools/tinyllama-q4_k_m.gguf'),'resident','273'],{**env,**settings})
san=r/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
for tool in ('memcheck','initcheck','racecheck','synccheck'):
 for script in ('check_kernel',):
  run(tool+'-'+script,[str(san),'--tool',tool,'--target-processes','application-only','--error-exitcode','99','C:/Users/Colin/AppData/Local/Python/pythoncore-3.14-64/python.exe',str(b/(script+'.py'))])
