import datetime,hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
env={**os.environ,'PATH':'C:/Program Files/nodejs'+os.pathsep+os.environ['PATH'],
 'NYA_CPU_THREADS':'6','NYA_CUDA_BATCH':'512',
 'NYA_CUDA_BLAS_LIBRARY':str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'),
 'NYA_CUDA_CUTLASS_LIBRARY':str(root/'build-cutlass/fyodor-cutlass.dll')}
for k in ['NYA_CUDA_PROFILE','NYA_CUDA_REFERENCE','NYA_TEST_GRAPH','NYA_TEST_ATTENTION_TILES','NYA_COMPUTE']:env.pop(k,None)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def call(cmd,name,runenv=env):
    with open(root/'.tools'/('stage7-'+name+'.log'),'wb') as f:
        p=subprocess.run(cmd,cwd=root,env=runenv,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    print(name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError((name,p.returncode))
def gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
san=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
runenv={**env,'NYA_COMPUTE':'cuda','NYA_CUDA_BLAS':'0','NYA_CUDA_CUTLASS':'0','NYA_TEST_ATTENTION_TILES':'1'}
for tool in ['memcheck','synccheck']:
    # Initialization tracking must include producer kernels, not just attention
    # consumers. Otherwise correct writes are invisible and reads look undefined.
    filters=['--kernel-name','kns=nya_attention_prefill'] if tool=='racecheck' else []
    call([str(san),'--tool',tool,'--error-exitcode','99',*filters,str(root/'build-cuda/nya-test-inference.exe'),'--attention-shapes',str(root/'.tools'/('stage7-sanitizer-'+tool))],'cuda-'+tool+'-full',runenv)
# Browser checks use their own helper process and stop signal. They never attach
# to or terminate the user's unrelated Vite server.
helper=subprocess.Popen([sys.executable,str(root/'.tools/start_frontend_qa.py')],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
try:
    for _ in range(60):
        if helper.poll() is not None:raise RuntimeError('QA helper exited')
        try:
            with urllib.request.urlopen('http://127.0.0.1:5174',timeout=1) as r:
                if r.status==200:break
        except OSError:time.sleep(1)
    else:raise RuntimeError('QA helper unavailable')
    call(['C:/Program Files/nodejs/node.exe',str(root/'.tools/stage3-ui-qa.mjs')],'browser-tests')
finally:
    (root/'.tools/frontend-qa.stop').touch()
    helper.wait(timeout=20)

