import os,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
env={**os.environ,'PATH':str(root/'.tools/w64devkit/bin')+os.pathsep+'C:/Program Files/nodejs'+os.pathsep+os.environ['PATH'],
 'NYA_CPU_THREADS':'6','NYA_CUDA_BLAS_LIBRARY':str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'),
 'NYA_CUDA_CUTLASS_LIBRARY':str(root/'build-cutlass/fyodor-cutlass.dll')}
def call(cmd,name,runenv=env):
    with open(root/'.tools'/('stage8-'+name+'.log'),'wb') as f:
        p=subprocess.run(cmd,cwd=root,env=runenv,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    print(name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError((name,p.returncode))
cmake=str(root/'.tools/cmake/cmake/data/bin/cmake.exe');ctest=str(root/'.tools/cmake/cmake/data/bin/ctest.exe')
call([cmake,'--build','build-cuda','-j4'],'final-build')
call([ctest,'--test-dir','build-cuda','--output-on-failure'],'accepted-c17-tests')
gpu={**env,'NYA_COMPUTE':'cuda','NYA_CUDA_BLAS':'1','NYA_CUDA_CUTLASS':'0','NYA_TEST_BLAS_CHUNKS':'1','NYA_CUDA_BLAS_SCRATCH_MIB':'1'}
test=str(root/'build-cuda/nya-test-quant-kernels.exe')
for config in ['debug','release']:
    runenv={**gpu,'NYA_CUDA_BLAS':'0','NYA_CUDA_CUTLASS':'1','NYA_TEST_CUTLASS':'1','NYA_CUDA_CUTLASS_LIBRARY':str(root/('build-cutlass-debug' if config=='debug' else 'build-cutlass')/'fyodor-cutlass.dll')}
    call([test],'cutlass-'+config+'-quant',runenv)
san=str(root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe')
for tool in ['memcheck','initcheck','racecheck','synccheck']:
    filters=['--kernel-name','kns=nya_expand_t_'] if tool=='racecheck' else []
    call([san,'--tool',tool,'--error-exitcode','99',*filters,test],'cuda-'+tool,gpu)
call([sys.executable,str(root/'.tools/stage8_final_bench.py')],'final-bench')
# Refresh the sidecar and distributable packages only after executable checks.
call(['cmd.exe','/d','/s','/c','npm --prefix frontend run prepare:sidecar'],'sidecar')
call(['cmd.exe','/d','/s','/c','"C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" && cd frontend && npm run tauri -- build'],'package')
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
    (root/'.tools/frontend-qa.stop').touch();helper.wait(timeout=20)
