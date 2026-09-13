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

