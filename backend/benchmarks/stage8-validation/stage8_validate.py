import os, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
env={**os.environ,'PATH':str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'],
 'NYA_CUDA_BLAS_LIBRARY':str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'),
 'NYA_CUDA_CUTLASS_LIBRARY':str(root/'build-cutlass/fyodor-cutlass.dll')}
for k in ['NYA_COMPUTE','NYA_CUDA_REFERENCE','NYA_CUDA_PROFILE','NYA_TEST_GRAPH','NYA_TEST_ATTENTION_TILES','NYA_CUDA_BLAS','NYA_CUDA_CUTLASS']:env.pop(k,None)
cmake=str(root/'.tools/cmake/cmake/data/bin/cmake.exe');ctest=str(root/'.tools/cmake/cmake/data/bin/ctest.exe')
def call(cmd,name,runenv=env):
    with open(root/'.tools'/('stage8-'+name+'.log'),'wb') as f:
        p=subprocess.run(cmd,cwd=root,env=runenv,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    print(name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError((name,p.returncode))
call([cmake,'--build','build-cuda','-j4'],'accepted-build')
call([ctest,'--test-dir','build-cuda','--output-on-failure'],'accepted-c17-tests')
gpu={**env,'NYA_COMPUTE':'cuda','NYA_CUDA_BLAS':'1','NYA_CUDA_CUTLASS':'0'}
call([str(root/'build-cuda/nya-test-inference.exe'),str(root/'.tools/tinyllama-q4_k_m.gguf'),'resident','65'],'real-tiny',gpu)
call([str(root/'build-cuda/nya-test-inference.exe'),'D:/Models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','resident','33'],'real-gemma',gpu)
for build in ['build-cuda-c23','build-cpu','build-ort','build-sanitize']:
    runenv=env.copy()
    if build=='build-sanitize':runenv['PATH']=str(root/'.tools/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64/bin')+os.pathsep+runenv['PATH']
    call([cmake,'--build',build,'-j4'],build+'-build',runenv)
    call([ctest,'--test-dir',build,'--output-on-failure'],build+'-tests',runenv)
for build,target,name in [('build-cuda','verify-without-cutlass','without-cutlass'),('build-cuda','verify-without-cuda','without-cuda-vulkan'),('build-cpu','verify-without-cuda','without-cuda-cpu')]:
    call([cmake,'--build',build,'--target',target],name)
node='C:/Program Files/nodejs/node.exe'
call([node,'--test',str(root/'frontend/tests/ui.test.mjs'),str(root/'frontend/tests/sidecar.test.mjs')],'frontend-tests')
call([str(Path(os.__file__).parents[1]/'python.exe'),str(root/'.tools/server_live_qa.py')],'api')
for version in ['before','after']:
    exe=root/('.tools/fyodor-bench-stage7.exe' if version=='before' else 'build-cuda/fyodor-bench.exe')
    for mode in ['cublas','native','cutlass']:
        runenv={**env,'NYA_CUDA_BLAS':'1' if mode=='cublas' else '0','NYA_CUDA_CUTLASS':'1' if mode=='cutlass' else '0','NYA_CUDA_PROFILE':'1'}
        call([str(exe),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','512','-n','0','-r','3','--warmup','1','--json'],'profile-'+version+'-'+mode,runenv)

