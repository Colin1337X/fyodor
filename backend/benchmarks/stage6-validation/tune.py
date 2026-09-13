import json,os,subprocess,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'backend/benchmarks/stage6-tile-tuning';out.mkdir(exist_ok=True)
env={**os.environ,'CUDACXX':str(root/'.tools/cuda-toolkit/bin/nvcc.exe'),
     'NYA_CUDA_CUTLASS':'1','NYA_CUDA_BLAS':'0','NYA_COMPUTE':'cuda','NYA_TEST_CUTLASS':'1',
     'NYA_CUDA_BATCH':'512','NYA_CPU_THREADS':'6'}
def call(cmd,name,runenv=env):
    p=subprocess.run(cmd,cwd=root,env=runenv,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    (out/(name+'.out')).write_bytes(p.stdout);(out/(name+'.log')).write_bytes(p.stderr)
    if p.returncode:raise RuntimeError((name,p.returncode))
    return p
def gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
records=[]
for tile in ['64x64x32','64x128x32','128x64x32']:
    build='build-cutlass-'+tile
    # The command is fixed repository tooling; none of its text comes from model data.
    cmd='"C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" && .tools\\cmake\\cmake\\data\\bin\\cmake.exe -S CUDA/cutlass -B '+build+' -G "NMake Makefiles" -DNYA_CUTLASS_ARCHITECTURES=120 -DNYA_CUTLASS_TILE='+tile+' -DCMAKE_BUILD_TYPE=Release && .tools\\cmake\\cmake\\data\\bin\\cmake.exe --build '+build
    call('cmd.exe /d /s /c "'+cmd+'"',tile+'-build')
    env['NYA_CUDA_CUTLASS_LIBRARY']=str(root/build/'fyodor-cutlass.dll')
    call([str(root/'build-cuda/nya-test-quant-kernels.exe')],tile+'-quant')
    call([str(root/'build-cuda/nya-test-inference.exe'),str(root/'build-cuda/backend/test_models/pretraining.trained.gguf'),'resident'],tile+'-inference')
    row={'tile':tile,'library_sha256':hashlib.file_digest(open(env['NYA_CUDA_CUTLASS_LIBRARY'],'rb'),'sha256').hexdigest(),'gpu_before':gpu()}
    p=call([str(root/'build-cuda/fyodor-bench.exe'),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','512','-n','0','-r','5','--warmup','3','--json'],tile+'-bench')
    row['result']=json.loads(p.stdout);row['gpu_after']=gpu();records.append(row)
    (out/'metadata.json').write_text(json.dumps(records,indent=2))
    print(tile,[(r['tokens_per_second'],r['stddev']) for r in row['result']['results']],flush=True)
