import os,json,subprocess,datetime,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'backend/benchmarks/stage8-layout';out.mkdir(exist_ok=False)
env={**os.environ,'NYA_CPU_THREADS':'6','NYA_CUDA_BATCH':'512','NYA_CUDA_BLAS':'1','NYA_CUDA_CUTLASS':'0','NYA_CUDA_BLAS_LIBRARY':str(root/'.tools/llama-b10809-cuda/cublas64_13.dll')}
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'),'runs':[]}
for i,mode in enumerate(['0','1','1','0']):
    runenv={**env,'NYA_CUDA_BLAS_TRANSPOSE':mode}
    cmd=[str(root/'build-cuda/fyodor-bench.exe'),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','512','-n','0','-r','7','--warmup','3','--json']
    row={'name':str(i)+'-'+mode,'command':cmd,'binary_sha256':sha(cmd[0]),'environment':{k:v for k,v in runenv.items() if k.startswith('NYA_')},'gpu_before':gpu()}
    p=subprocess.run(cmd,env=runenv,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    row.update(exit_code=p.returncode,gpu_after=gpu());meta['runs'].append(row)
    (out/(row['name']+'.json')).write_bytes(p.stdout);(out/(row['name']+'.log')).write_bytes(p.stderr)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    if p.returncode:raise RuntimeError(row)
    data=json.loads(p.stdout)
    if data['execution']!='resident':raise RuntimeError(data)
    print(row['name'],data['results'][0]['tokens_per_second'],data['results'][0]['stddev'],flush=True)
