"""Exploratory reversed-order ablation, separate from acceptance results."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent/'split-probe'
out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CPU_THREADS='6',NYA_CUDA_BATCH='512',NYA_CUDA_CUTLASS='0',NYA_CUDA_BLAS='1',NYA_CUDA_BLAS_LIBRARY=str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'))
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
meta={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'),'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'runs':[],'caveat':'Desktop apps open, exploratory only.'}
for prefix in (0,512,1024):
    for index,split in enumerate(('0','1','1','0')):
        exe=root/'build-cuda/fyodor-bench.exe'
        cmd=[str(exe),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','512' if not prefix else '0','-n','128','--context',str(prefix),'-r','3','--warmup','1','--json']
        name=f'{prefix}-{index}-{split}'
        time.sleep(2)
        row={'name':name,'command':cmd,'binary_sha256':sha(exe),'gpu_before':gpu(),'split':split}
        p=subprocess.run(cmd,env={**env,'NYA_CUDA_SPLIT_ATTENTION':split},capture_output=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        row.update(exit_code=p.returncode,gpu_after=gpu());meta['runs'].append(row)
        (out/(name+'.json')).write_bytes(p.stdout);(out/(name+'.log')).write_bytes(p.stderr)
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))
        if p.returncode:raise RuntimeError(row)
        print(name,[(r['test'],r['tokens_per_second'],r['stddev']) for r in json.loads(p.stdout)['results']],flush=True)
