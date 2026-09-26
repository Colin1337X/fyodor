"""Re-profile both binaries after acceptance; diagnostic timing only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent/'reprofile'
out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CPU_THREADS='6',NYA_CUDA_BATCH='512',NYA_CUDA_BLAS='1',NYA_CUDA_CUTLASS='0',NYA_CUDA_PROFILE='1',NYA_CUDA_BLAS_LIBRARY=str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'))
rows=[]
for prefix in (0,1024):
    for mode in ('before','after'):
        exe=root/('.tools/fyodor-bench-before-split-attention.exe' if mode=='before' else 'build-cuda/fyodor-bench.exe')
        command=[str(exe),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','512' if not prefix else '0','-n','0' if not prefix else '128','--context',str(prefix),'-r','3','--warmup','1','--json']
        with exe.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        p=subprocess.run(command,env=env,capture_output=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        name=f'{prefix}-{mode}'
        (out/(name+'.json')).write_bytes(p.stdout);(out/(name+'.log')).write_bytes(p.stderr)
        rows.append({'name':name,'command':command,'binary_sha256':digest,'exit_code':p.returncode})
        (out/'metadata.json').write_text(json.dumps({'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'caveat':'CUDA events disable graphs, include warmup and prefix preparation, and may include host submission gaps. Not normal throughput.','runs':rows},indent=2))
        assert p.returncode==0
        print(name,p.stderr.decode().strip(),flush=True)
