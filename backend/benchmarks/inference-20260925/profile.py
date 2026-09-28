"""Capture current resident CUDA costs; profiling disables graphs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'baseline'
OUT.mkdir(exist_ok=False)

def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()

env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CPU_THREADS='6', NYA_CUDA_BATCH='512', NYA_CUDA_BLAS='1', NYA_CUDA_CUTLASS='0', NYA_CUDA_BLAS_LIBRARY=str(ROOT / '.tools/llama-b10809-cuda/cublas64_13.dll'))
exe = ROOT / 'build-cuda/fyodor-bench.exe'
model = ROOT / '.tools/tinyllama-q4_k_m.gguf'
metadata = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'exe_sha256': sha(exe), 'model_sha256': sha(model), 'environment': {k: v for k,v in env.items() if k.startswith('NYA_')}, 'caveat': 'Diagnostic desktop runs. Other apps remain open; no performance acceptance from these measurements. CUDA event profiling disables graphs and includes host submission gaps.', 'runs': []}
(OUT/'processes.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
for profile in (False, True):
    for name, prompt, generate, prefix in [('prefill',512,0,0),('short',0,128,0),('long',0,128,1024)]:
        time.sleep(2)
        name = ('profile-' if profile else 'normal-') + name
        cmd = [str(exe), '-m', str(model), '-b','cuda','-p',str(prompt),'-n',str(generate),'--context',str(prefix),'-r','3','--warmup','1','--json']
        row = {'name':name,'command':cmd,'gpu_before':gpu()}
        result = subprocess.run(cmd, env={**env, 'NYA_CUDA_PROFILE':str(int(profile))}, capture_output=True, timeout=240, creationflags=subprocess.CREATE_NO_WINDOW)
        row.update(exit_code=result.returncode, gpu_after=gpu())
        metadata['runs'].append(row)
        (OUT/(name+'.json')).write_bytes(result.stdout)
        (OUT/(name+'.log')).write_bytes(result.stderr)
        (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
        if result.returncode: raise RuntimeError(row)
        data=json.loads(result.stdout)
        if data['execution'] != 'resident': raise RuntimeError(data)
        print(name, [(r['test'],r['tokens_per_second']) for r in data['results']],result.stderr.decode(errors='replace').strip(),flush=True)
