import datetime, hashlib, json, os, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
out = root / 'backend/benchmarks/stage7-final'
out.mkdir(exist_ok=False)
def sha(p):
    with open(p, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
env = {**os.environ, 'NYA_CPU_THREADS':'6', 'NYA_CUDA_BATCH':'512', 'NYA_CUDA_BLAS':'1', 'NYA_CUDA_CUTLASS':'0', 'NYA_CUDA_BLAS_LIBRARY':str(root/'.tools/llama-b10809-cuda/cublas64_13.dll')}
for k in ['NYA_CUDA_REFERENCE','NYA_CUDA_PROFILE','NYA_CUDA_GRAPHS','NYA_CUDA_MEMORY_MIB','NYA_CUDA_BLAS_SCRATCH_MIB']: env.pop(k,None)
meta = {'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'), 'environment':{k:v for k,v in env.items() if k.startswith('NYA_')}, 'runs':[]}
for prefix in [0,1024]:
    for phase, order in [('a',['before','after','llama']),('b',['llama','after','before'])]:
        for version in order:
            exe = root / ('.tools/fyodor-bench-stage6.exe' if version=='before' else '.tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe' if version=='llama' else 'build-cuda/fyodor-bench.exe')
            cmd = [str(exe),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','0' if prefix else '512','-n','128','--context',str(prefix),'-r','9','--warmup','3','--json']
            if version=='llama': cmd += ['--batch','512','--threads','6']
            name = f'{prefix}-{version}-{phase}'
            row = {'name':name,'command':cmd,'binary_sha256':sha(exe),'gpu_before':gpu()}
            run = subprocess.run(cmd,env=env,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(exit_code=run.returncode,gpu_after=gpu()); meta['runs'].append(row)
            (out/(name+'.json')).write_bytes(run.stdout);(out/(name+'.log')).write_bytes(run.stderr)
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            if run.returncode: raise RuntimeError((name,run.returncode))
            result=json.loads(run.stdout)
            if (version!='llama' and result['execution']!='resident') or result['kv_type']!='f32': raise RuntimeError('execution mismatch')
            print(name,[(r['test'],r['tokens_per_second'],r['stddev']) for r in result['results']],flush=True)


