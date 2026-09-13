import datetime, hashlib, json, os, subprocess, time, csv
from pathlib import Path
root = Path(__file__).resolve().parents[1]
out = root / 'backend/benchmarks/stage8-final'
out.mkdir(exist_ok=True)
if any(out.iterdir()):raise RuntimeError('Refusing to overwrite benchmark evidence')
def sha(p):
    with open(p, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()

def quiet_gpu():
    # Let utilization sampling expire after this process's preceding GPU work.
    time.sleep(2)
    consecutive=0
    for _ in range(15):
        state=gpu()
        # The active Windows compositor/Codex state occupies about 9%. Require
        # the same band throughout instead of mixing its active and idle states.
        utilization=int(state.split('%')[0].strip())
        consecutive=consecutive+1 if 7<=utilization<=12 else 0
        if consecutive==2:return state
        time.sleep(1)
    raise RuntimeError('GPU did not return to desktop idle: '+state)

def competitors():
    names={'thorium.exe','chrome.exe','msedge.exe','firefox.exe','RobloxPlayerBeta.exe','Spotify.exe','steamwebhelper.exe','Photos.exe'}
    p=subprocess.run(['tasklist','/FO','CSV','/NH'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if p.returncode:raise RuntimeError('Cannot verify competing applications')
    return [row[0] for row in csv.reader(p.stdout.splitlines()) if row and row[0] in names]
env = {**os.environ, 'NYA_CPU_THREADS':'6', 'NYA_CUDA_BATCH':'512', 'NYA_CUDA_BLAS':'1', 'NYA_CUDA_CUTLASS':'0', 'NYA_CUDA_BLAS_LIBRARY':str(root/'.tools/llama-b10809-cuda/cublas64_13.dll')}
for k in ['NYA_CUDA_REFERENCE','NYA_CUDA_PROFILE','NYA_CUDA_GRAPHS','NYA_CUDA_MEMORY_MIB','NYA_CUDA_BLAS_SCRATCH_MIB']: env.pop(k,None)
meta = {'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'), 'environment':{k:v for k,v in env.items() if k.startswith('NYA_')}, 'idle_gate':'Two consecutive readings in 7..12% before and after each run, following two seconds of settling. Windows GPU Engine counters identify Desktop Window Manager and Codex as remaining activity. Known competing apps must be absent. This is not a fully idle/headless GPU measurement.', 'runs':[]}
for prefix in [0,1024]:
    for phase, order in [('a',['before','after','llama']),('b',['llama','after','before'])]:
        for version in order:
            exe = root / ('.tools/fyodor-bench-stage7.exe' if version=='before' else '.tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe' if version=='llama' else 'build-cuda/fyodor-bench.exe')
            cmd = [str(exe),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','0' if prefix else '512','-n','128','--context',str(prefix),'-r','9','--warmup','3','--json']
            if version=='llama': cmd += ['--batch','512','--threads','6']
            name = f'{prefix}-{version}-{phase}'
            if competitors():raise RuntimeError('Competing application before '+name)
            row = {'name':name,'command':cmd,'binary_sha256':sha(exe),'gpu_before':quiet_gpu()}
            run = subprocess.run(cmd,env=env,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(exit_code=run.returncode,gpu_after=gpu()); meta['runs'].append(row)
            (out/(name+'.json')).write_bytes(run.stdout);(out/(name+'.log')).write_bytes(run.stderr)
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            if run.returncode: raise RuntimeError((name,run.returncode))
            if competitors():raise RuntimeError('Competing application during '+name)
            row['gpu_idle_after']=quiet_gpu()
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            result=json.loads(run.stdout)
            if (version!='llama' and result['execution']!='resident') or result['kv_type']!='f32': raise RuntimeError('execution mismatch')
            print(name,[(r['test'],r['tokens_per_second'],r['stddev']) for r in result['results']],flush=True)



