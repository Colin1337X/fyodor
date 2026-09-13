import datetime,hashlib,json,os,platform,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'backend/benchmarks/stage6-quiet';out.mkdir(exist_ok=False)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return p.stdout.strip()
base={**os.environ,'NYA_CPU_THREADS':'6','NYA_CUDA_BATCH':'512',
      'NYA_CUDA_BLAS_LIBRARY':str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'),
      'NYA_CUDA_CUTLASS_LIBRARY':str(root/'build-cutlass/fyodor-cutlass.dll')}
for k in ['NYA_CUDA_REFERENCE','NYA_CUDA_PROFILE','NYA_CUDA_GRAPHS','NYA_CUDA_MEMORY_MIB','NYA_CUDA_BLAS_SCRATCH_MIB']:
    base.pop(k,None)
metadata={'machine':platform.platform(),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'),
 'library_sha256':{str(p):sha(p) for p in [root/'build-cutlass/fyodor-cutlass.dll',root/'.tools/llama-b10809-cuda/llama.dll',root/'.tools/llama-b10809-cuda/ggml-cuda.dll',root/'.tools/llama-b10809-cuda/cublas64_13.dll']},
 'llama_commit':'5266f24da75dc449bd56cbed7addb9c8e4a6a73e',
 'caveats':['User authorized stopping competing GPU applications; process list archived separately.',
 'Windows compositor and Codex remain active; pre-run utilization was about 9-11%. This is not headless isolation.',
 'No builds, tests, packaging or other Fyodor jobs run during the timed comparisons.',
 'All engines use identical deterministic token IDs, F32 KV, full GPU offload, six threads, batch 512, nine repetitions and three warmups.',
 'llama uses the independent C API harness with flash attention disabled; internal precision and cache alignment remain engine-specific.'],
 'runs':[]}
for workload,prefix in [('short',0),('long',1024)]:
    engines=['native','cublas','cutlass','llama-capi']
    for phase,order in [('a',engines),('b',list(reversed(engines)))]:
        for engine in order:
            env=base.copy();env['NYA_CUDA_BLAS']='1' if engine=='cublas' else '0';env['NYA_CUDA_CUTLASS']='1' if engine=='cutlass' else '0'
            exe=root/('.tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe' if engine=='llama-capi' else 'build-cuda/fyodor-bench.exe')
            cmd=[str(exe),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','0' if prefix else '512','-n','128','--context',str(prefix),'-r','9','--warmup','3','--json']
            if engine=='llama-capi':cmd+=['--batch','512','--threads','6']
            name=f'{workload}-{engine}-{phase}'
            row={'name':name,'command':cmd,'binary_sha256':sha(exe),'gpu_before':gpu(),'environment':{k:v for k,v in env.items() if k.startswith('NYA_')}}
            p=subprocess.run(cmd,env=env,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(exit_code=p.returncode,gpu_after=gpu());metadata['runs'].append(row)
            (out/(name+'.json')).write_bytes(p.stdout);(out/(name+'.log')).write_bytes(p.stderr)
            (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
            if p.returncode:raise RuntimeError((name,p.returncode))
            result=json.loads(p.stdout)
            if result['kv_type']!='f32':raise RuntimeError('cache mismatch')
            if engine!='llama-capi' and result['execution']!='resident':raise RuntimeError('CPU fallback')
            if engine=='cutlass' and not prefix and not result['results'][0]['cutlass_matmul_calls']:raise RuntimeError('CUTLASS not exercised')
            print(name,[(r['test'],r['tokens_per_second'],r['stddev']) for r in result['results']],flush=True)
