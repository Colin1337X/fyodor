import hashlib, json, os, subprocess
from pathlib import Path
root=Path.cwd(); out=root/'backend/benchmarks/inference-long-dot-20260926'
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cuda',NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0',NYA_TEST_NO_BLAS='1',NYA_TEST_CPU_OPTIMIZED='1',NYA_CPU_THREADS='6')
exe=root/'.tools/fyodor-test-long-dot-candidate.exe'
model=root/'.tools/tinyllama-q4_k_m.gguf'
records=[]
for tile in ('32','64'):
    command=[str(exe),str(model),'resident','513']
    current={**env,'NYA_CUDA_GEMM_TILE':tile}
    result=subprocess.run(command,env=current,capture_output=True,timeout=240)
    log=f'real-native-tile-{tile}.log'
    (out/log).write_bytes(result.stdout+result.stderr)
    records.append({'command':command,'environment':{k:v for k,v in current.items() if k.startswith('NYA_')},'exit_code':result.returncode,'log':log,'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()})
    (out/'real-model.json').write_text(json.dumps(records,indent=2))
    print(tile,result.returncode,result.stderr.decode(errors='replace'),flush=True)
    assert result.returncode==0
