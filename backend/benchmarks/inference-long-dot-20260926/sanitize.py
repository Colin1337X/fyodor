import json, os, subprocess
from pathlib import Path
root=Path.cwd(); out=root/'backend/benchmarks/inference-long-dot-20260926'
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cuda',NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0',NYA_TEST_NATIVE_LONG='1',NYA_TEST_BLAS_CHUNKS='1')
records=[]
for tile in ('32','64'):
    for tool in ('memcheck','synccheck'):
        current={**env,'NYA_CUDA_GEMM_TILE':tile}
        command=[str(sanitizer),'--tool',tool,'--error-exitcode','99',str(root/'.tools/fyodor-quant-long-dot-candidate.exe')]
        result=subprocess.run(command,env=current,capture_output=True,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
        log=f'sanitize-{tool}-{tile}.log'
        (out/log).write_bytes(result.stdout+result.stderr)
        records.append({'tool':tool,'tile':tile,'command':command,'environment':{k:v for k,v in current.items() if k.startswith('NYA_')},'exit_code':result.returncode,'log':log})
        (out/'sanitizers.json').write_text(json.dumps(records,indent=2))
        print(tool,tile,result.returncode,flush=True)
        assert result.returncode==0 and b'ERROR SUMMARY: 0 errors' in result.stdout+result.stderr
