"""Instrument both decode graphs and transitions on a long F32 fixture."""
import json
import os
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
model=root/'build-train-profile/split-attention-sanitizer.gguf'
shutil.copyfile(root/'build-cuda/backend/test_models/attention-shape-1.gguf',model)
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cuda',NYA_TEST_GRAPH='1',NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0')
records=[]
for tool in ('memcheck','racecheck','initcheck','synccheck'):
    command=[str(sanitizer),'--tool',tool,'--error-exitcode','99','--launch-timeout','120',str(root/'build-cuda/nya-test-inference.exe'),str(model),'resident','273']
    result=subprocess.run(command,env=env,capture_output=True,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
    log=out/('cuda-'+tool+'.log')
    log.write_bytes(result.stdout+result.stderr)
    records.append({'tool':tool,'command':command,'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'exit_code':result.returncode,'log':log.name})
    (out/'cuda-sanitizers.json').write_text(json.dumps(records,indent=2))
    if result.returncode or b'2 graph captures' not in result.stderr:raise RuntimeError(records[-1])
    print(tool,result.returncode,flush=True)
