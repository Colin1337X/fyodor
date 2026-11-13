from pathlib import Path
import subprocess,os,json,hashlib
root=Path.cwd();out=root/'backend/benchmarks/gemma4-ring-runtime-20261113'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((root/'backend/benchmarks/gemma4-gate-20261113/model-oracle-exit.json').read_text())['environment'])
# Exercise FDPP, packed projections and fused gate selection together.
env.update(NYA_CUDA_FLASH_DECODING_PP='short',NYA_CUDA_BYTE_DOT='1',NYA_CUDA_GEMMA_GATE='1',NYA_CUDA_GEMMA_GATE_PARTIAL='1',NYA_CUDA_DEBUG='1',NYA_CUDA_NVRTC=str(root/'.tools/cuda-sdk/cuda_nvrtc-windows-x86_64-13.1.115-archive/bin/x64/nvrtc64_130_0.dll'))
exe=root/'build-cuda-c23/gemma-ring-runtime-oracle.exe'
runs=[]
for count in [2049,8194]:
 p=subprocess.run([str(exe),str(count)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (out/f'optimized-{count}.log').write_bytes(p.stdout+p.stderr)
 runs.append(dict(count=count,exit_code=p.returncode))
 (out/'optimized-results.json').write_text(json.dumps(dict(environment={k:v for k,v in env.items() if k.startswith('NYA_')},binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),runs=runs),indent=2)+'\n')
 print(count,p.returncode,p.stdout.decode(errors='replace'),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
