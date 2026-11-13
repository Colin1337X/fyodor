from pathlib import Path
import subprocess,os,json,hashlib,sys
root=Path.cwd();out=root/'backend/benchmarks/gemma4-wide-value-20261113'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((root/'backend/benchmarks/gemma4-gate-20261113/model-oracle-exit.json').read_text())['environment'])
# Exercise FDPP, packed projections and fused gate selection together.
env.update(NYA_CUDA_FLASH_DECODING_PP='short',NYA_CUDA_BYTE_DOT='1',NYA_CUDA_GEMMA_GATE='1',NYA_CUDA_GEMMA_GATE_PARTIAL='1',NYA_CUDA_DEBUG='1',NYA_CUDA_NVRTC=str(root/'.tools/cuda-sdk/cuda_nvrtc-windows-x86_64-13.1.115-archive/bin/x64/nvrtc64_130_0.dll'))
build=sys.argv[1] if len(sys.argv)>1 else 'build-cuda'
ring=sys.argv[2] if len(sys.argv)>2 else '1'
counts=[int(x) for x in sys.argv[3:]] or [257,2049,8194]
tag='' if build=='build-cuda' and ring=='1' else build+'-ring'+ring+'-'
exe=root/build/'gemma-wide-value-oracle.exe'
runs=[]
env["NYA_CUDA_RING_KV"]=ring
for count in counts:
 p=subprocess.run([str(exe),str(count)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (out/f'{tag}optimized-{count}.log').write_bytes(p.stdout+p.stderr)
 runs.append(dict(count=count,exit_code=p.returncode))
 (out/(tag+'optimized-results.json')).write_text(json.dumps(dict(environment={k:v for k,v in env.items() if k.startswith('NYA_')},binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),runs=runs),indent=2)+'\n')
 print(count,p.returncode,p.stdout.decode(errors='replace'),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
