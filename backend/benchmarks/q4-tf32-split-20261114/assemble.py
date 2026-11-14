"""Assemble verified PTX offline; never create a CUDA context."""
import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
proof_path=out/'compile.json';proof=json.loads(proof_path.read_text())
if (out/'assembly.json').exists():raise SystemExit('Refusing assembly overwrite')
if proof.get('compile_passed') is not True:raise ValueError('Compile failed')
inputs={root/name for name in proof['sha256']}
for name,h in proof['sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
work=root/'build-cuda/q4-tf32-split-20261114'
compiler=root/'.tools/cuda-sdk/cuda_nvcc-windows-x86_64-13.1.115-archive/bin/ptxas.exe'
command=[str(compiler),str(work/'module.ptx'),'-arch=sm_120','-O3','-v','-o',str(work/'module.cubin')]
completed=subprocess.run(command,capture_output=True)
log=out/'assembly.log';log.write_bytes(completed.stdout+completed.stderr)
if completed.returncode:raise SystemExit(completed.returncode)
inputs.update((Path(__file__),proof_path,compiler,log,work/'module.cubin'))
result=dict(assembled=True,gpu_execution=False,accepted_benchmark=False,command=command,
 sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
(out/'assembly.json').write_text(json.dumps(result,indent=2))
print('SM120 assembly passed; no GPU execution')
