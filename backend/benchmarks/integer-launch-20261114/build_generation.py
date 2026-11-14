"""Link validated joint-base core with isolated Q4 launch override."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/integer-launch-20261114';dest=out/'generation-build.json'
if dest.exists() or (work/'generation-c17.exe').exists() or (work/'generation-c23.exe').exists():
 raise SystemExit('Refusing to overwrite generation build')
proofs=[out/'model-build.json',out.parent/'mtp-joint-base-v2-20261114/generation-build.json']
for p in proofs:
 for name,h in json.loads(p.read_text())['sha256'].items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
library=root/'build-cuda/libnya-engine.a';before=hashlib.sha256(library.read_bytes()).hexdigest()
model=out.parent/'mtp-joint-base-v2-20261114/generation.c';commands=[];inputs=[*proofs,model,library,Path(__file__)]
for standard in ('c17','c23'):
 core=root/f'build-cuda/joint-base-generation-v2-20261114/core-{standard}.o'
 cuda=work/f'cuda-{standard}.o';exe=work/f'generation-{standard}.exe'
 command=[str(compiler),f'-std={standard}','-O3','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Werror',
 '-I','backend/include','-I','backend/core',str(model),str(core),str(cuda),str(library),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
 commands.append(command);subprocess.run(command,env=env,check=True);inputs += [core,cuda,exe]
assert hashlib.sha256(library.read_bytes()).hexdigest()==before
dest.write_text(json.dumps(dict(passed=True,accepted_benchmark=False,commands=commands,
 sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
print('Linked isolated joint-base generation C17/C23')
