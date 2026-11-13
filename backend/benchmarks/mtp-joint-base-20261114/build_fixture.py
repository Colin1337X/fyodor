"""Build the existing architecture/MTP correctness suite against the experiment."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/joint-base-generation-20261114'
destination=out/'fixture-build.json'
if destination.exists():raise SystemExit('Refusing to overwrite fixture evidence')
build=json.loads((out/'generation-build.json').read_text())
for name,expected in build['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
original=root/'backend/tests/test_architectures.c'
source=original.read_text().replace('#include "../core/llm_cpu.c"',f'#include "{(work/"core.c").as_posix()}"').replace('#include "../core/multimodal.h"','#include "multimodal.h"')
fixture=work/'test_architectures.c';fixture.write_text(source)
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
commands=[];inputs=[original,fixture,work/'core.c',root/'build-cuda/libnya-engine.a',Path(__file__)]
for standard in ('c17','c23'):
    exe=work/f'architecture-{standard}.exe'
    command=[str(compiler),f'-std={standard}','-O3','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Werror','-I','backend/include','-I','backend/core','-I','backend/tests',str(fixture),str(root/'build-cuda/libnya-engine.a'),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
    subprocess.run(command,env=env,check=True);commands.append(command);inputs.append(exe)
destination.write_text(json.dumps(dict(accepted_benchmark=False,executed=False,commands=commands,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
print('Built C17/C23 architecture fixtures; no GPU execution')
