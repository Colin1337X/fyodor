"""Build isolated CUDA override with the validated eight-row/bound-three kernel."""
import hashlib,json,os,re,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/integer-launch-20261114';manifest=out/'model-build.json'
if work.exists() or manifest.exists(): raise SystemExit('Refusing to overwrite build evidence')
proof=json.loads((out/'memcheck-run.json').read_text());assert proof['passed']
for n,h in proof['input_sha256'].items(): assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
for n,h in proof['evidence_sha256'].items(): assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
work.mkdir()
original=root/'build-cuda/cuda-provider/cuda_source.h'
header=original.read_text();marker='static const unsigned char nya_cuda_source[] = {'
assert header.count(marker)==1
start=header.index(marker)+len(marker);end=header.index('};',start)
raw=bytes(int(v.strip(),0) for v in header[start:end].split(',') if v.strip())
assert raw[-1:]==b'\0'
candidate=root/'optcpp/matvec/integer_launch.cu'
raw=raw[:-1]+b'\n'+candidate.read_bytes()+b'\0'
header=header[:start]+','.join(str(v) for v in raw)+header[end:]
(work/'cuda_source.h').write_text(header)
source=(root/'CUDA/cuda.c').read_text()
assert source.count('"nya_interleaved_integer_12_4"')==1
source=source.replace('"nya_interleaved_integer_12_4"','"nya_integer_launch_8_3"')
(work/'cuda.c').write_text(source)
flags=(root/'build-cuda/CMakeFiles/nya-engine.dir/flags.make').read_text()
defines=re.search(r'^C_DEFINES = (.*)$',flags,re.M)[1].split()
cflags=re.search(r'^C_FLAGS = (.*)$',flags,re.M)[1].split()
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
library=root/'build-cuda/libnya-engine.a';library_hash=hashlib.sha256(library.read_bytes()).hexdigest()
commands=[]
for standard in ('c17','c23'):
 commands.append([str(compiler),*defines,*[f'-std={standard}' if f.startswith('-std=') else f for f in cflags],
 '-I',str(root/'CUDA'),'-I',str(root/'backend/core'),'@'+str(root/'build-cuda/CMakeFiles/nya-engine.dir/includes_C.rsp'),
 '-c',str(work/'cuda.c'),'-o',str(work/f'cuda-{standard}.o')])
for command in commands: subprocess.run(command,env=env,check=True)
assert hashlib.sha256(library.read_bytes()).hexdigest()==library_hash
inputs=[original,candidate,library,Path(__file__),out/'memcheck-run.json',work/'cuda_source.h',work/'cuda.c',work/'cuda-c17.o',work/'cuda-c23.o']
inputs+=list((root/'CUDA').glob('*.inc'))+list((root/'CUDA').glob('*.h'))+[root/'CUDA/cuda.c',root/'CUDA/cutlass/loader.inc',root/'CUDA/cutlass/host.h']
manifest.write_text(json.dumps(dict(passed=True,accepted_benchmark=False,commands=commands,
 sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
print('Strict C17/C23 isolated CUDA overrides built; production library unchanged')
