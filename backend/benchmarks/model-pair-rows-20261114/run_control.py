"""Exercise the actual extracted C dispatch predicate, without a GPU."""
import ctypes as C
import hashlib,itertools,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;work=root/'build-cuda/model-pair-rows-20261114'
if (out/'control.json').exists():raise SystemExit('Refusing overwrite')
path=work/'resident.inc';source=path.read_text()
a=source.index('static int plan_pair_rows(');b=source.index('static int plan_matvec_impl(',a)
text='''#include <stddef.h>
typedef struct { void *pair_rows; } fake_cuda;
typedef struct { fake_cuda *cuda; size_t batch; int interleaved_integer_enabled; } nya_cuda_plan;
'''+source[a:b]+'''
int choose(int present, int enabled, size_t batch, unsigned type, size_t rows,
           size_t columns, unsigned long long stride) {
    int sentinel=0;
    fake_cuda cuda={present ? &sentinel : NULL};
    nya_cuda_plan plan={&cuda,batch,enabled};
    return plan_pair_rows(&plan,type,rows,columns,stride);
}
'''
cpath=work/'pair-control.c';cpath.write_text(text)
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=dict(os.environ,PATH=str(compiler.parent)+os.pathsep+os.environ['PATH'])
inputs=[Path(__file__).resolve(),path,cpath,compiler];counts={}
for standard in ('c17','c23'):
    dll=work/f'pair-control-{standard}.dll'
    subprocess.run([str(compiler),'-std='+standard,'-O3','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Werror','-shared',str(cpath),'-o',str(dll)],env=env,check=True)
    lib=C.CDLL(str(dll));f=lib.choose;f.argtypes=[C.c_int,C.c_int,C.c_size_t,C.c_uint,C.c_size_t,C.c_size_t,C.c_ulonglong];f.restype=C.c_int
    n=0
    for present,enabled,batch,kind,shape,delta in itertools.product((0,1),(0,1),(0,1,2,3,4,8,512),(0,1,12,14,30),((3840,15360),(8192,3840),(15360,3840),(3840,512),(3840,8192),(4096,3840),(0,0),(3841,15360),(3840,15359)),(-1,0,1)):
        columns,rows=shape;stride=max(0,columns//256*144+delta)
        expected=bool(present and enabled and batch==4 and kind==12 and shape in ((3840,15360),(8192,3840),(15360,3840)) and delta==0)
        actual=bool(f(present,enabled,batch,kind,rows,columns,stride))
        if actual!=expected:raise ValueError((present,enabled,batch,kind,shape,delta,actual))
        n+=1
    counts[standard]=n;inputs.append(dll)
(out/'control.json').write_text(json.dumps(dict(passed=True,gpu_execution=False,cases=counts,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
print('C17/C23 actual dispatch checks passed:',counts)
