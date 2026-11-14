"""Compile and exercise the actual model dispatch predicate without CUDA."""
import ctypes as C
import hashlib, itertools, json, os, subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/model-q4-batch-gate-20261115'
if (out/'control.json').exists():raise SystemExit('Refusing overwrite')
source=(work/'resident.inc').read_text()
a=source.index('static int plan_batch_gate_eligible(')
b=source.index('static int plan_batch_gate(',a)
text='''#include <stddef.h>
typedef struct { unsigned type; size_t data_size; } nya_llm_tensor;
typedef struct { int batch_gate, reference; } fake_cuda;
typedef struct { int is_gemma; } fake_model;
typedef struct {
 fake_cuda *cuda; fake_model *model; size_t batch;
 int fuse_gate_up, interleaved_integer_enabled, scratch, blas_storage, vendor;
} nya_cuda_plan;
static int plan_interleaved_kernel(const nya_cuda_plan *p,unsigned t,size_t c)
{ (void)t;(void)c;return p->scratch; }
static int nya_blas_bytes(fake_cuda *c,size_t r,size_t k,size_t n)
{ (void)c;(void)r;(void)k;(void)n;return 1; }
'''+source[a:b]+'''
int choose(unsigned flags,size_t batch,size_t rows,size_t columns,
           unsigned gt,unsigned ut,size_t gb,size_t ub) {
 fake_cuda cuda={(flags&1)!=0,(flags&2)!=0};
 fake_model model={(flags&4)!=0};
 nya_cuda_plan p={&cuda,&model,batch,(flags&8)!=0,(flags&16)!=0,
                 (flags&32)!=0,(flags&64)!=0,0};
 nya_llm_tensor gate={gt,gb},up={ut,ub};
 return !!plan_batch_gate_eligible(&p,(flags&128)?NULL:&gate,
                                 (flags&256)?NULL:&up,rows,columns);
}
'''
cpath=work/'batch-gate-control.c';cpath.write_text(text)
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=dict(os.environ,PATH=str(compiler.parent)+os.pathsep+os.environ['PATH'])
inputs=[Path(__file__).resolve(),work/'resident.inc',cpath,compiler];counts={}
size=15360*15*144
cases=[(flags,4,15360,3840,12,12,size,size,flags==61) for flags in range(512)]
for batch,rows,cols,gt,ut,gd,ud in itertools.product(
 (0,1,2,3,4,8,512),(0,15359,15360,15361),(0,3839,3840,3841),
 (0,12,14),(0,12,14),(-1,0,1),(-1,0,1)):
 cases.append((61,batch,rows,cols,gt,ut,size+gd,size+ud,
               (batch,rows,cols,gt,ut,gd,ud)==(4,15360,3840,12,12,0,0)))
for standard in ('c17','c23'):
 dll=work/f'batch-gate-control-{standard}.dll'
 subprocess.run([str(compiler),'-std='+standard,'-O3','-Wall','-Wextra','-Wpedantic',
                 '-Wconversion','-Wshadow','-Werror','-shared',str(cpath),'-o',str(dll)],env=env,check=True)
 lib=C.CDLL(str(dll));f=lib.choose
 f.argtypes=[C.c_uint,C.c_size_t,C.c_size_t,C.c_size_t,C.c_uint,C.c_uint,C.c_size_t,C.c_size_t]
 f.restype=C.c_int
 for *args,expected in cases:
  if bool(f(*args))!=expected:raise ValueError(args)
 counts[standard]=len(cases);inputs.append(dll)
(out/'control.json').write_text(json.dumps(dict(passed=True,gpu_execution=False,cases=counts,
 sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
print('Actual C dispatch checks passed:',counts)
