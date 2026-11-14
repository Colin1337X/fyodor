"""Compile and test the extracted C predicate, including optional-kernel absence."""
import ctypes as C
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/model-attention-query-cache-v2-20261115'
source=(work/'resident.inc').read_text()
a=source.index('static int plan_cached_query_eligible(');b=source.index('static int plan_gate_up(',a)
text='''#include <stddef.h>
typedef struct { int reference, cached_query[2]; } fake_cuda;
typedef struct { int is_gemma; } fake_model;
typedef struct { fake_cuda *cuda; fake_model *model; } nya_cuda_plan;
'''+source[a:b]+'''
int choose(unsigned flags,size_t count,unsigned width,unsigned heads,
           unsigned kvheads,unsigned window,unsigned ring) {
 fake_cuda cuda={(flags&4)!=0,{(flags&1)!=0,(flags&2)!=0}};
 fake_model model={(flags&8)!=0};nya_cuda_plan plan={&cuda,&model};
 return !!plan_cached_query_eligible(&plan,(flags&16)!=0,count,width,heads,kvheads,window,ring);
}
'''
cpath=work/'cached-query-control.c';cpath.write_text(text)
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=dict(os.environ,PATH=str(compiler.parent)+os.pathsep+os.environ['PATH'])
shapes=[(256,16,8,1024),(512,16,1,0),(256,16,1,1024),(512,16,8,0),
        (256,16,8,0),(512,16,1,1024),(256,15,8,1024),(512,17,1,0),
        (255,16,8,1024),(513,16,1,0),(0,0,0,0),(256,16,8,1023)]
inputs=[Path(__file__).resolve(),work/'resident.inc',cpath,compiler];counts={}
for standard in ('c17','c23'):
 dll=work/f'cached-query-control-{standard}.dll'
 subprocess.run([str(compiler),'-std='+standard,'-O3','-Wall','-Wextra','-Wpedantic',
   '-Wconversion','-Wshadow','-Werror','-shared',str(cpath),'-o',str(dll)],env=env,check=True)
 lib=C.CDLL(str(dll));f=lib.choose
 f.argtypes=[C.c_uint,C.c_size_t,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint];f.restype=C.c_int
 n=0
 for flags,count,shape,ring in itertools.product(range(32),(0,1,2,3,4,8,512),shapes,(0,1,1027)):
  expected=bool(flags&16 and flags&8 and not flags&4 and flags&(2 if ring else 1)
                and count in (2,4) and shape in ((256,16,8,1024),(512,16,1,0)))
  if bool(f(flags,count,*shape,ring))!=expected:raise ValueError((flags,count,shape,ring))
  n+=1
 counts[standard]=n;inputs.append(dll)
with (out/'control.json').open('x') as output:
 json.dump(dict(passed=True,gpu_execution=False,cases=counts,sha256={str(p.relative_to(root)):
     hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),output,indent=2)
print('Extracted C dispatch checks passed',counts)
