"""Exercise the real optional symbol-loading block with missing CUDA entries."""
import ctypes as C
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/model-attention-query-cache-v2-20261115'
source=(work/'cuda.c').read_text()
a=source.index('#ifdef NYA_WIDE_BATCH_SPLIT\n    if(!c->reference)')
b=source.index('#endif',a)+len('#endif')
text='''#include <stddef.h>
#include <string.h>
#define CUDA_SUCCESS 0
#define NYA_WIDE_BATCH_SPLIT 1
typedef void *CUfunction;
static unsigned symbols, calls;
static const char *setting;
static int sentinel;
static const char *fake_getenv(const char *name) {
 return strcmp(name,"NYA_CUDA_CACHED_QUERY") ? NULL : setting;
}
#define getenv fake_getenv
static int lookup(CUfunction *function,void *module,const char *name) {
 (void)module;
 const char *names[]={"nya_attention_batch_split16_split","nya_attention_batch_split16_ring_split",
 "nya_attention_cached_query_split","nya_attention_cached_query_ring_split"};
 for(unsigned i=0;i<4;++i) if(!strcmp(name,names[i])) {
  calls|=1u<<i;
  if(symbols&(1u<<i)){*function=&sentinel;return CUDA_SUCCESS;}
  return 1;
 }
 return 2;
}
typedef struct {
 int reference;
 CUfunction wide_batch_split[2],cached_query[2];
 void *module;
 int (*function_get)(CUfunction *,void *,const char *);
} context;
unsigned choose(unsigned available,int reference,const char *flag) {
 symbols=available;calls=0;setting=flag;
 context instance={0};context *c=&instance;c->reference=reference;c->function_get=lookup;
'''+source[a:b]+'''
 return (c->wide_batch_split[0]?1u:0u)|(c->wide_batch_split[1]?2u:0u)|
        (c->cached_query[0]?4u:0u)|(c->cached_query[1]?8u:0u)|(calls<<4);
}
'''
path=work/'cached-query-lookup.c';path.write_text(text)
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=dict(os.environ,PATH=str(compiler.parent)+os.pathsep+os.environ['PATH'])
inputs=[Path(__file__).resolve(),work/'cuda.c',path,compiler];counts={}
for standard in ('c17','c23'):
 dll=work/f'cached-query-lookup-{standard}.dll'
 subprocess.run([str(compiler),'-std='+standard,'-O3','-Wall','-Wextra','-Wpedantic',
  '-Wconversion','-Wshadow','-Werror','-shared',str(path),'-o',str(dll)],env=env,check=True)
 lib=C.CDLL(str(dll));f=lib.choose;f.argtypes=[C.c_uint,C.c_int,C.c_char_p];f.restype=C.c_uint
 count=0
 for available,reference,flag in itertools.product(range(16),(0,1),(None,b'',b'0',b'1',b'2',b'01')):
  queried=0 if reference else (15 if flag==b'1' else 3)
  expected=(available&queried)|(queried<<4)
  if f(available,reference,flag)!=expected:raise ValueError((available,reference,flag))
  count+=1
 counts[standard]=count;inputs.append(dll)
with (out/'lookup.json').open('x') as output:
 json.dump(dict(passed=True,gpu_execution=False,cases=counts,sha256={str(p.relative_to(root)):
   hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),output,indent=2)
print('Optional symbol lookup checks passed',counts)
