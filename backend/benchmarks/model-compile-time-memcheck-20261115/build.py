"""Build isolated progress diagnostics with compiler-instrumented CUDA code."""
import ctypes as C,hashlib,json,os,re,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
base=root/'build-cuda/lifecycle-progress-v2-20261115';work=root/'build-cuda/model-compile-time-memcheck-20261115'
proof_path=root/'backend/benchmarks/lifecycle-progress-20261115/build-v2.json';proof=json.loads(proof_path.read_text())
for name,h in proof['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
if work.exists():raise SystemExit('Refusing overwrite')
work.mkdir()
for name in ('cuda.c','cuda_source.h','resident.inc','lifecycle.c'):(work/name).write_bytes((base/name).read_bytes())
header=(work/'cuda_source.h').read_text()
match=re.search(r'nya_cuda_source\[\]\s*=\s*\{([^}]+)\};',header);assert match
source=bytes(int(v.strip(),0) for v in match[1].split(',') if v.strip());assert source[-1]==0
library=Path(re.search(r'nya_nvrtc_default\[\]\s*=\s*"([^"]+)"',header)[1]);dll_dir=os.add_dll_directory(str(library.parent));nv=C.CDLL(str(library))
P=C.c_void_p;I=C.c_int;Z=C.c_size_t
functions={}
for name,args in [('nvrtcCreateProgram',[C.POINTER(P),C.c_char_p,C.c_char_p,I,P,P]),('nvrtcCompileProgram',[P,I,C.POINTER(C.c_char_p)]),('nvrtcGetPTXSize',[P,C.POINTER(Z)]),('nvrtcGetPTX',[P,P]),('nvrtcDestroyProgram',[C.POINTER(P)]),('nvrtcGetProgramLogSize',[P,C.POINTER(Z)]),('nvrtcGetProgramLog',[P,P])]:
 f=getattr(nv,name);f.argtypes=args;f.restype=I;functions[name]=f
program=P();assert functions['nvrtcCreateProgram'](C.byref(program),source,b'fyodor_inference.cu',0,None,None)==0
options=[b'--gpu-architecture=compute_120',b'--std=c++11',b'--fmad=true',b'-DNYA_CUDA_FAST_MATH=1']
try:
 status=functions['nvrtcCompileProgram'](program,len(options),(C.c_char_p*len(options))(*options))
 n=Z();assert functions['nvrtcGetProgramLogSize'](program,C.byref(n))==0
 log=C.create_string_buffer(n.value);assert functions['nvrtcGetProgramLog'](program,log)==0;(out/'nvrtc.log').write_bytes(log.value)
 assert status==0,status
 assert functions['nvrtcGetPTXSize'](program,C.byref(n))==0;ptx=C.create_string_buffer(n.value);assert functions['nvrtcGetPTX'](program,ptx)==0
 (work/'kernels.ptx').write_bytes(ptx.value)
finally:assert functions['nvrtcDestroyProgram'](C.byref(program))==0
ptxas=root/'.tools/cuda-sdk/cuda_nvcc-windows-x86_64-13.1.115-archive/bin/ptxas.exe'
command=[str(ptxas),'-arch=sm_120','-O3','-sanitize=memcheck',str(work/'kernels.ptx'),'-o',str(work/'kernels.cubin')]
with (out/'ptxas.stdout').open('x') as stdout,(out/'ptxas.stderr').open('x') as stderr:subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
helper=r'''
/* Test build only: require exact runtime PTX before loading checked CUBIN. */
static int diagnostic_checked_module(char **ptx, size_t ptx_bytes)
{
    FILE *file=fopen("PTX_PATH", "rb");
    if (!file) return 1;
    int ok=ptx_bytes>0;
    for (size_t i=0;ok && i+1<ptx_bytes;++i) {
        int value=fgetc(file);
        if (value==EOF || (unsigned char)value!=(unsigned char)(*ptx)[i]) ok=0;
    }
    if (ok && fgetc(file)!=EOF) ok=0;
    if (ferror(file)) ok=0;
    if (fclose(file)!=0) ok=0;
    if (!ok) return 1;
    file=fopen("CUBIN_PATH", "rb");
    if (!file) return 1;
    if (fseek(file,0,SEEK_END)!=0) { fclose(file); return 1; }
    long length=ftell(file);
    if (length<=0 || length>64L*1024L*1024L || fseek(file,0,SEEK_SET)!=0) { fclose(file); return 1; }
    size_t bytes=(size_t)length;
    char *image=(char *)malloc(bytes);
    if (!image) { fclose(file); return 1; }
    ok=fread(image,1,bytes,file)==bytes && !ferror(file);
    if (fclose(file)!=0) ok=0;
    if (!ok) { free(image); return 1; }
    free(*ptx); *ptx=image;
    fprintf(stderr,"SANITIZER_MODULE compile_time_memcheck=1 exact_ptx_match=1 bytes=%zu\n",bytes);
    fflush(stderr);
    return 0;
}
'''.replace('PTX_PATH',(work/'kernels.ptx').as_posix()).replace('CUBIN_PATH',(work/'kernels.cubin').as_posix())
s=(work/'cuda.c').read_text();needle='    c->program_destroy(&program); program = NULL;';assert s.count(needle)==1
s=s.replace(needle,needle+'\n    if (training || c->reference || major!=12 || minor!=0 || diagnostic_checked_module(&ptx,ptx_bytes)) goto failure;')
# Place helper before all CUDA definitions, after standard headers have been included.
needle='#include <nvrtc.h>';assert s.count(needle)==1
s=s.replace(needle,needle+'\n#include <stdio.h>\n#include <stdlib.h>\n'+helper)
(work/'cuda.c').write_text(s)
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH']);commands=[command]
for original in proof['commands']:
 command=[arg.replace(str(base),str(work)) for arg in original];subprocess.run(command,env=env,check=True);commands.append(command)
assert len(commands)==5
for name,h in proof['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
paths=[Path(__file__),proof_path,library,ptxas,*work.iterdir()]
result=dict(build_passed=True,accepted_benchmark=False,options=[v.decode() for v in options],commands=commands,source_sha256=hashlib.sha256(source).hexdigest(),sha256={str(p.relative_to(root)) if p.is_relative_to(root) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(out/'build.json').write_text(json.dumps(result,indent=2));print('Strict C17/C23 instrumented model builds passed')
