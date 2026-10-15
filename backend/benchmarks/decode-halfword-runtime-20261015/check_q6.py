"""Check fused Q4 gate/up against the original three production kernels."""
import ctypes as C, hashlib, json, os
from pathlib import Path
import array,random,struct,math
root=Path(__file__).resolve().parents[3]
p=root/'.tools/cuda-sdk/cuda_nvrtc-windows-x86_64-13.1.115-archive/bin/x64'
loader=os.add_dll_directory(str(p))
builtins=C.CDLL(str(p/'nvrtc-builtins64_131.dll'))
nv=C.CDLL(str(p/'nvrtc64_130_0.dll'));cu=C.WinDLL('nvcuda.dll')
P=C.c_void_p; I=C.c_int; U=C.c_uint; Z=C.c_size_t; Q=C.c_uint64
def bind(lib,name,args):
 f=getattr(lib,name);f.argtypes=args;f.restype=I;return f
def check(status):
 assert status==0,status
create=bind(nv,'nvrtcCreateProgram',[C.POINTER(P),C.c_char_p,C.c_char_p,I,P,P])
compile=bind(nv,'nvrtcCompileProgram',[P,I,C.POINTER(C.c_char_p)])
size=bind(nv,'nvrtcGetPTXSize',[P,C.POINTER(Z)])
get=bind(nv,'nvrtcGetPTX',[P,P]);destroy=bind(nv,'nvrtcDestroyProgram',[C.POINTER(P)])
load=bind(cu,'cuModuleLoadData',[C.POINTER(P),P]);unload=bind(cu,'cuModuleUnload',[P])
function=bind(cu,'cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p])
launch=bind(cu,'cuLaunchKernel',[P,U,U,U,U,U,U,U,P,C.POINTER(P),P])
init=bind(cu,'cuInit',[U]);deviceget=bind(cu,'cuDeviceGet',[C.POINTER(I),I])
retain=bind(cu,'cuDevicePrimaryCtxRetain',[C.POINTER(P),I]);release=bind(cu,'cuDevicePrimaryCtxRelease_v2',[I])
current=bind(cu,'cuCtxSetCurrent',[P]);sync=bind(cu,'cuCtxSynchronize',[])
attribute=bind(cu,'cuDeviceGetAttribute',[C.POINTER(I),I,I])
alloc=bind(cu,'cuMemAlloc_v2',[C.POINTER(Q),Z]);free=bind(cu,'cuMemFree_v2',[Q])
upload=bind(cu,'cuMemcpyHtoD_v2',[Q,P,Z]);download=bind(cu,'cuMemcpyDtoH_v2',[P,Q,Z])
check(init(0));device=I();check(deviceget(C.byref(device),0));context=P();check(retain(C.byref(context),device))
check(current(context));major=I();minor=I();check(attribute(C.byref(major),75,device));check(attribute(C.byref(minor),76,device))
major,minor=major.value,minor.value
owned=[]
def buffer(data):
 address=Q();check(alloc(C.byref(address),len(data)));owned.append(address)
 host=C.create_string_buffer(bytes(data));check(upload(address,host,len(data)))
 return address.value
def read(address,size):
 host=C.create_string_buffer(size);check(download(host,Q(address),size));return host.raw
source=b'\n'.join((root/'CUDA'/f).read_bytes() for f in ('matvec.cu','gemm.cu','resident.cu'))
baseline=(Path(__file__).resolve().parent/'baseline-matvec.cu.out').read_text()
old=baseline[baseline.index('__device__ __forceinline__ void nya_row_dot'):baseline.index('#define NYA_DOT')].replace('nya_row_dot','nya_row_dot_old')
source+=old.encode()+b'\nextern "C" __global__ void nya_dot_old(const unsigned char *w,const float *x,float *y,unsigned long long c,unsigned long long s,unsigned long long r) { nya_row_dot_old(w,x,y,c,s,r,14); }\n'
program=P();module=P()
check(create(C.byref(program),source,b'gate_pair_test.cu',0,None,None))
opts=[f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++11',b'--fmad=true',b'-DNYA_CUDA_FAST_MATH=1']
try:
 status=compile(program,len(opts),(C.c_char_p*len(opts))(*opts))
 if status:
  logs=bind(nv,'nvrtcGetProgramLogSize',[P,C.POINTER(Z)])
  logget=bind(nv,'nvrtcGetProgramLog',[P,P]);n=Z();check(logs(program,C.byref(n)))
  log=C.create_string_buffer(n.value);check(logget(program,log));print(log.value.decode(),flush=True)
 check(status)
 n=Z();check(size(program,C.byref(n)));ptx=C.create_string_buffer(n.value);check(get(program,ptx))
 check(load(C.byref(module),ptx))
finally:check(destroy(C.byref(program)))
funcs={}
for name in ('nya_dot_14','nya_dot_old'):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

def call(name,blocks,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,1,1,256,1,1,0,None,packed,None))

random.seed(712);results=[]
try:
 for k in (256,768,2048,2304,8192):
  for rows in (1,7,8,9,63,64,65):
   stride=k//256*210
   weights=[]
   for unused in range(2):
    w=bytearray(random.randbytes(rows*stride))
    for offset in range(0,len(w),210):w[offset+208:offset+210]=b'\x00\x28'
    weights.append(buffer(w))
   x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k))).tobytes())
   g=buffer(bytes(rows*4));u=buffer(bytes(rows*4))
   guards=array.array('f',[12345.]*(rows+2)).tobytes()
   expected=buffer(guards);actual=buffer(guards)
   call('nya_dot_old',(rows+7)//8,[Q(weights[0]),Q(x),Q(expected+4),Q(k),Q(stride),Q(rows)])
   call('nya_dot_14',(rows+7)//8,[Q(weights[0]),Q(x),Q(actual+4),Q(k),Q(stride),Q(rows)])
   check(sync());a=read(actual,(rows+2)*4);b=read(expected,(rows+2)*4)
   assert a==b,(k,rows)
   assert a[:4]==guards[:4] and a[-4:]==guards[-4:]
   assert all(math.isfinite(v) for v in struct.unpack('<'+'f'*(rows+2),a))
   results.append({'columns':k,'rows':rows,'exact_match_including_canaries':True})
   for address in owned:check(free(address))
   owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
print(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'options':[x.decode() for x in opts],'cases':results},indent=2))
