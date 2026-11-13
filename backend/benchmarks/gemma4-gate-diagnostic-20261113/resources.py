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
def pack_q4(w):
 out=bytearray()
 for offset in range(0,len(w),144):
  b=w[offset:offset+144];sc=b[4:16];out.extend(b[:4])
  out.extend((sc[j]&63) if j<4 else (sc[j+4]&15)|((sc[j-4]>>6)<<4) for j in range(8))
  out.extend((sc[j+4]&63) if j<4 else (sc[j+4]>>4)|((sc[j]>>6)<<4) for j in range(8))
  out.extend(bytes(12));out.extend(b[16:])
 return out

def buffer(data):
 address=Q();check(alloc(C.byref(address),len(data)));owned.append(address)
 host=C.create_string_buffer(bytes(data));check(upload(address,host,len(data)))
 return address.value
def read(address,size):
 host=C.create_string_buffer(size);check(download(host,Q(address),size));return host.raw
source=b'\n'.join((root/'CUDA'/f).read_bytes() for f in ('matvec.cu','gemm.cu','resident.cu'))
source+=b'\n'+(root/'optcpp/matvec/byte_scale_dot.cu').read_bytes()+b'\n'+(root/'optcpp/matvec/gemma_gate.cu').read_bytes()
import sys
unroll=int(sys.argv[1]) if len(sys.argv)>1 else 8
assert unroll in (1,2,4,8)
prefix,gate=source.rsplit(b'#pragma unroll 8',1)
inner=int(sys.argv[2]) if len(sys.argv)>2 else 0
assert inner in (0,1)
if inner:gate=gate.replace(b'#pragma unroll\n',b'#pragma unroll 1\n').replace(b'#pragma unroll\r\n',b'#pragma unroll 1\r\n')
source=prefix+('#pragma unroll '+str(unroll)).encode()+gate
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
for name in ('nya_dot_byte_12','nya_vector','nya_gate_up_gemma_byte_3840'):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

def call(name,blocks,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,1,1,256,1,1,0,None,packed,None))

get_attribute=bind(cu,'cuFuncGetAttribute',[C.POINTER(I),I,P])
result={}
try:
 for name,func in funcs.items():
  result[name]={}
  for key,code in [('max_threads',0),('shared_bytes',1),('local_bytes',3),('registers_per_thread',4)]:
   value=I();check(get_attribute(C.byref(value),code,func));result[name][key]=value.value
 print(json.dumps({'unroll':unroll,'inner_unroll_disabled':bool(inner),'compute_capability':str(major)+str(minor),'source_sha256':hashlib.sha256(source).hexdigest(),'attributes':result},indent=2),flush=True)
finally:
 check(unload(module));check(release(device))
