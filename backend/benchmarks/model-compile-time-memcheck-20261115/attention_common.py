"""Compile baseline and wide value attention kernels for comparison."""
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
import re
header=(root/'build-cuda/model-compile-time-memcheck-v2-20261115/cuda_source.h').read_text()
match=re.search(r'nya_cuda_source\[\]\s*=\s*\{([^}]+)\};',header);assert match
source=bytes(int(v.strip(),0) for v in match[1].split(',') if v.strip());assert source[-1]==0
for name in ('wide_value16.cu','wide_batch_split16.cu','wide_batch_cached_query.cu'):
 assert (root/'optcpp/attention'/name).read_bytes() in source,name
module=P();cubin=C.create_string_buffer((root/'build-cuda/model-compile-time-memcheck-v2-20261115/kernels.cubin').read_bytes())
check(load(C.byref(module),cubin))
funcs={}
for name in ('nya_attention_vec16','nya_attention_vec16_ring','nya_attention_vec16_split','nya_attention_vec16_ring_split','nya_attention_batch_split16_split','nya_attention_batch_split16_ring_split','nya_attention_combine','nya_attention_cached_query_split','nya_attention_cached_query_ring_split'):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

def call(name,gx,gy,bx,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],gx,gy,1,bx,1,1,0,None,packed,None))

def floats(count,seed):
 return array.array('f',(((i*17+seed)%257-128)/256 for i in range(count)))
