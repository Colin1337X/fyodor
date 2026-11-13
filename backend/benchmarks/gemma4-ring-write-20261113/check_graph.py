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
for name in ('nya_rope','nya_rope_ring','nya_decode_request'):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f
stream=None

def call(name,blocks,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,1,1,256,1,1,0,stream,packed,None))

def put(address,data):
 host=C.create_string_buffer(bytes(data));check(upload(Q(address),host,len(data)))

def floats(count,seed):return array.array('f',(((i*17+seed)%257-128)/256 for i in range(count))).tobytes()

stream=P();stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P]);check(stream_create(C.byref(stream),1))
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P]);graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
width=256;heads=2;slots=7;stride=width*heads
input_data=floats(stride,31);values=floats(stride,71)
freq=array.array('f',(10000**(-2*j/width) for j in range(width//2))).tobytes()
x=buffer(input_data);reference=buffer(input_data);v=buffer(values);frequencies=buffer(freq)
guards=array.array('f',[12345.]*(slots*stride+2)).tobytes();keys=buffer(guards);cachev=buffer(guards)
request=buffer(array.array('I',[0,0]).tobytes());check(sync())
args=[Q(x),Q(frequencies),U(width),U(heads),U(777),U(1),U(1),Q(request),Q(keys+4),Q(cachev+4),Q(v),U(slots)]
graph=P();executable=P();check(begin(stream,0));call('nya_rope_ring',(stride//2+255)//256,args)
check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
expectedk=bytearray(guards);expectedv=bytearray(guards);results=[]
try:
 for position in (0,1,6,7,8,14,63,0,6):
  put(x,input_data);put(reference,input_data)
  call('nya_rope',(stride//2+255)//256,[Q(reference),Q(frequencies),U(width),U(heads),U(position),U(1),U(1),Q(0),Q(0),Q(0),Q(0)])
  call('nya_decode_request',1,[Q(request),U(17),U(position)])
  check(graphlaunch(executable,stream));check(sync())
  rotated=read(reference,stride*4);assert read(x,stride*4)==rotated
  target=4+(position%slots)*stride*4
  expectedk[target:target+stride*4]=rotated;expectedv[target:target+stride*4]=values
  assert read(keys,len(guards))==expectedk and read(cachev,len(guards))==expectedv
  results.append(dict(position=position,bit_exact=True,graph_reused=True))
finally:
 check(execdestroy(executable));check(graphdestroy(graph));check(stream_destroy(stream))
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
print(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),cases=results),indent=2))
