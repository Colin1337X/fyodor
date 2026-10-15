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
variant=(root/'CUDA/matvec.cu').read_text().split('/* Decode-only Q4_K')[1]
for warps in (1,2,4,8,16):
 source+= ('\n/* Decode-only Q4_K'+variant.replace('nya_gate_up_q4',f'nya_gate_up_q4_w{warps}').replace('blockIdx.x * 8',f'blockIdx.x * {warps}')).encode()
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
for name in ('nya_dot_12','nya_vector','nya_gate_up_q4',*(f'nya_gate_up_q4_w{w}' for w in (1,2,4,8,16))):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

def call(name,blocks,args,threads=256):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,1,1,threads,1,1,0,None,packed,None))


import statistics, subprocess, datetime
out=Path(__file__).resolve().parent
(out/'processes-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P])
event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P])
event_destroy=bind(cu,'cuEventDestroy_v2',[P])
aevent=P();bevent=P();check(event_create(C.byref(aevent),0));check(event_create(C.byref(bevent),0))
random.seed(712);results=[]
try:
 for k,rows in ((2048,5632),(4096,11008),(8192,28672)):
  stride=k//256*144;weights=[]
  for unused in range(2):
   w=bytearray(random.randbytes(rows*stride))
   for offset in range(0,len(w),144):w[offset:offset+4]=b'\x00\x28\x00\x24'
   weights.append(buffer(w))
  x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k))).tobytes())
  guards=array.array('f',[12345.]*(rows+2)).tobytes();expected=buffer(guards);actual=buffer(guards)
  args=[Q(weights[0]),Q(weights[1]),Q(x),Q(expected+4),Q(k),Q(stride),Q(rows)]
  call('nya_gate_up_q4',(rows+7)//8,args);check(sync());reference=read(expected,len(guards))
  args[3]=Q(actual+4)
  timings={w:[] for w in (1,2,4,8,16)}
  for order in (list(timings),list(reversed(timings))):
   for warps in order:
    name=f'nya_gate_up_q4_w{warps}';blocks=(rows+warps-1)//warps
    for _ in range(10):call(name,blocks,args,warps*32)
    check(sync());assert read(actual,len(guards))==reference,(k,rows,warps)
    for repeat in range(5):
     check(event_record(aevent,None))
     for _ in range(50):call(name,blocks,args,warps*32)
     check(event_record(bevent,None));check(event_sync(bevent));ms=C.c_float();check(elapsed(C.byref(ms),aevent,bevent))
     timings[warps].append(ms.value/50)
  row={'columns':k,'rows':rows,'exact':True,'timings_ms':timings,'mean_ms':{w:statistics.mean(v) for w,v in timings.items()}}
  results.append(row);print(json.dumps(row),flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(event_destroy(aevent));check(event_destroy(bevent));check(unload(module));check(release(device))
(out/'results.json').write_text(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source).hexdigest(),'options':[x.decode() for x in opts],'caveat':'Diagnostic repeated kernel timing; includes launch gaps, hot weights; not end-to-end acceptance.','cases':results},indent=2))
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
