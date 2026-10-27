"""Compare native Q6 row dot with signed integer-product candidate."""
import ctypes as C, hashlib, json, os, sys
dtype=int(sys.argv[1]) if len(sys.argv)>1 else 12
assert dtype in (12,14)
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
source=b'\n'.join(((Path(__file__).resolve().parent/'baseline-matvec.cu.out') if f=='matvec.cu' else root/'CUDA'/f).read_bytes() for f in ('matvec.cu','gemm.cu','resident.cu'))
baseline=(Path(__file__).resolve().parent/'baseline-matvec.cu.out').read_text()
source+=b'\n'+(root/'optcpp/matvec/integer_scale_q6.cu').read_bytes()
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
for name in ('nya_dot_14','nya_dot_q6_integer_14'):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

stream=P()
def call(name,blocks,args,threads=256):
 if name=='fixed': name='nya_dot_q6_integer_14'
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,1,1,threads,1,1,0,stream,packed,None))


import statistics, subprocess, datetime
out=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else Path(__file__).resolve().parent/f'type-{dtype}'
out.mkdir(exist_ok=False)
(out/'processes-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P])
event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P])
event_destroy=bind(cu,'cuEventDestroy_v2',[P])
aevent=P();bevent=P();check(event_create(C.byref(aevent),0));check(event_create(C.byref(bevent),0))
stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P])
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P])
graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
check(stream_create(C.byref(stream),1))
funcattr=bind(cu,'cuFuncGetAttribute',[C.POINTER(I),I,P]);resources={}
for name,f in funcs.items():
 row={}
 for label,attr in (('registers',4),('local_bytes',3)):
  value=I();check(funcattr(C.byref(value),attr,f));row[label]=value.value
 resources[name]=row
(out/'kernel-resources.json').write_text(json.dumps(resources,indent=2))
random.seed(712);results=[]
try:
 for k,rows in (((2304,257),(4096,11008),(8192,8192)) if len(sys.argv)>3 and sys.argv[3]=='extended' else ((256,9),(768,65),(2048,256),(2048,2048),(5632,2048),(2048,32000))):
  blockbytes=144 if dtype==12 else 210
  stride=k//256*blockbytes;weights=[]
  for unused in range(2):
   w=bytearray(random.randbytes(rows*stride))
   for offset in range(0,len(w),blockbytes):
    if dtype==12:
     bits=(0,1,0x3ff,0x400,0x3555,0x3c00,0x7bff,0x8000,0x8001,0x83ff,0x8400,0xb555,0xbc00,0xfbff)
     block=offset//blockbytes;w[offset:offset+4]=struct.pack('<HH',bits[block%14],bits[(block*5+3)%14])
    else:
     bits=(0,1,0x3ff,0x400,0x3555,0x3c00,0x7bff,0x8000,0x8001,0x83ff,0x8400,0xb555,0xbc00,0xfbff)
     w[offset+208:offset+210]=struct.pack('<H',bits[(offset//blockbytes)%14])
   weights.append(buffer(w))
  x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k))).tobytes())
  guards=array.array('f',[12345.]*(rows+2)).tobytes();expected=buffer(guards);actual=buffer(guards)
  check(sync()) # Order all pageable uploads before nonblocking stream work.
  args=[Q(weights[0]),Q(x),Q(expected+4),Q(k),Q(stride),Q(rows)]
  call(f'nya_dot_{dtype}',(rows+7)//8,args);check(sync());reference=read(expected,len(guards))
  args[2]=Q(actual+4)
  timings={w:[] for w in (f'nya_dot_{dtype}','fixed')}
  for order in (list(timings),list(reversed(timings))):
   for warps in order:
    name=warps;blocks=(rows+7)//8
    for _ in range(10):call(name,blocks,args,256)
    check(sync());assert read(actual,len(guards))==reference,(k,rows,warps)
    graph=P();executable=P();check(begin(stream,0))
    for _ in range(50):call(name,blocks,args,256)
    check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
    for _ in range(3):check(graphlaunch(executable,stream))
    for repeat in range(5):
     check(event_record(aevent,stream))
     for _ in range(10):check(graphlaunch(executable,stream))
     check(event_record(bevent,stream));check(event_sync(bevent));ms=C.c_float();check(elapsed(C.byref(ms),aevent,bevent))
     timings[warps].append(ms.value/500)
    check(execdestroy(executable));check(graphdestroy(graph))
  row={'columns':k,'rows':rows,'exact':True,'timings_ms':timings,'mean_ms':{w:statistics.mean(v) for w,v in timings.items()}}
  results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='timings_ms'}),flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(stream_destroy(stream));check(event_destroy(aevent));check(event_destroy(bevent));check(unload(module));check(release(device))
(out/'results.json').write_text(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source).hexdigest(),'options':[x.decode() for x in opts],'caveat':'Diagnostic repeated kernel timing; CUDA graph replay, hot weights; not end-to-end acceptance.','cases':results},indent=2))
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
