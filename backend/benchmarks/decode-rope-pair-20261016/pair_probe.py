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
variant=(Path(__file__).resolve().parent/'candidate-template.cu.out').read_text()
for shared in (0,1):
 setup='__shared__ float cs[256],sn[256]; if(threadIdx.x<half) { cs[threadIdx.x]=cosf((float)position*frequencies[threadIdx.x]);sn[threadIdx.x]=sinf((float)position*frequencies[threadIdx.x]); } __syncthreads();' if shared else ''
 trig='float co=cs[pair],si=sn[pair];' if shared else 'float co=cosf((float)position*frequencies[pair]);float si=sinf((float)position*frequencies[pair]);'
 source+=variant.replace('MODE',str(shared)).replace('SHARED_SETUP',setup).replace('TRIG',trig).encode()
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
for name in ('nya_rope','nya_rope_pair_0','nya_rope_pair_1'):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f
stream=P()
def call(name,blocks,args,threads=256):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,1,1,threads,1,1,0,stream,packed,None))


import statistics, subprocess, datetime
import sys
out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent/'graphs'
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
funcattr=bind(cu,'cuFuncGetAttribute',[C.POINTER(I),I,P])
resources={}
for name,f in funcs.items():
 row={}
 for label,attr in (('registers',4),('local_bytes',3)):
  value=I();check(funcattr(C.byref(value),attr,f));row[label]=value.value
 resources[name]=row
(out/'kernel-resources.json').write_text(json.dumps(resources,indent=2))
device_limits={}
for label,attr in (('multiprocessors',16),('threads_per_multiprocessor',39),('registers_per_multiprocessor',82)):
 value=I();check(attribute(C.byref(value),attr,device));device_limits[label]=value.value
(out/'device-limits.json').write_text(json.dumps(device_limits,indent=2))
random.seed(715);results=[]
try:
 for width,qh,kh,pos,request_mode in ((64,32,4,0,False),(64,32,4,257,True),(128,32,8,1023,True),(256,32,8,257,False),(80,3,1,17,True),(512,8,2,63,True)):
  half=width//2;freq=buffer(array.array('f',(random.uniform(.001,1) for _ in range(half))).tobytes())
  request=buffer(array.array('I',[0,pos]).tobytes()) if request_mode else 0
  original=[];addresses=[]
  for n in (qh*width,kh*width,(pos+1)*kh*width,(pos+1)*kh*width):
   data=array.array('f',[12345.]+[random.uniform(-3,3) for _ in range(n)]+[12345.]).tobytes();original.append(data);addresses.append(buffer(data))
  q,k,ck,cv=[a+4 for a in addresses];value=buffer(array.array('f',(random.uniform(-3,3) for _ in range(kh*width))).tobytes())
  def reset():
   for a,data in zip(addresses,original):host=C.create_string_buffer(data);check(upload(Q(a),host,len(data)))
  def separate():
   for x,heads,key_cache,value_cache in ((k,kh,ck,cv),(q,qh,0,0)):
    call('nya_rope',(heads*half+255)//256,[Q(x),Q(freq),U(width),U(heads),U(0 if request_mode else pos),U(0),U(1),Q(request),Q(key_cache),Q(value_cache),Q(value)])
  args=[Q(q),Q(k),Q(freq),U(width),U(qh),U(kh),U(0 if request_mode else pos),Q(request),Q(ck),Q(cv),Q(value)]
  def fused(shared):call('nya_rope_pair_'+str(shared),((qh+kh)*half+255)//256,args)
  reset();separate();check(sync());expected=[read(a,len(d)) for a,d in zip(addresses,original)]
  timings={'separate':[],'pair':[]}
  for order in (list(timings),list(reversed(timings))):
   for name in order:
    invoke=separate if name=='separate' else lambda:fused(int(name=='shared'))
    reset();invoke();check(sync());assert [read(a,len(d)) for a,d in zip(addresses,original)]==expected,(width,qh,kh,pos,name)
    graph=P();executable=P();check(begin(stream,0))
    for _ in range(50):invoke()
    check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
    for _ in range(3):check(graphlaunch(executable,stream))
    for repeat in range(5):
     check(event_record(aevent,stream))
     for _ in range(10):check(graphlaunch(executable,stream))
     check(event_record(bevent,stream));check(event_sync(bevent));ms=C.c_float();check(elapsed(C.byref(ms),aevent,bevent))
     timings[name].append(ms.value/500)
    check(execdestroy(executable));check(graphdestroy(graph))
  means={k:statistics.mean(v) for k,v in timings.items()}
  row={'width':width,'query_heads':qh,'kv_heads':kh,'position':pos,'request':request_mode,'exact_including_cache_and_guards':True,'timings_ms':timings,'mean_ms':means,'gain_percent':{k:100*(means['separate']/v-1) for k,v in means.items()}}
  results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='timings_ms'}),flush=True)
  for a in owned:check(free(a))
  owned.clear()
finally:
 for a in owned:check(free(a))
 check(stream_destroy(stream));check(event_destroy(aevent));check(event_destroy(bevent));check(unload(module));check(release(device))
(out/'results.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'options':[x.decode() for x in opts],'cases':results},indent=2))
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
