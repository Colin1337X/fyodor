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
source=b'\n'.join(((Path(__file__).resolve().parent/'baseline-matvec.cu.out') if f=='matvec.cu' else root/'CUDA'/f).read_bytes() for f in ('matvec.cu','gemm.cu','resident.cu'))
text=(Path(__file__).resolve().parent/'baseline-matvec.cu.out').read_text()
helper=text[text.index('__device__ __forceinline__ void nya_row_dot_at('):text.index('#define NYA_DOT')]
helper=helper.replace('nya_row_dot_at(','nya_row_dot_add(').replace('output[row] = sum','output[row] = output[row]+sum')
source+=helper.encode()
for typ in (12,14):
 source+=('extern "C" __global__ void nya_add_'+str(typ)+'(const unsigned char *w,const float *x,float *y,unsigned long long c,unsigned long long s,unsigned long long r) { nya_row_dot_add(w,x,y,c,s,r,'+str(typ)+',(unsigned long long)blockIdx.x*8+(threadIdx.x>>5)); }\n').encode()
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
for name in ('nya_dot_12','nya_dot_14','nya_add_12','nya_add_14','nya_vector'):
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
random.seed(714);results=[]
try:
 for n,rows,typ in ((256,9,12),(768,65,14),(2048,2048,12),(5632,2048,12),(5632,2048,14),(11008,4096,14)):
  block=144 if typ==12 else 210;stride=n//256*block
  data=bytearray(random.randbytes(rows*stride))
  for off in range(0,len(data),block):
   if typ==12:data[off:off+4]=b'\x00\x28\x00\x24'
   else:data[off+208:off+210]=b'\x00\x28'
  weights=buffer(data);x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(n))).tobytes())
  initial=array.array('f',[12345.]+[random.uniform(-4,4) for _ in range(rows)]+[12345.]).tobytes()
  reference=buffer(initial);actual=buffer(initial);temporary=buffer(bytes(rows*4))
  def separate(out):
   call('nya_dot_'+str(typ),(rows+7)//8,[Q(weights),Q(x),Q(temporary),Q(n),Q(stride),Q(rows)])
   call('nya_vector',(rows+255)//256,[Q(out+4),Q(out+4),Q(temporary),Q(rows),U(1),C.c_float(1)])
  def fused():call('nya_add_'+str(typ),(rows+7)//8,[Q(weights),Q(x),Q(actual+4),Q(n),Q(stride),Q(rows)])
  separate(reference);check(sync());expected=read(reference,len(initial))
  timings={'separate':[],'fused':[]}
  for order in (list(timings),list(reversed(timings))):
   for name in order:
    host=C.create_string_buffer(initial);check(upload(Q(actual),host,len(initial)))
    invoke=(lambda:separate(actual)) if name=='separate' else fused
    invoke();check(sync());assert read(actual,len(initial))==expected,(n,rows,typ,name)
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
  row={'width':n,'rows':rows,'type':typ,'exact_including_guards':True,'timings_ms':timings,'mean_ms':means,'gain_percent':100*(means['separate']/means['fused']-1)}
  results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='timings_ms'}),flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(stream_destroy(stream));check(event_destroy(aevent));check(event_destroy(bevent));check(unload(module));check(release(device))
(out/'results.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'options':[x.decode() for x in opts],'cases':results},indent=2))
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
