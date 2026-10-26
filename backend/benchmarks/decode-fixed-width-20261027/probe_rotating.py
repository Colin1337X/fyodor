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
baseline=(Path(__file__).resolve().parent/'baseline-matvec.cu.out').read_text()
variant=baseline[baseline.index('extern "C" __global__ __launch_bounds__(256,4) void nya_gate_up_q4'):baseline.index('/* Decode Q/K/V')]
source+=('\n'+variant.replace('nya_gate_up_q4','nya_gate_up_q4_w1')).encode()
source+=b'\n'+(root/'optcpp/matvec/fixed_width_gate.cu').read_bytes()
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
for name in ('nya_dot_12','nya_vector','nya_gate_up_q4','nya_gate_up_q4_w1',*(f'nya_gate_up_q4_w2_{k}' for k in (256,768,2048,2304,4096,8192))):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

stream=P()
def call(name,blocks,args,threads=256):
 if name=='nya_gate_up_q4_w2': name+=f'_{args[-3].value}'
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
random.seed(712);results=[]
try:
 for k,rows in ((2048,5632),(4096,11008)):
  stride=k//256*144;weights=[]
  bank_count=max(8,(64*1024*1024+2*rows*stride-1)//(2*rows*stride))
  for unused in range(2*bank_count):
   w=bytearray(random.randbytes(rows*stride))
   for offset in range(0,len(w),144):w[offset:offset+4]=b'\x00\x28\x00\x24'
   weights.append(buffer(w))
  x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k))).tobytes())
  guards=array.array('f',[12345.]*(rows+2)).tobytes();expected=buffer(guards);actual=buffer(guards)
  check(sync())
  args=[Q(weights[0]),Q(weights[1]),Q(x),Q(expected+4),Q(k),Q(stride),Q(rows)]
  call('nya_gate_up_q4',(rows+7)//8,args);check(sync());reference=read(expected,len(guards))
  args[3]=Q(actual+4)
  timings={w:[] for w in (1,2)}
  for order in (list(timings),list(reversed(timings))):
   for warps in order:
    name=f'nya_gate_up_q4_w{warps}';blocks=(rows+7)//8
    for bank in range(bank_count):
     args[0]=Q(weights[2*bank]);args[1]=Q(weights[2*bank+1]);args[3]=Q(expected+4)
     call('nya_gate_up_q4',blocks,args);check(sync());reference=read(expected,len(guards))
     args[3]=Q(actual+4);call(name,blocks,args);check(sync())
     assert read(actual,len(guards))==reference,(k,rows,warps,bank)
    graph=P();executable=P();check(begin(stream,0))
    for index in range(50):
     bank=index%bank_count;args[0]=Q(weights[2*bank]);args[1]=Q(weights[2*bank+1]);call(name,blocks,args,256)
    check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
    for _ in range(3):check(graphlaunch(executable,stream))
    for repeat in range(5):
     check(event_record(aevent,stream))
     for _ in range(10):check(graphlaunch(executable,stream))
     check(event_record(bevent,stream));check(event_sync(bevent));ms=C.c_float();check(elapsed(C.byref(ms),aevent,bevent))
     timings[warps].append(ms.value/500)
    check(execdestroy(executable));check(graphdestroy(graph))
  row={'banks':bank_count,'weight_bank_bytes':2*bank_count*rows*stride,'columns':k,'rows':rows,'exact':True,'timings_ms':timings,'mean_ms':{w:statistics.mean(v) for w,v in timings.items()}}
  results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='timings_ms'}),flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(stream_destroy(stream));check(event_destroy(aevent));check(event_destroy(bevent));check(unload(module));check(release(device))
(out/'results.json').write_text(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source).hexdigest(),'options':[x.decode() for x in opts],'caveat':'Diagnostic repeated kernel timing; CUDA graph replay, rotating weight banks; not end-to-end acceptance.','cases':results},indent=2))
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
