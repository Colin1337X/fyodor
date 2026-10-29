"""Direct double oracle for the portable FlashAttention-style F32 kernel."""
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
source+=b'\n'+(root/'optcpp/attention/portable_f32.cu').read_bytes()
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

f=P();check(function(C.byref(f),module,b'nya_attention_portable_split_f32'))
combine=P();check(function(C.byref(combine),module,b'nya_attention_combine'))
import sys,subprocess,statistics
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
funcs={}
for name in ('nya_attention_portable_f32','nya_attention_portable_split_f32','nya_attention_split'):
 fn=P();check(function(C.byref(fn),module,name.encode()));funcs[name]=fn
stream=P();stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P]);check(stream_create(C.byref(stream),1))
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P]);graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P]);event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P]);event_destroy=bind(cu,'cuEventDestroy_v2',[P])
a=P();b=P();check(event_create(C.byref(a),0));check(event_create(C.byref(b),0))
random.seed(617);results=[]
try:
 for width,heads,kh,pos,window in ((256,16,8,1024,1024),(512,16,1,1024,0),(512,16,1,4096,0),(512,16,1,8192,0)):
  n=pos+1
  q=buffer(array.array('f',(random.uniform(-1,1) for _ in range(heads*width))).tobytes())
  k=buffer(array.array('f',(random.uniform(-1,1) for _ in range(n*kh*width))).tobytes());v=buffer(array.array('f',(random.uniform(-1,1) for _ in range(n*kh*width))).tobytes())
  o=buffer(bytes(heads*width*4));parts=buffer(bytes(heads*4*(width+2)*4));scores=buffer(bytes(heads*n*4));request=buffer(array.array('I',[0,pos]).tobytes());check(sync())
  samples={name:[] for name in funcs}
  for order in (list(funcs),list(reversed(funcs))):
   for name in order:
    split=name!='nya_attention_portable_f32'
    if name=='nya_attention_split':args=[Q(q),Q(k),Q(v),Q(parts),Q(scores),U(width),U(heads),U(kh),U(window),U(pos),U(n),C.c_float(1/math.sqrt(width)),Q(request)]
    else:args=[Q(q),Q(k),Q(v),Q(parts if split else o),U(width),U(heads),U(kh),U(window),U(pos),U(1),C.c_float(1/math.sqrt(width)),Q(request)]
    packed=(P*len(args))(*(C.cast(C.byref(x),P) for x in args));ca=[Q(parts),Q(o),U(width)];cp=(P*3)(*(C.cast(C.byref(x),P) for x in ca))
    graph=P();executable=P();check(begin(stream,0))
    for i in range(20):
     check(launch(funcs[name],heads,4 if split else 1,1,256,1,1,0,stream,packed,None))
     if split:check(launch(combine,heads,1,1,max(128,width),1,1,0,stream,cp,None))
    check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
    for i in range(3):check(graphlaunch(executable,stream))
    for repeat in range(5):
     check(event_record(a,stream))
     for i in range(10):check(graphlaunch(executable,stream))
     check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b));samples[name].append(ms.value/200)
    check(execdestroy(executable));check(graphdestroy(graph))
   check(sync())
  row={'width':width,'heads':heads,'kvheads':kh,'position':pos,'window':window,'samples_ms':samples,'mean_ms':{name:statistics.mean(values) for name,values in samples.items()}}
  results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
(out/'results.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'caveat':'Repeated warm KV bank, graph replay; diagnostic kernel timing, not model acceptance','cases':results},indent=2))
