"""Check fused Q4 gate/up against the original three production kernels."""
import ctypes as C, hashlib, json, os
from pathlib import Path
from compact_source import compact_source
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
original=(root/'optcpp/matvec/gemma_gate.cu').read_text()
variants={'u8b4':(8,4),'u1b4':(1,4),**{f'compact{u}':(u,4) for u in (1,4,8,15)}}
source=b'\n'.join((root/'CUDA'/f).read_bytes() for f in ('matvec.cu','gemm.cu','resident.cu'))
source+=b'\n'+(root/'optcpp/matvec/byte_scale_dot.cu').read_bytes()
for name,(unroll,blocks) in variants.items():
 candidate=original.replace('nya_gate_up_gemma_byte_3840','gate_'+name).replace('__launch_bounds__(256,4)',f'__launch_bounds__(256,{blocks})').replace('#pragma unroll 8',f'#pragma unroll {unroll}')
 if name.startswith('compact'):
  candidate=compact_source(root).replace('nya_gate_up_gemma_compact_3840','gate_'+name).replace('#pragma unroll 1',f'#pragma unroll {unroll}')
 source+=b'\n'+candidate.encode()
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
for name in ['nya_dot_byte_12','nya_vector',*('gate_'+v for v in variants)]:
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

def call(name,blocks,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,1,1,256,1,1,0,stream,packed,None))

stream=P();stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P]);check(stream_create(C.byref(stream),1))
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P]);graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P]);event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P]);event_destroy=bind(cu,'cuEventDestroy_v2',[P])
a=P();b=P();check(event_create(C.byref(a),0));check(event_create(C.byref(b),0))

import statistics,sys
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
before=environment.snapshot()
random.seed(712);rows=15360;k=3840;weights=[];compact=[]
for unused in range(2):
 w=bytearray(random.randbytes(rows*(k//256)*144))
 for offset in range(0,len(w),144):w[offset:offset+4]=b'\x00\x28\x00\x24'
 weights.append(buffer(pack_q4(w)));compact.append(buffer(w))
x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k))).tobytes())
g=buffer(bytes(rows*4));u=buffer(bytes(rows*4));output=buffer(bytes(rows*4));check(sync())
samples={name:[] for name in ['separate',*variants]};snapshots=[]
get_function_attribute=bind(cu,'cuFuncGetAttribute',[C.POINTER(I),I,P])
resources={}
for name in variants:
 attrs={}
 for label,index in [('registers',4),('local_bytes',3),('max_threads',0)]:
  value=I();check(get_function_attribute(C.byref(value),index,funcs['gate_'+name]));attrs[label]=value.value
 resources[name]=attrs
# Compare all outputs before any timing. Loop scheduling must preserve arithmetic.
for w,y in zip(weights,(g,u)):
 call('nya_dot_byte_12',(rows+7)//8,[Q(w),Q(x),Q(y),Q(k),Q(k//256*160),Q(rows)])
call('nya_vector',(rows+255)//256,[Q(output),Q(g),Q(u),Q(rows),U(3),C.c_float(1)])
check(sync());reference=read(output,rows*4)
assert all(math.isfinite(v) for v in struct.unpack('<'+'f'*rows,reference))
for name in variants:
 selected=compact if name.startswith('compact') else weights
 call('gate_'+name,(rows+7)//8,[Q(selected[0]),Q(selected[1]),Q(x),Q(output),Q(k),Q(k//256*160),Q(rows)])
 check(sync());assert read(output,rows*4)==reference,name
print('All compact/packed scheduling variants match separate projections exactly',flush=True)
try:
 for order in (list(samples),list(reversed(samples))):
  for name in order:
   graph=P();executable=P();check(begin(stream,0))
   for i in range(20):
    if name=='separate':
     for w,y in zip(weights,(g,u)):
      call('nya_dot_byte_12',(rows+7)//8,[Q(w),Q(x),Q(y),Q(k),Q(k//256*160),Q(rows)])
     call('nya_vector',(rows+255)//256,[Q(output),Q(g),Q(u),Q(rows),U(3),C.c_float(1)])
    else:
     kernel='gate_'+name
     selected=compact if name.startswith('compact') else weights
     call(kernel,(rows+7)//8,[Q(selected[0]),Q(selected[1]),Q(x),Q(output),Q(k),Q(k//256*160),Q(rows)])
   check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
   for i in range(3):check(graphlaunch(executable,stream))
   for repeat in range(5):
    check(event_record(a,stream))
    for i in range(10):check(graphlaunch(executable,stream))
    check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b));samples[name].append(ms.value/200)
   check(execdestroy(executable));check(graphdestroy(graph));check(sync())
   snapshots.append(environment.snapshot());print(name,statistics.mean(samples[name]),flush=True)
finally:
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
result={'accepted':False,'bit_exact_all_variants':True,'resources':resources,'variants':variants,'caveat':'Contaminated desktop; warm repeated weight banks, synthetic kernel diagnostic only, no model performance claim','source_sha256':hashlib.sha256(source).hexdigest(),'rows':rows,'columns':k,'samples_ms':samples,'means_ms':{name:statistics.mean(v) for name,v in samples.items()},'sample_stddev_ms':{name:statistics.stdev(v) for name,v in samples.items()},'environment':{'before':before,'after_each_variant':snapshots}}
Path(__file__).with_name('probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='environment'},indent=2))
