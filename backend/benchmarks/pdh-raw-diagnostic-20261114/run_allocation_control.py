"""Large-allocation request-update kernel: idle/direct/graph diagnostic, no acceptance."""
import ctypes as C,hashlib,json,os,struct,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
if len(sys.argv)!=2 or not sys.argv[1].isascii() or not sys.argv[1].isalnum():raise SystemExit('Supply unique alphanumeric label')
proof_path=HERE/'request-compile/compile.json';proof=json.loads(proof_path.read_text())
if proof.get('compile_passed') is not True:raise ValueError('Missing compilation proof')
hashes={}
for name,h in proof['sha256'].items():
 p=ROOT/name
 if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Changed compile input: '+name)
 hashes[str(p.resolve())]=h
sys.path.insert(0,str(HERE.parent/'gemma4-qkv-20261029'));import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):raise SystemExit('Another GPU workload remains')
folder=HERE/sys.argv[1];folder.mkdir()
for p in (Path(__file__),proof_path,Path(environment.__file__)):hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(accepted_benchmark=False,passed=False,process_id=os.getpid(),phases=[],limitation='12 GiB initialized allocation plus request-kernel control; allocation is not proof of physical residency, causal System attribution or model performance.')
lib=C.WinDLL('nvcuda.dll');P=C.c_void_p;I=C.c_int;U=C.c_uint;Z=C.c_size_t;Q=C.c_uint64

def bind(name,args):
 f=getattr(lib,name);f.argtypes=args;f.restype=I;return f

def check(status):
 if status:raise RuntimeError('CUDA error '+str(status))

init=bind('cuInit',[U]);deviceget=bind('cuDeviceGet',[C.POINTER(I),I]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(P),I]);release=bind('cuDevicePrimaryCtxRelease_v2',[I]);current=bind('cuCtxSetCurrent',[P])
alloc=bind('cuMemAlloc_v2',[C.POINTER(Q),Z]);free=bind('cuMemFree_v2',[Q]);upload=bind('cuMemcpyHtoD_v2',[Q,P,Z]);download=bind('cuMemcpyDtoH_v2',[P,Q,Z])
load=bind('cuModuleLoadData',[C.POINTER(P),P]);unload=bind('cuModuleUnload',[P]);get=bind('cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p]);launch=bind('cuLaunchKernel',[P,U,U,U,U,U,U,U,P,C.POINTER(P),P])
streamcreate=bind('cuStreamCreate',[C.POINTER(P),U]);streamfree=bind('cuStreamDestroy_v2',[P]);sync=bind('cuStreamSynchronize',[P])
begin=bind('cuStreamBeginCapture_v2',[P,I]);end=bind('cuStreamEndCapture',[P,C.POINTER(P)]);instantiate=bind('cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind('cuGraphLaunch',[P,P]);graphfree=bind('cuGraphDestroy',[P]);execfree=bind('cuGraphExecDestroy',[P])
pressure=Q();pressure_bytes=12*1024**3
meminfo=bind('cuMemGetInfo_v2',[C.POINTER(Z),C.POINTER(Z)])
memset=bind('cuMemsetD8_v2',[Q,C.c_ubyte,Z]);ctxsync=bind('cuCtxSynchronize',[])
device=I();context=P();gpu=Q();module=P();function=P();stream=P();graph=P();executable=P()
payload=bytes(range(24));host=C.create_string_buffer(payload);output=C.create_string_buffer(24)
try:
 check(init(0));check(deviceget(C.byref(device),0));check(retain(C.byref(context),device));check(current(context))
 available=Z();total=Z();check(meminfo(C.byref(available),C.byref(total)))
 result['allocation_control']=dict(bytes=pressure_bytes,free_before=available.value,total=total.value)
 if available.value < pressure_bytes+2*1024**3:raise RuntimeError('Insufficient free memory for control plus 2 GiB reserve')
 check(alloc(C.byref(pressure),pressure_bytes));check(memset(pressure,0xa5,pressure_bytes));check(ctxsync())
 check(meminfo(C.byref(available),C.byref(total)));result['allocation_control']['free_after']=available.value
 check(alloc(C.byref(gpu),24));check(upload(gpu,host,24));check(streamcreate(C.byref(stream),1))
 ptx=C.create_string_buffer((ROOT/'build-cuda/request-control-20261115/module.ptx').read_bytes());check(load(C.byref(module),ptx));check(get(C.byref(function),module,b'nya_decode_request'))
 address=Q(gpu.value+8);token=U(123);position=U(456);args=(P*3)(*(C.cast(C.byref(v),P) for v in (address,token,position)))
 def direct():check(launch(function,1,1,1,1,1,1,0,stream,args,None))
 check(begin(stream,0));direct();check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
 time.sleep(2)
 for round_,order in enumerate((('idle','direct','graph'),('graph','direct','idle'),('idle','direct','graph'))):
  for mode in order:
   phase=dict(round=round_,mode=mode,start=time.perf_counter(),launches=0)
   while time.perf_counter()<phase['start']+6:
    if mode!='idle':
     if mode=='direct':direct()
     else:check(graphlaunch(executable,stream))
     check(sync(stream));phase['launches']+=1
    time.sleep(.02)
   phase['end']=time.perf_counter()
   check(download(output,gpu,24))
   expected=payload if round_==0 and mode=='idle' else payload[:8]+struct.pack('<II',123,456)+payload[16:]
   if output.raw!=expected:raise ValueError('Request output/guard mismatch')
   guard=C.create_string_buffer(8)
   for offset in (0,pressure_bytes-8):
    check(download(guard,Q(pressure.value+offset),8))
    if guard.raw!=bytes([0xa5])*8:raise ValueError('Large allocation sentinel mismatch')
   phase['content_verified']=True;phase['allocation_sentinels_verified']=True;result['phases'].append(phase)
   with (folder/'phases.jsonl').open('a') as f:f.write(json.dumps(phase)+'\n')
   print(round_,mode,phase['launches'],flush=True)
 result['passed']=True
except BaseException as error:
 result['error']=str(error);raise
finally:
 errors=[]
 for address,operation in ((executable,execfree),(graph,graphfree),(stream,streamfree),(module,unload),(gpu,free),(pressure,free)):
  if address.value:
   status=operation(address)
   if status:errors.append(status)
 if context.value:
  status=release(device)
  if status:errors.append(status)
 result['cleanup_errors']=errors
 result['source_unchanged']=all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in hashes.items())
 result['passed']=result['passed'] and not errors and result['source_unchanged']
 log=folder/'phases.jsonl'
 if log.exists():hashes[str(log.resolve())]=hashlib.sha256(log.read_bytes()).hexdigest()
 result['sha256']=hashes;(folder/'result.json').write_text(json.dumps(result,indent=2))
