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

import sys
split_mode=len(sys.argv)<3 or sys.argv[2]!='unsplit'
f=P();check(function(C.byref(f),module,b'nya_attention_portable_split_f32' if split_mode else b'nya_attention_portable_f32'))
combine=P();check(function(C.byref(combine),module,b'nya_attention_combine'))
import sys,subprocess
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
(out/'processes-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
random.seed(716);results=[]
try:
 for width,heads,kh,pos,count,window,extreme in ((512,4,1,0,1,0,False),(257,4,2,65,1,3,False),(512,4,1,8192,1,0,False),(512,4,1,1024,1,0,False),(257,4,2,65,3,17,False),(512,4,2,130,2,67,True),(64,4,1,0,1,0,False),(80,6,2,65,3,0,False),(128,4,2,257,1,67,False),(256,4,1,1024,1,0,False),(64,2,1,129,2,1,False),(64,4,2,130,1,0,True)):
  if split_mode and count!=1:continue
  length=pos+count;scale=1/math.sqrt(width)
  def floats(n):return array.array('f',(random.uniform(-1,1) for _ in range(n)))
  q=floats(count*heads*width);k=floats(length*kh*width);v=floats(length*kh*width)
  if extreme:
   q=array.array('f',[100.]*len(q));k=array.array('f',[(100. if i//(kh*width)%2 else -100.) for i in range(len(k))])
  expected=[]
  for query in range(count):
   end=pos+query;first=max(0,end-window+1) if window else 0
   for head in range(heads):
    keyhead=head//(heads//kh);qb=(query*heads+head)*width
    scores=[sum(q[qb+j]*k[(t*kh+keyhead)*width+j] for j in range(width))*scale for t in range(first,end+1)]
    maximum=max(scores);prob=[math.exp(x-maximum) for x in scores];den=sum(prob)
    expected.extend(sum(prob[t-first]*v[(t*kh+keyhead)*width+j] for t in range(first,end+1))/den for j in range(width))
  addresses=[buffer(x.tobytes()) for x in (q,k,v)]
  guards=array.array('f',[12345.]+[0.]*len(expected)+[12345.]);output=buffer(guards.tobytes());partial=buffer(bytes(heads*4*(width+2)*4));request=buffer(array.array('I',[0,pos]).tobytes());check(sync())
  args=[Q(a) for a in addresses]+[Q(partial if split_mode else output+4),U(width),U(heads),U(kh),U(window),U(0),U(count),C.c_float(scale),Q(request)]
  packed=(P*len(args))(*(C.cast(C.byref(a),P) for a in args))
  check(launch(f,heads,4 if split_mode else count,1,256,1,1,0,None,packed,None))
  combine_args=[Q(partial),Q(output+4),U(width)];combine_packed=(P*3)(*(C.cast(C.byref(a),P) for a in combine_args))
  if split_mode:check(launch(combine,heads,1,1,max(128,width),1,1,0,None,combine_packed,None))
  check(sync())
  actual=array.array('f');actual.frombytes(read(output,len(guards)*4));assert actual[0]==actual[-1]==12345.
  error=max(abs(a-b)/(1+abs(b)) for a,b in zip(actual[1:-1],expected));assert error<=0.00015,(width,error)
  row=dict(width=width,heads=heads,kvheads=kh,position=pos,count=count,window=window,extreme=extreme,max_scaled_error=error);results.append(row);print(row,flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
(out/'results.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'cases':results,'tolerance':0.00015,'mode':'split' if split_mode else 'unsplit'},indent=2))
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
