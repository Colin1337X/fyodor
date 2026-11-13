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
for name in ('nya_attention','nya_attention_split','nya_attention_ring','nya_attention_ring_split','nya_attention_combine'):
 f=P();check(function(C.byref(f),module,name.encode()));funcs[name]=f

def call(name,gx,gy,bx,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],gx,gy,1,bx,1,1,0,None,packed,None))

def floats(count,seed):
 return array.array('f',(((i*17+seed)%257-128)/256 for i in range(count)))

results=[]
# Batch cases straddle physical wrap boundaries; short cases cover initial fill.
cases=[(64,0,1,7),(128,6,1,7),(256,1023,1,1024),(256,1024,1,1024),
       (256,8192,1,1024),(512,2048,1,1024),(256,1020,8,1024),
       (256,2044,8,1024),(128,8190,4,1024)]
try:
 for width,position,batch,window in cases:
  heads=4;kh=2;stride=kh*width;n=position+batch;slots=min(n,window+batch-1)
  keys=floats(n*stride,31);values=floats(n*stride,71);query=floats(batch*heads*width,47)
  rk=array.array('f',[float('nan')])*(slots*stride);rv=array.array('f',[float('nan')])*(slots*stride)
  for t in range(n):
   offset=(t%slots)*stride;rk[offset:offset+stride]=keys[t*stride:(t+1)*stride];rv[offset:offset+stride]=values[t*stride:(t+1)*stride]
  q=buffer(query.tobytes());k=buffer(keys.tobytes());v=buffer(values.tobytes());kr=buffer(rk.tobytes());vr=buffer(rv.tobytes())
  scores=buffer(bytes(batch*heads*n*4));parts=buffer(bytes(heads*4*(width+2)*4))
  size=batch*heads*width;guards=array.array('f',[12345.]*(size+2)).tobytes();expected=buffer(guards);actual=buffer(guards)
  request=buffer(array.array('I',[0,position]).tobytes())
  for partitions in ((1,4) if batch==1 and min(n,window)>=257 else (1,)):
   for dynamic in ((False,True) if batch==1 else (False,)):
    for ring in (False,True):
     dest=actual if ring else expected
     name='nya_attention'+('_ring' if ring else '')+('_split' if partitions==4 else '')
     args=[Q(q),Q(kr if ring else k),Q(vr if ring else v),Q(parts if partitions==4 else dest+4),Q(scores),U(width),U(heads),U(kh),U(window),U(0 if dynamic else position),U(n),C.c_float(1/math.sqrt(width)),Q(request if dynamic else 0)]
     if ring:args.append(U(slots))
     call(name,heads,partitions if partitions==4 else batch,256,args)
     if partitions==4:call('nya_attention_combine',heads,1,max(128,width),[Q(parts),Q(dest+4),U(width)])
    check(sync());a=read(actual,(size+2)*4);b=read(expected,(size+2)*4)
    assert a==b,(width,position,batch,partitions,dynamic)
    assert a[:4]==guards[:4] and a[-4:]==guards[-4:]
    assert all(math.isfinite(x) for x in struct.unpack('<'+'f'*(size+2),a))
    results.append(dict(width=width,position=position,batch=batch,window=window,slots=slots,partitions=partitions,device_request=dynamic,bit_exact=True))
  if batch>1 and n>window:
   badk=array.array('f',[0])*(window*stride);badv=array.array('f',[0])*(window*stride)
   for t in range(n):
    offset=(t%window)*stride;badk[offset:offset+stride]=keys[t*stride:(t+1)*stride];badv[offset:offset+stride]=values[t*stride:(t+1)*stride]
   bk=buffer(badk.tobytes());bv=buffer(badv.tobytes())
   args=[Q(q),Q(bk),Q(bv),Q(actual+4),Q(scores),U(width),U(heads),U(kh),U(window),U(position),U(n),C.c_float(1/math.sqrt(width)),Q(0),U(window)]
   call('nya_attention_ring',heads,batch,256,args);check(sync())
   assert read(actual,(size+2)*4)!=read(expected,(size+2)*4),'Negative control did not detect overwritten keys'
   results[-1]['undersized_window_only_ring_detected']=True
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
print(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),options=[x.decode() for x in opts],cases=results),indent=2))
