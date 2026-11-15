"""Intentional device OOB write in a disposable context, for filter diagnostics."""
import ctypes as C
from pathlib import Path
import sys,os

def main():
 if len(sys.argv)!=2 or sys.argv[1] not in ('nya_attention_cached_query_probe','unrelated_probe'):
  raise SystemExit('Choose one fixture kernel')
 cu=C.WinDLL('nvcuda.dll');P=C.c_void_p;I=C.c_int;U=C.c_uint;Q=C.c_uint64;Z=C.c_size_t
 def bind(name,args):
  f=getattr(cu,name);f.argtypes=args;f.restype=I;return f
 def check(status):
  if status:raise RuntimeError('CUDA status '+str(status))
 init=bind('cuInit',[U]);deviceget=bind('cuDeviceGet',[C.POINTER(I),I])
 retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(P),I]);release=bind('cuDevicePrimaryCtxRelease_v2',[I])
 current=bind('cuCtxSetCurrent',[P]);sync=bind('cuCtxSynchronize',[])
 load=bind('cuModuleLoadData',[C.POINTER(P),P]);unload=bind('cuModuleUnload',[P])
 function=bind('cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p])
 alloc=bind('cuMemAlloc_v2',[C.POINTER(Q),Z]);free=bind('cuMemFree_v2',[Q])
 launch=bind('cuLaunchKernel',[P,U,U,U,U,U,U,U,P,C.POINTER(P),P])
 check(init(0));device=I();check(deviceget(C.byref(device),0));context=P();check(retain(C.byref(context),device))
 module=P();address=Q()
 try:
  check(current(context));ptx=C.create_string_buffer((Path.cwd()/('build-cuda/sanitizer-kernel-overhead-20261115/'+os.environ['FYODOR_SHARED_FIXTURE'])).read_bytes())
  check(load(C.byref(module),ptx));f=P();check(function(C.byref(f),module,sys.argv[1].encode()))
  check(alloc(C.byref(address),4))
  args=(P*1)(C.cast(C.byref(address),P))
  print('FIXTURE kernel='+sys.argv[1]+' allocation_bytes=4 shared_write_offset=4096 dynamic_shared_bytes=4',flush=True)
  check(launch(f,1,1,1,1,1,1,4,None,args,None));check(sync())
 finally:
  if address.value:free(address)
  if module.value:unload(module)
  release(device)
if __name__=='__main__':main()
