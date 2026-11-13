"""Known CUDA PID for monitor validation, not a performance benchmark."""
import sys,time,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'interleaved-two-20261114'))
from common import *
try:
 columns=3840;rows=15360;stride=columns//256*144
 w=buffer(bytes(rows*stride));x=buffer(array.array('f',[1.]*(columns*2)).tobytes());y=buffer(bytes(rows*2*4))
 args=[Q(w),Q(x),Q(y),Q(columns),Q(stride),U(rows),U(2)]
 start=time.monotonic();calls=0
 while time.monotonic()-start<10:
  for i in range(64):call('nya_small_reuse_12_2',(rows+7)//8,1,args);calls+=1
  check(sync())
 assert read(y,rows*2*4)==bytes(rows*2*4)
 print(json.dumps({'pid':os.getpid(),'calls':calls,'elapsed_seconds':time.monotonic()-start,'zero_output':True}))
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
