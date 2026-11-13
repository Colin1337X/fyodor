"""Exact serial equivalence, buffer guards and sampled float64 oracle."""
from common import *
import sys
out=Path(__file__).with_name(sys.argv[1] if len(sys.argv)>1 else 'oracle.json')
if out.exists():raise SystemExit('Refusing to overwrite evidence')
guard=struct.pack('<f',-123456.0)*32
results=[]
def guarded(size):
 base=buffer(guard+bytes(size)+guard);return base,base+len(guard)
def guards(base,size):
 assert read(base,len(guard))==guard and read(base+len(guard)+size,len(guard))==guard
cases=[(w,b,pos,win,mode,4,2) for w in (64,128,256)
       for b,pos in ((2,0),(4,17),(32,224),(256,0)) for win in (0,31) for mode in ('normal','retry')]
cases.extend([(256,32,17,31,'late',4,2),(256,256,0,1024,'normal',16,8)])
try:
 for width,batch,position,window,mode,heads,kvheads in cases:
  capacity=position+batch;partitions=4;stride=kvheads*width
  q=floats(batch*heads*width,13);k=floats(capacity*stride,47);v=floats(capacity*stride,71)
  for i in range(len(q)):q[i]*=.125
  for i in range(len(k)):k[i]*=.125
  if mode!='normal':
   for query in range(batch):
    for h in range(heads):q[(query*heads+h)*width]=64
   for token in range(capacity):
    for kh in range(kvheads):k[token*stride+kh*width]=1 if mode=='retry' or token==capacity-1 else 0
  dq,dk,dv=(buffer(a.tobytes()) for a in (q,k,v))
  part_bytes=batch*heads*partitions*(width+2)*4;output_bytes=batch*heads*width*4
  pb,dp=guarded(part_bytes);ob,dy=guarded(output_bytes);sb,ds=guarded(batch*heads*capacity*4)
  common=[U(width),U(heads),U(kvheads),U(window),U(position),C.c_float(1),Q(0),U(partitions)]
  call('nya_fdpp_partial_batch',heads,partitions*batch,256,[Q(dq),Q(dk),Q(dv),Q(dp),*common])
  call('nya_fdpp_combine_batch',heads,batch,256,[Q(dq),Q(dk),Q(dv),Q(dy),Q(dp),*common,Q(ds),U(capacity)])
  check(sync());actual=read(dy,output_bytes);parts=read(dp,part_bytes)
  for base,size_bytes in ((pb,part_bytes),(ob,output_bytes),(sb,batch*heads*capacity*4)):guards(base,size_bytes)
  retry_flags=[struct.unpack_from('<f',parts,((query*heads+h)*partitions+part)*(width+2)*4+(width+1)*4)[0]
               for query in range(batch) for h in range(heads) for part in range(partitions)]
  if mode!='normal':assert any(retry_flags)
  for query in range(batch):
   common[4]=U(position+query)
   qp=Q(dq+query*heads*width*4);pp=Q(dp+query*heads*partitions*(width+2)*4)
   call('nya_fdpp_partial_warp',heads,partitions,256,[qp,Q(dk),Q(dv),pp,*common])
   call('nya_fdpp_combine_warp',heads,1,256,[qp,Q(dk),Q(dv),Q(dy+query*heads*width*4),pp,*common,Q(ds),U(capacity)])
  check(sync());assert read(dy,output_bytes)==actual;assert read(dp,part_bytes)==parts
  output=array.array('f');output.frombytes(actual);maximum=0
  for query in sorted({0,batch-1}):
   end=position+query;first=max(0,end-window+1) if window else 0
   for h in (0,heads-1):
    kh=h//(heads//kvheads);offset=(query*heads+h)*width
    scores=[math.fsum(float(q[offset+j])*k[t*stride+kh*width+j] for j in range(width)) for t in range(first,end+1)]
    top=max(scores);probabilities=[math.exp(s-top) for s in scores];denominator=math.fsum(probabilities)
    for j in (0,width//2,width-1):
     reference=math.fsum(prob*v[t*stride+kh*width+j] for t,prob in zip(range(first,end+1),probabilities))/denominator
     error=abs(output[offset+j]-reference)/(1+abs(reference));assert math.isfinite(error) and error<=.0005
     maximum=max(maximum,error)
  results.append(dict(width=width,batch=batch,position=position,window=window,mode=mode,heads=heads,kvheads=kvheads,
                      exact_serial=True,guards=True,oracle_max_scaled=maximum,retry_partitions=sum(f!=0 for f in retry_flags)))
  print(json.dumps(results[-1]),flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
with out.open('x') as f:json.dump({'accepted_benchmark':False,'source_sha256':hashlib.sha256(source).hexdigest(),
 'cases':results,'note':'Original optional batch kernel, not runtime-integrated; exact serial comparison includes retry partials.'},f,indent=2)
