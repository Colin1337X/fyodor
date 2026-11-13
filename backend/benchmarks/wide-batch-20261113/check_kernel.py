"""Exact native parity and sampled float64 oracle for narrow batched attention."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'gemma4-wide-value-20261113'))
from common import *
results=[]
try:
 for width in (256,512):
  for position in (255,256,1023,8192):
   for batch in (2,4):
    for ring in (False,True):
     heads=4;kh=2 if width==256 else 1;window=257 if ring else 0
     n=position+batch;stride=kh*width;slots=min(n,window+batch-1) if ring else n
     keys=floats(n*stride,31);values=floats(n*stride,71);query=floats(batch*heads*width,47)
     rk=array.array('f',[float('nan')])*(slots*stride);rv=rk[:]
     for t in range(n):
      offset=t%slots*stride;rk[offset:offset+stride]=keys[t*stride:(t+1)*stride];rv[offset:offset+stride]=values[t*stride:(t+1)*stride]
     q=buffer(query.tobytes());k=buffer(rk.tobytes());v=buffer(rv.tobytes());scores=buffer(bytes(batch*heads*n*4))
     count=batch*heads*width;guards=array.array('f',[12345.]*(count+2)).tobytes();outputs=[]
     for variant in ('','_vec16'):
      dest=buffer(guards)
      args=[Q(q),Q(k),Q(v),Q(dest+4),Q(scores),U(width),U(heads),U(kh),U(window),U(position),U(n),C.c_float(1),Q(0)]
      if ring:args.append(U(slots))
      call('nya_attention'+variant+('_ring' if ring else ''),heads,batch,256,args)
      check(sync());outputs.append(read(dest,len(guards)))
     assert outputs[0]==outputs[1]
     assert outputs[1][:4]==guards[:4] and outputs[1][-4:]==guards[-4:]
     actual=array.array('f');actual.frombytes(outputs[1][4:-4]);assert all(math.isfinite(x) for x in actual)
     # Independent double-precision dot, softmax and value reduction for the
     # final query/head; exact comparison above covers every output element.
     qi=batch-1;end=position+qi;first=max(0,end-window+1) if window else 0
     logits=[sum(float(query[qi*heads*width+j])*keys[t*stride+j] for j in range(width)) for t in range(first,end+1)]
     maximum=max(logits);weights=[math.exp(x-maximum) for x in logits];den=sum(weights);error=0
     for j in (0,17,width-1):
      expected=sum(s*values[t*stride+j] for s,t in zip(weights,range(first,end+1)))/den
      error=max(error,abs(actual[qi*heads*width+j]-expected)/(1+abs(expected)))
     assert error<=.0005
     results.append(dict(width=width,position=position,batch=batch,ring=ring,exact=True,scaled_error=error))
     for address in owned:check(free(address))
     owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
Path(__file__).with_name('oracle.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'cases':results},indent=2)+'\n')
print('Passed',len(results),'batched attention cases')
