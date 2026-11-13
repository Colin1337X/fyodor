"""Batched split attention: serial split parity, guards and float64 oracle."""
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
     q=buffer(query.tobytes());k=buffer(rk.tobytes());v=buffer(rv.tobytes())
     def guarded(count):
      data=array.array('f',[12345.]*(count+2)).tobytes()
      return buffer(data),data
     scores,score_guard=guarded(batch*heads*n)
     parts,part_guard=guarded(batch*heads*4*(width+2))
     dest,guard=guarded(batch*heads*width);reference,_=guarded(batch*heads*width);unsplit,_=guarded(batch*heads*width)
     def arguments(input_ptr,output_ptr,end):
      args=[Q(input_ptr),Q(k),Q(v),Q(output_ptr),Q(scores+4),U(width),U(heads),U(kh),U(window),U(end),U(n),C.c_float(1),Q(0)]
      if ring:args.append(U(slots))
      return args
     suffix='_ring' if ring else ''
     call('nya_attention_batch_split16'+suffix+'_split',heads,batch*4,256,arguments(q,parts+4,position))
     call('nya_attention_combine',heads*batch,1,width,[Q(parts+4),Q(dest+4),U(width)])
     check(sync());actual_bytes=read(dest,len(guard))
     for address,data in ((parts,part_guard),(scores,score_guard),(dest,guard)):
      observed=read(address,len(data));assert observed[:4]==data[:4] and observed[-4:]==data[-4:]
     for token in range(batch):
      call('nya_attention_vec16'+suffix+'_split',heads,4,256,arguments(q+token*heads*width*4,parts+4,position+token))
      call('nya_attention_combine',heads,1,width,[Q(parts+4),Q(reference+4+token*heads*width*4),U(width)])
     call('nya_attention_vec16'+suffix,heads,batch,256,arguments(q,unsplit+4,position))
     check(sync());assert actual_bytes==read(reference,len(guard))
     actual=array.array('f');actual.frombytes(actual_bytes[4:-4]);baseline=array.array('f');baseline.frombytes(read(unsplit+4,batch*heads*width*4))
     assert all(math.isfinite(x) for x in actual)
     delta=max(abs(a-b)/(1+abs(b)) for a,b in zip(actual,baseline));assert delta<=.0005
     # Independent double precision, final query and first head.
     qi=batch-1;end=position+qi;first=max(0,end-window+1) if window else 0
     logits=[math.fsum(float(query[qi*heads*width+j])*keys[t*stride+j] for j in range(width)) for t in range(first,end+1)]
     maximum=max(logits);weights=[math.exp(x-maximum) for x in logits];den=math.fsum(weights);error=0
     for j in (0,17,width-1):
      expected=math.fsum(s*values[t*stride+j] for s,t in zip(weights,range(first,end+1)))/den
      error=max(error,abs(actual[qi*heads*width+j]-expected)/(1+abs(expected)))
     assert error<=.0005
     results.append(dict(width=width,position=position,batch=batch,ring=ring,serial_split_exact=True,unsplit_scaled_error=delta,float64_scaled_error=error))
     print(width,position,batch,ring,'passed',flush=True)
     for address in owned:check(free(address))
     owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
with Path(__file__).with_name('memcheck-oracle.json').open('x') as output:
 json.dump({'source_sha256':hashlib.sha256(source).hexdigest(),'cases':results},output,indent=2)
