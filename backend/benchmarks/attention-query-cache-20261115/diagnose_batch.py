from common import *
import json,hashlib
out=Path(__file__).resolve().parent
results=[]
try:
 for width,position,batch in ((512,256,4),(512,1024,4),(512,8192,4),(256,256,4)):
  heads=16;kh=1;n=position+batch
  q=buffer(floats(batch*heads*width,47).tobytes());k=buffer(floats(n*width,31).tobytes());v=buffer(floats(n*width,71).tobytes())
  scores=buffer(bytes(batch*heads*n*4));parts=buffer(bytes(batch*heads*4*(width+2)*4))
  dest=buffer(bytes(batch*heads*width*4));ref=buffer(bytes(batch*heads*width*4))
  check(sync())
  def args(query,output,end):return [Q(query),Q(k),Q(v),Q(output),Q(scores),U(width),U(heads),U(kh),U(0),U(end),U(n),C.c_float(1),Q(0)]
  outputs={}
  for name in ('nya_attention_batch_split16_split','nya_attention_cached_query_split'):
   call(name,heads,batch*4,256,args(q,parts,position))
   call('nya_attention_combine',heads*batch,1,width,[Q(parts),Q(dest),U(width)])
   check(sync());outputs[name]=read(dest,batch*heads*width*4)
  for token in range(batch):
   call('nya_attention_vec16_split',heads,4,256,args(q+token*heads*width*4,parts,position+token))
   call('nya_attention_combine',heads,1,width,[Q(parts),Q(ref+token*heads*width*4),U(width)])
  check(sync());reference=read(ref,batch*heads*width*4)
  values_ref=array.array('f');values_ref.frombytes(reference)
  row=dict(width=width,position=position,batch=batch,variants={})
  for name,data in outputs.items():
   actual=array.array('f');actual.frombytes(data)
   row['variants'][name]=dict(exact=data==reference,differing_values=sum(a!=b for a,b in zip(actual,values_ref)),max_scaled_error=max(abs(a-b)/(1+abs(b)) for a,b in zip(actual,values_ref)))
  results.append(row);print(row,flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
(out/'batch-diagnostic.json').write_text(json.dumps(dict(accepted_benchmark=False,source_sha256=hashlib.sha256(source).hexdigest(),results=results),indent=2))
