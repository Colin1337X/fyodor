from common import *
results=[]
try:
 for width,position,batch,window,heads,kh in [(64,0,1,1,4,2),(130,273,3,17,4,2),(256,1023,1,1024,16,8),(256,8192,1,1024,16,8),(512,8192,1,0,16,1),(512,2048,1,1024,4,2),(256,1020,8,1024,4,2)]:
  stride=kh*width;n=position+batch;slots=min(n,window+batch-1) if window else n
  keys=floats(n*stride,31);values=floats(n*stride,71);query=floats(batch*heads*width,47)
  rk=array.array('f',[float('nan')])*(slots*stride);rv=array.array('f',[float('nan')])*(slots*stride)
  for t in range(n):
   off=t%slots*stride;rk[off:off+stride]=keys[t*stride:(t+1)*stride];rv[off:off+stride]=values[t*stride:(t+1)*stride]
  q=buffer(query.tobytes());k=buffer(rk.tobytes());v=buffer(rv.tobytes());scores=buffer(bytes(batch*heads*n*4));parts=buffer(bytes(heads*4*(width+2)*4))
  size=batch*heads*width;guards=array.array('f',[12345.]*(size+2)).tobytes();expected=buffer(guards);actual=buffer(guards)
  request=buffer(array.array('I',[0,position]).tobytes());unaligned=buffer(bytes(4)+rv.tobytes())+4
  for partitions in ((1,4) if batch==1 and min(n,window or n)>=257 else (1,)):
   for dynamic in ((False,True) if batch==1 else (False,)):
    ring=slots<n
    for variant in ('baseline','vec4','vec8','vec16'):
     dest=expected if variant=='baseline' else actual
     name='nya_attention'+('_'+variant if variant!='baseline' else '')+('_ring' if ring else '')+('_split' if partitions==4 else '')
     args=[Q(q),Q(k),Q(v),Q(parts if partitions==4 else dest+4),Q(scores),U(width),U(heads),U(kh),U(window),U(0 if dynamic else position),U(n),C.c_float(1/math.sqrt(width)),Q(request if dynamic else 0)]
     if ring:args.append(U(slots))
     call(name,heads,partitions if partitions==4 else batch,256,args)
     if partitions==4:call('nya_attention_combine',heads,1,max(128,width),[Q(parts),Q(dest+4),U(width)])
     if variant!='baseline':
      check(sync());assert read(actual,(size+2)*4)==read(expected,(size+2)*4),(variant,width,position,batch,partitions,dynamic)
      if variant=='vec16':
       args[2]=Q(unaligned);call(name,heads,partitions if partitions==4 else batch,256,args)
       if partitions==4:call('nya_attention_combine',heads,1,max(128,width),[Q(parts),Q(dest+4),U(width)])
       check(sync());assert read(actual,(size+2)*4)==read(expected,(size+2)*4),('unaligned',width,position)

    check(sync());a=read(actual,(size+2)*4);b=read(expected,(size+2)*4)
    assert a==b,(width,position,batch,partitions,dynamic)
    assert a[:4]==guards[:4] and a[-4:]==guards[-4:] and all(math.isfinite(x) for x in struct.unpack('<'+'f'*(size+2),a))
    results.append(dict(width=width,position=position,batch=batch,partitions=partitions,ring=ring,device_request=dynamic,bit_exact=True))
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
print(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),options=[x.decode() for x in opts],cases=results),indent=2))
