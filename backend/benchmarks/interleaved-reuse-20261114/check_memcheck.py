from common import *
random.seed(713);results=[]
try:
 for dtype,block_bytes in ((12,144),(14,210)):
  for columns,rows in [(256,1),(256,7),(256,9)]+[(k,r) for t,k,r in [[12, 3840, 512], [12, 3840, 2048], [12, 3840, 4096], [12, 3840, 8192], [12, 3840, 15360], [12, 4096, 3840], [12, 8192, 3840], [12, 15360, 3840], [14, 3840, 2048], [14, 3840, 262144], [14, 15360, 3840]] if t==dtype]:
   stride=columns//256*block_bytes
   weights_host=bytearray(random.randbytes(rows*stride))
   for offset in range(0,len(weights_host),block_bytes):
    if dtype==12:
     weights_host[offset:offset+4]=struct.pack('<ee',(.03125,.03,-.017,.125)[(offset//block_bytes)%4],(.00390625,.004,.002,-.001)[(offset//block_bytes)%4])
    else:weights_host[offset+208:offset+210]=struct.pack('<e',(.03125,.03,-.017,.125)[(offset//block_bytes)%4])
   weights=buffer(weights_host)
   input_host=array.array('f',(random.uniform(-1,1) for _ in range(columns*8)))
   x=buffer(input_host.tobytes())
   packed_guard=array.array('f',[12345.]*(columns*4+8)).tobytes()
   packed_input=buffer(packed_guard)
   call('nya_interleave_four',(columns+255)//256,1,[Q(x),Q(packed_input+16),Q(columns)])
   check(sync())
   packed_expected=array.array('f',[12345.]*4+[input_host[t*columns+j] for j in range(columns) for t in range(4)]+[12345.]*4).tobytes()
   assert read(packed_input,len(packed_guard))==packed_expected
   for batch in (4,):
    guard=array.array('f',[12345.]*(rows*batch+2)).tobytes()
    reference=buffer(guard);actual=buffer(guard)
    args=[Q(weights),Q(x),Q(reference+4),Q(columns),Q(stride),U(rows),U(batch)]
    call('nya_small_batch_'+str(dtype),(rows+7)//8,batch,args)
    check(sync());expected=read(reference,len(guard))
    values=array.array('f');values.frombytes(expected)
    assert all(math.isfinite(v) for v in values)
    assert expected[:4]==guard[:4] and expected[-4:]==guard[-4:]
    if columns==256:
     for token in range(batch):
      for row in range(rows):
       b=weights_host[row*stride:(row+1)*stride];products=[]
       for j in range(columns):
        if dtype==12:
         g=j//32;sc=(b[4+g]&63) if g<4 else (b[8+g]&15)|((b[g]>>6)<<4)
         mn=(b[8+g]&63) if g<4 else (b[8+g]>>4)|((b[4+g]>>6)<<4)
         q=(b[16+g//2*32+j%32]>>(g%2*4))&15
         d,m=struct.unpack('<ee',b[:4]);weight=d*sc*q-m*mn
        else:
         half=j//128;quarter=j%128//32;lane=j%32
         lo=b[half*64+quarter%2*32+lane];hi=b[128+half*32+lane]
         q=((lo>>(quarter//2*4))&15)|(((hi>>(quarter*2))&3)<<4)
         sc=b[192+j//16];sc=sc if sc<128 else sc-256
         d=struct.unpack('<e',b[208:210])[0];weight=d*sc*(q-32)
        products.append(weight*input_host[token*columns+j])
       error=abs(values[1+token*rows+row]-math.fsum(products))
       assert error<=2e-6*math.fsum(abs(v) for v in products)+1e-6
    for tile in (4,):
     args[2]=Q(actual+4);args[1]=Q(packed_input+16)
     call(f'nya_interleaved_reuse_{dtype}_{tile}',(rows+7)//8,(batch+tile-1)//tile,args)
     check(sync());assert read(actual,len(guard))==expected,(dtype,columns,rows,batch,tile)
    results.append(dict(dtype=dtype,columns=columns,rows=rows,batch=batch,all_variants_exact=True,float64_checked=columns==256))
    for address in owned[-2:]:check(free(address))
    del owned[-2:]
   print(dtype,columns,rows,flush=True)
   for address in owned:check(free(address))
   owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
Path(__file__).with_name('memcheck-oracle.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),cases=results),indent=2)+'\n')
