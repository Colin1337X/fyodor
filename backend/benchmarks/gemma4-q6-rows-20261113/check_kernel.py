from common import *
random.seed(712);results=[]
try:
 for k,rows in [(256,1),(256,7),(256,9),(3840,65),(15360,3840),(3840,262144)]:
  stride=k//256*210
  w=bytearray(random.randbytes(rows*stride))
  for offset in range(0,len(w),210):w[offset+208:offset+210]=b'\x00\x28'
  weights=buffer(w);del w
  x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k))).tobytes())
  guard=array.array('f',[12345.]*(rows+2)).tobytes();reference=buffer(guard);actual=buffer(guard)
  call('nya_dot_q6_integer_14',(rows+7)//8,[Q(weights),Q(x),Q(reference+4),Q(k),Q(stride),Q(rows)])
  check(sync());expected=read(reference,(rows+2)*4)
  assert all(math.isfinite(v) for v in struct.unpack('<'+'f'*(rows+2),expected))
  for r in (1,2,4,8):
   call('nya_q6_rows_'+str(r),(rows+8*r-1)//(8*r),[Q(weights),Q(x),Q(actual+4),Q(k),Q(stride),Q(rows)])
   check(sync());assert read(actual,(rows+2)*4)==expected,(k,rows,r)
   assert expected[:4]==guard[:4] and expected[-4:]==guard[-4:]
  results.append(dict(columns=k,rows=rows,all_variants_exact=True));print(k,rows,flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(unload(module));check(release(device))
Path(__file__).with_name('oracle.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),cases=results),indent=2)+'\n')
