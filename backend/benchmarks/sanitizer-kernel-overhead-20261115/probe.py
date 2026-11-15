"""Real Q4 kernels, synthetic weights: overhead isolation, not inference scores."""
import array,ctypes as C,hashlib,importlib.util,json,math,random,statistics,struct,sys,time
from pathlib import Path
if not __debug__:raise SystemExit('Assertions required')
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=2 or not sys.argv[1].isascii() or not sys.argv[1].isalnum():raise SystemExit('Unique label required')
destination=out/(sys.argv[1]+'.json')
if destination.exists():raise SystemExit('Refusing overwrite')
common=root/'backend/benchmarks/q4-pair-rows-20261114/common.py'
spec=importlib.util.spec_from_file_location('kernel_common',common);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
P,Q,U,Z,I=m.P,m.Q,m.U,m.Z,m.I
k,rows=3840,15360;stride=k//256*144
rng=random.Random(3719);w=bytearray(rng.randbytes(rows*stride))
for offset in range(0,len(w),144):w[offset:offset+4]=struct.pack('<ee',.03125,.00390625)
xhost=array.array('f',(rng.uniform(-1,1) for _ in range(k*4))).tobytes()
oracle_path=root/'build-cuda/q4-tf32-three-oracle-20261114/oracle-c17.dll'
lib=C.CDLL(str(oracle_path));oracle=lib.q4_oracle;oracle.argtypes=[P,P,Z,Z,P,P];oracle.restype=I
expected,bounds=(C.c_double*(rows*4))(),(C.c_double*(rows*4))()
assert oracle(C.create_string_buffer(bytes(w)),C.create_string_buffer(xhost),k,rows,expected,bounds)==0
weights=m.buffer(w);x=m.buffer(xhost);y=m.buffer(bytes(rows*4*4));packed=m.buffer(bytes(k*4*4));m.check(m.sync())
for name in ('nya_matvec','nya_dot_12'):
 f=P();m.check(m.function(C.byref(f),m.module,name.encode()));m.funcs[name]=f
m.call('nya_interleave_four',(k+255)//256,1,[Q(x),Q(packed),Q(k)]);m.check(m.sync())
eventcreate=m.bind(m.cu,'cuEventCreate',[C.POINTER(P),U]);record=m.bind(m.cu,'cuEventRecord',[P,P]);eventsync=m.bind(m.cu,'cuEventSynchronize',[P]);elapsed=m.bind(m.cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P]);destroy=m.bind(m.cu,'cuEventDestroy_v2',[P])
a,b=P(),P();m.check(eventcreate(C.byref(a),0));m.check(eventcreate(C.byref(b),0));extra=Q();results=[]
cases=[('nya_matvec',rows,[Q(weights),Q(x),Q(y),Q(k),Q(stride),U(12)],rows),('nya_dot_12',(rows+7)//8,[Q(weights),Q(x),Q(y),Q(k),Q(stride),Q(rows)],rows),('nya_q4_pair_rows',(rows+15)//16,[Q(weights),Q(packed),Q(y),Q(k),Q(stride),U(rows),U(4)],rows*4)]
try:
 for pad in (0,8*1024**3,0):
  if pad:m.check(m.alloc(C.byref(extra),pad))
  for name,blocks,args,count in cases:
   packed_args=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
   def call():m.check(m.launch(m.funcs[name],blocks,1,1,256,1,1,0,None,packed_args,None))
   call();m.check(m.sync());gpu_ms=[];wall_ms=[]
   for repeat in range(5):
    start=time.perf_counter();m.check(record(a,None));call();m.check(record(b,None));m.check(eventsync(b));wall_ms.append((time.perf_counter()-start)*1000)
    ms=C.c_float();m.check(elapsed(C.byref(ms),a,b));assert math.isfinite(ms.value) and ms.value>0;gpu_ms.append(ms.value)
   actual=array.array('f');actual.frombytes(m.read(y,count*4))
   assert all(math.isfinite(v) and abs(v-expected[j])<=bounds[j] for j,v in enumerate(actual)),name
   r=dict(kernel=name,unused_allocation_bytes=pad,gpu_ms=gpu_ms,wall_ms=wall_ms,mean_gpu_ms=statistics.mean(gpu_ms),mean_wall_ms=statistics.mean(wall_ms),numerical_passed=True)
   results.append(r);print('PROGRESS '+json.dumps(r),flush=True)
  if extra.value:m.check(m.free(extra));extra=Q()
finally:
 if extra.value:m.check(m.free(extra))
 for address in m.owned:m.check(m.free(address))
 m.owned.clear();m.check(destroy(a));m.check(destroy(b));m.check(m.unload(m.module));m.check(m.release(m.device))
paths=[Path(__file__),common,oracle_path,*[root/'CUDA'/f for f in ('matvec.cu','gemm.cu','resident.cu')],*[root/'optcpp/matvec'/f for f in ('small_batch.cu','small_batch_reuse.cu','interleaved_reuse.cu','interleaved_integer.cu','integer_launch.cu','integer_metadata.cu','q4_tf32_three.cu','q4_pair_rows.cu')]]
result=dict(passed=True,accepted_benchmark=False,columns=k,rows=rows,source_sha256=hashlib.sha256(m.source).hexdigest(),options=[s.decode() for s in m.opts],results=results,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
with destination.open('x') as f:json.dump(result,f,indent=2)
