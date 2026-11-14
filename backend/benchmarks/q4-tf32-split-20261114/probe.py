"""Synthetic Q4 kernel diagnostic; never an accepted end-to-end score."""
import hashlib
import json
from pathlib import Path

if not __debug__: raise SystemExit('Timing assertions are required')
out = Path(__file__).resolve().parent
root = out.parents[2]
destination = out / 'timing-upload-sync.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite timing evidence')
oracle_path = out / 'oracle.json'
if not oracle_path.exists():
    raise SystemExit('Complete the raw correctness sweep before timing')
oracle = json.loads(oracle_path.read_text())
from run_sanitizer import validate_oracle
validate_oracle(oracle)
for name, expected in oracle['input_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
for tool in ('memcheck','racecheck','synccheck'):
    sanitizer_path=out/(tool+'-run.json')
    if not sanitizer_path.exists(): raise SystemExit('Complete '+tool+' before timing')
    sanitizer=json.loads(sanitizer_path.read_text())
    assert sanitizer['passed'] is True and sanitizer['exit_code']==0
    for name,expected in sanitizer['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    for name,expected in sanitizer['evidence_sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected,name
import sys
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
# The common import creates a CUDA context; all evidence checks must precede it.
from common import *
stream=P();stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P]);check(stream_create(C.byref(stream),1))
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P]);graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P]);event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P]);event_destroy=bind(cu,'cuEventDestroy_v2',[P])
a=P();b=P();check(event_create(C.byref(a),0));check(event_create(C.byref(b),0))


def graph_call(name,blocks,grid_y,args):
 threads=128 if name=='nya_q4_tf32_split' else 256
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,grid_y,1,threads,1,1,0,stream,packed,None))
import statistics,sys
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
library=C.CDLL(str(root/'build-cuda/q4-tf32-three-oracle-20261114/oracle-c17.dll'))
scalar_oracle=library.q4_oracle;scalar_oracle.argtypes=[P,P,Z,Z,P,P];scalar_oracle.restype=I
random.seed(713);results=[]
try:
 for dtype,k,rows in [(12, 3840, 512), (12, 3840, 2048), (12, 3840, 4096), (12, 3840, 8192), (12, 3840, 15360), (12, 4096, 3840), (12, 8192, 3840), (12, 15360, 3840)]:
  block_bytes=144 if dtype==12 else 210
  stride=k//256*block_bytes;w=bytearray(random.randbytes(rows*stride))
  for offset in range(0,len(w),block_bytes):
   if dtype==12:w[offset:offset+4]=struct.pack('<ee',.03125,.00390625)
   else:w[offset+208:offset+210]=struct.pack('<e',.03125)
  weights=buffer(w)
  x_host=array.array('f',(random.uniform(-1,1) for _ in range(k*4))).tobytes();x=buffer(x_host);y=buffer(bytes(rows*4*4));packed_input=buffer(bytes(k*4*4))
  expected,bounds=(C.c_double*(rows*4))(),(C.c_double*(rows*4))()
  assert scalar_oracle(C.create_string_buffer(bytes(w)),C.create_string_buffer(x_host),k,rows,expected,bounds)==0
  del w
  check(sync()) # Complete default-stream uploads before nonblocking graph use.
  for batch in (4,):
   names=['metadata','tensor'];samples={n:[] for n in names};snapshots=[];expected_output=None
   orders=[names,list(reversed(names)),list(reversed(names)),names]
   for order in orders:
    for name in order:
     count=4
     kernel='nya_integer_metadata_12_4' if name=='metadata' else 'nya_q4_tf32_split'
     tile=8 if name=='metadata' else 16
     graph=P();executable=P();check(begin(stream,0))
     for i in range(10):
      graph_call('nya_interleave_four',(k+255)//256,1,[Q(x),Q(packed_input),Q(k)])
      graph_call(kernel,(rows+tile-1)//tile,1,[Q(weights),Q(packed_input),Q(y),Q(k),Q(stride),U(rows),U(batch)])
     check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
     for i in range(100):check(graphlaunch(executable,stream))
     for repeat in range(10):
      check(event_record(a,stream))
      for i in range(10):check(graphlaunch(executable,stream))
      check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b));samples[name].append(ms.value/100)
      assert math.isfinite(ms.value) and ms.value>0, 'Invalid event duration'
     check(execdestroy(executable));check(graphdestroy(graph));check(sync());snapshots.append(environment.snapshot())
     actual_output=read(y,rows*4*4)
     actual=array.array('f');actual.frombytes(actual_output)
     assert all(math.isfinite(v) and abs(v-expected[j])<=bounds[j] for j,v in enumerate(actual)), 'Timed numerical bound failed'
     print(dtype,k,rows,batch,name,statistics.mean(samples[name]),flush=True)
   results.append(dict(dtype=dtype,columns=k,rows=rows,batch=batch,order=[name for order in orders for name in order],samples_ms=samples,means_ms={n:statistics.mean(v) for n,v in samples.items()},sample_stddev_ms={n:statistics.stdev(v) for n,v in samples.items()},environment=snapshots))
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
with destination.open('x') as output:
 json.dump(dict(accepted_benchmark=False, reason='Synthetic repeated weight banks; not end-to-end decode and no interval contamination audit', source_sha256=hashlib.sha256(source).hexdigest(), oracle_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest(), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), options=[option.decode() for option in opts], compute_capability=[major,minor], operations_per_sample=100, warmup_operations_per_order=1000, results=results),output,indent=2)
