"""Synthetic Q4 kernel diagnostic; never an accepted end-to-end score."""
import hashlib
import json
from pathlib import Path

if not __debug__: raise SystemExit('Timing assertions are required')
out = Path(__file__).resolve().parent
root = out.parents[2]
destination = out / 'timing.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite timing evidence')
oracle_path = out / 'oracle.json'
if not oracle_path.exists():
    raise SystemExit('Complete the raw correctness sweep before timing')
oracle = json.loads(oracle_path.read_text())
assert oracle['passed'] is True and len(oracle['cases']) == 207
assert all(case['byte_exact'] is True and case['guards_clear'] is True for case in oracle['cases'])
for name, expected in oracle['harness_sha256'].items():
    assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected, name
paths = [root / 'CUDA' / name for name in ('matvec.cu', 'gemm.cu', 'resident.cu')]
paths += [root / 'optcpp/matvec' / name for name in
          ('small_batch.cu', 'small_batch_reuse.cu', 'interleaved_reuse.cu', 'interleaved_integer.cu', 'integer_launch.cu', 'integer_metadata.cu', 'integer_metadata_unroll2.cu')]
assert hashlib.sha256(b'\n'.join(path.read_bytes() for path in paths)).hexdigest() == oracle['source_sha256']
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
 threads=128 if name.startswith('nya_integer_launch_4_') else 256
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,grid_y,1,threads,1,1,0,stream,packed,None))
import statistics,sys
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
random.seed(713);results=[]
try:
 for dtype,k,rows in [(12, 3840, 512), (12, 3840, 2048), (12, 3840, 4096), (12, 3840, 8192), (12, 3840, 15360), (12, 4096, 3840), (12, 8192, 3840), (12, 15360, 3840)]:
  block_bytes=144 if dtype==12 else 210
  stride=k//256*block_bytes;w=bytearray(random.randbytes(rows*stride))
  for offset in range(0,len(w),block_bytes):
   if dtype==12:w[offset:offset+4]=struct.pack('<ee',.03125,.00390625)
   else:w[offset+208:offset+210]=struct.pack('<e',.03125)
  weights=buffer(w);del w
  x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k*4))).tobytes());y=buffer(bytes(rows*4*4));packed_input=buffer(bytes(k*4*4))
  for batch in (4,):
   names=['metadata','shared'];samples={n:[] for n in names};snapshots=[];expected_output=None
   for order in [names,list(reversed(names))]:
    for name in order:
     count=4
     kernel='nya_integer_metadata_12_4' if name=='metadata' else 'nya_integer_metadata_unroll2'
     tile=8
     graph=P();executable=P();check(begin(stream,0))
     for i in range(10):
      graph_call('nya_interleave_four',(k+255)//256,1,[Q(x),Q(packed_input),Q(k)])
      graph_call(kernel,(rows+tile-1)//tile,1,[Q(weights),Q(packed_input),Q(y),Q(k),Q(stride),U(rows),U(batch)])
     check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
     for i in range(3):check(graphlaunch(executable,stream))
     for repeat in range(5):
      check(event_record(a,stream))
      for i in range(10):check(graphlaunch(executable,stream))
      check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b));samples[name].append(ms.value/100)
      assert math.isfinite(ms.value) and ms.value>0, 'Invalid event duration'
     check(execdestroy(executable));check(graphdestroy(graph));check(sync());snapshots.append(environment.snapshot())
     actual_output=read(y,rows*4*4)
     if expected_output is None: expected_output=actual_output
     else: assert actual_output==expected_output, 'Timed kernels disagree'
     print(dtype,k,rows,batch,name,statistics.mean(samples[name]),flush=True)
   results.append(dict(dtype=dtype,columns=k,rows=rows,batch=batch,order=names+list(reversed(names)),samples_ms=samples,means_ms={n:statistics.mean(v) for n,v in samples.items()},sample_stddev_ms={n:statistics.stdev(v) for n,v in samples.items()},environment=snapshots))
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
with destination.open('x') as output:
 json.dump(dict(accepted_benchmark=False, reason='Synthetic repeated weight banks; not end-to-end decode and no interval contamination audit', source_sha256=hashlib.sha256(source).hexdigest(), oracle_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest(), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), options=[option.decode() for option in opts], compute_capability=[major,minor], operations_per_sample=100, warmup_operations_per_order=30, results=results),output,indent=2)
