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
inputs={str(Path(name).resolve()):expected for name,expected in oracle['input_sha256'].items()}
for tool in ('memcheck','racecheck','synccheck'):
 path=out/(tool+'-run.json');proof=json.loads(path.read_text())
 inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
 for name,expected in proof['input_sha256'].items():inputs[str(Path(name).resolve())]=expected
 for name,expected in proof['evidence_sha256'].items():inputs[str((out/name).resolve())]=expected
for path in (oracle_path,Path(__file__),out/'common.py',out/'run_sanitizer.py',Path(environment.__file__)):
 inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
# The common import creates a CUDA context; all evidence checks must precede it.
from common import *
stream=P();stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P]);check(stream_create(C.byref(stream),1))
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P]);graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P]);event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P]);event_destroy=bind(cu,'cuEventDestroy_v2',[P])
a=P();b=P();check(event_create(C.byref(a),0));check(event_create(C.byref(b),0))


def graph_call(name,blocks,grid_y,args):
 threads=256
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,grid_y,1,threads,1,1,0,stream,packed,None))
import statistics
random.seed(715)
results=[]
result=dict(passed=False,accepted_benchmark=False,results=results,input_sha256=inputs,
    reason='Synthetic repeated weight banks; no interval coverage or end-to-end performance acceptance',
    operations_per_sample=100,warmup_operations_per_order=1000)
graphs=[];executables=[]
try:
 for k,rows in ((256,17),(3840,512),(3840,2048),(3840,4096),(3840,8192),(3840,15360)):
  stride=k//256*144
  weights=[]
  for matrix in range(2):
   data=bytearray(random.randbytes(rows*stride))
   for offset in range(0,len(data),144):data[offset:offset+4]=struct.pack('<ee',.03125+matrix*.0078125,.00390625)
   weights.append(buffer(data))
  x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k*4))).tobytes())
  packed_input=buffer(bytes(k*16));gate=buffer(bytes(rows*16));up=buffer(bytes(rows*16))
  guard=array.array('f',[12345.]*(rows*4+2)).tobytes()
  outputs={name:buffer(guard) for name in ('separate','fused')}
  check(sync())  # Fence default-stream uploads before nonblocking stream capture.
  def operation(name):
   graph_call('nya_interleave_four',(k+255)//256,1,[Q(x),Q(packed_input),Q(k)])
   if name=='separate':
    for weight,target in zip(weights,(gate,up)):
     graph_call('nya_integer_metadata_12_4',(rows+7)//8,1,[Q(weight),Q(packed_input),Q(target),Q(k),Q(stride),U(rows),U(4)])
    graph_call('nya_vector',(rows*4+255)//256,1,[Q(outputs[name]+4),Q(gate),Q(up),Q(rows*4),U(3),C.c_float(1.)])
   else:
    graph_call('nya_q4_batch_gate',(rows+7)//8,1,[Q(weights[0]),Q(weights[1]),Q(packed_input),Q(outputs[name]+4),Q(k),Q(stride),U(rows),U(4)])
  # Establish a separate-pipeline output before any measurements or candidate execution.
  operation('separate');check(sync())
  expected=read(outputs['separate'],len(guard))
  assert expected[:4]==guard[:4] and expected[-4:]==guard[-4:],'Reference guards changed'
  values=array.array('f');values.frombytes(expected[4:-4]);assert all(math.isfinite(v) for v in values)
  samples={name:[] for name in outputs};snapshots=[]
  orders=[['separate','fused'],['fused','separate'],['fused','separate'],['separate','fused']]
  for order in orders:
   for name in order:
    # Reset even previously tested outputs so a no-op cannot inherit valid output.
    check(sync());host=C.create_string_buffer(guard);check(upload(Q(outputs[name]),host,len(guard)));check(sync())
    graph=P();executable=P();check(begin(stream,0))
    for _ in range(10):operation(name)
    check(end(stream,C.byref(graph)));graphs.append(graph)
    check(instantiate(C.byref(executable),graph,0));executables.append(executable)
    for _ in range(100):check(graphlaunch(executable,stream))
    for repeat in range(10):
     check(event_record(a,stream))
     for _ in range(10):check(graphlaunch(executable,stream))
     check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b))
     assert math.isfinite(ms.value) and ms.value>0,'Invalid event duration'
     samples[name].append(ms.value/100)
    check(sync());assert read(outputs[name],len(guard))==expected,(k,rows,name,'Output or guards changed')
    snapshots.append(environment.snapshot())
    check(execdestroy(executable));executables.remove(executable)
    check(graphdestroy(graph));graphs.remove(graph)
    print(k,rows,name,statistics.mean(samples[name]),flush=True)
  results.append(dict(columns=k,rows=rows,batch=4,order=orders,samples_ms=samples,
      means_ms={n:statistics.mean(v) for n,v in samples.items()},
      sample_stddev_ms={n:statistics.stdev(v) for n,v in samples.items()},
      byte_exact=True,environment=snapshots))
  for address in owned:check(free(address))
  owned.clear()
 for name,expected_hash in inputs.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected_hash,name
 result['passed']=True
except BaseException as error:
 result['failure']=str(error) or type(error).__name__;raise
finally:
 check(sync())
 for executable in executables:check(execdestroy(executable))
 for graph in graphs:check(graphdestroy(graph))
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
 result['source_sha256']=hashlib.sha256(source).hexdigest()
 result['options']=[option.decode() for option in opts]
 result['compute_capability']=[major,minor]
 with destination.open('x') as output:json.dump(result,output,indent=2)
