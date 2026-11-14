from pathlib import Path
import hashlib,json,sys
if not __debug__: raise SystemExit('Assertions must remain enabled')
root=Path.cwd();out=Path(__file__).resolve().parent
if (out/'timing-fenced.json').exists():raise SystemExit('Refusing to overwrite evidence')
validation=json.loads((out/'sanitizers-validated.json').read_text())
if validation.get('passed') is not True:raise SystemExit('Sanitizer validation required')
inputs=dict(validation['input_sha256'])
inputs[str((out/'sanitizers-validated.json').relative_to(root))]=hashlib.sha256((out/'sanitizers-validated.json').read_bytes()).hexdigest()
for tool in ('memcheck','racecheck','synccheck'):
 p=out/('gemma-'+tool+'-run.json');proof=json.loads(p.read_text())
 if proof.get('exit_code') != 0:raise SystemExit('Sanitizer process failed')
 inputs.update(proof['input_sha256'])
 for name,h in proof['evidence_sha256'].items():inputs[str((out/name).relative_to(root))]=h
 inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
inputs[str(Path(__file__).resolve().relative_to(root))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for name,h in inputs.items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise SystemExit('Changed input: '+name)
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
envdir=out/'timing-fenced-environment';envdir.mkdir(exist_ok=False)
environment.configure(envdir);environment.idle()
from common import *
stream=P();stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P]);check(stream_create(C.byref(stream),1))
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P]);graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P]);event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P]);event_destroy=bind(cu,'cuEventDestroy_v2',[P])
a=P();b=P();check(event_create(C.byref(a),0));check(event_create(C.byref(b),0))

import statistics,sys
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment

def call(name,gx,gy,bx,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],gx,gy,1,bx,1,1,0,stream,packed,None))
results=[]
try:
 for width,window,kh,batch,position in [(w,win,k,b,p) for w,win,k in [(512,0,1),(256,1024,8)] for b in (2,4) for p in (256,1024,8192)]:
  heads=16;n=position+batch;slots=min(n,window+batch-1) if window else n;stride=kh*width
  q=buffer(floats(batch*heads*width,47).tobytes());k=buffer(floats(slots*stride,31).tobytes());v=buffer(floats(slots*stride,71).tobytes())
  scores=buffer(bytes(batch*heads*n*4));parts=buffer(bytes(batch*heads*4*(width+2)*4));dest=buffer(bytes(batch*heads*width*4))
  check(sync()) # Fence pageable uploads before work on the nonblocking stream.
  for token in range(batch):
   refargs=[Q(q+token*heads*width*4),Q(k),Q(v),Q(parts),Q(scores),U(width),U(heads),U(kh),U(window),U(position+token),U(n),C.c_float(1),Q(0)]
   if window:refargs.append(U(slots))
   call('nya_attention_vec16'+('_ring' if window else '')+'_split',heads,4,256,refargs)
   call('nya_attention_combine',heads,1,width,[Q(parts),Q(dest+token*heads*width*4),U(width)])
  check(sync());reference_bytes=read(dest,batch*heads*width*4)
  samples={name:[] for name in ('baseline','cached')};snapshots=[]
  for order in (list(samples),list(reversed(samples))):
   for variant in order:
    name=('nya_attention_batch_split16' if variant=='baseline' else 'nya_attention_cached_query')+('_ring' if window else '')+'_split'
    args=[Q(q),Q(k),Q(v),Q(parts),Q(scores),U(width),U(heads),U(kh),U(window),U(position),U(n),C.c_float(1),Q(0)]
    if window:args.append(U(slots))
    graph=P();executable=P();check(begin(stream,0))
    for i in range(20):
     call(name,heads,batch*4,256,args)
     call('nya_attention_combine',heads*batch,1,width,[Q(parts),Q(dest),U(width)])
    check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
    for i in range(50):check(graphlaunch(executable,stream))
    for repeat in range(20):
     check(event_record(a,stream))
     for i in range(10):check(graphlaunch(executable,stream))
     check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b));samples[variant].append(ms.value/200)
    check(execdestroy(executable));check(graphdestroy(graph));check(sync());snapshots.append(environment.snapshot())
    actual=read(dest,batch*heads*width*4)
    if reference_bytes is None:reference_bytes=actual
    elif actual!=reference_bytes:
     av=array.array('f');av.frombytes(actual);rv=array.array('f');rv.frombytes(reference_bytes)
     (out/'fenced-output-failure.json').write_text(json.dumps(dict(width=width,position=position,batch=batch,variant=variant,exact=False,differences=sum(a!=b for a,b in zip(av,rv)),max_scaled_error=max(abs(a-b)/(1+abs(b)) for a,b in zip(av,rv))),indent=2))
     raise RuntimeError('Fenced graph output mismatch')
  row={'batch':batch,'width':width,'heads':heads,'kvheads':kh,'position':position,'window':window,'slots':slots,'samples_ms':samples,'means_ms':{name:statistics.mean(v) for name,v in samples.items()},'sample_stdev_ms':{name:statistics.stdev(v) for name,v in samples.items()},'environment':snapshots}
  results.append(row);print(width,row['means_ms'],flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
result={'accepted':False,'reason':'Synthetic repeated-bank kernel diagnostic with desktop activity; not end-to-end model throughput or an accepted performance score.','source_sha256':hashlib.sha256(source).hexdigest(),'results':results}
result['input_sha256']=inputs
result['changed_inputs']=[name for name,h in inputs.items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h]
result['operations_per_sample']=200
result['warmup_operations_per_order']=1000
result['output_parity']=True
(out/'timing-fenced.json').write_text(json.dumps(result,indent=2)+'\n')
if result['changed_inputs']:raise SystemExit('Inputs changed during timing')
