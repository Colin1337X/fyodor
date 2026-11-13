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
  samples={name:[] for name in ('unsplit','split')};snapshots=[]
  for order in (list(samples),list(reversed(samples))):
   for variant in order:
    name=('nya_attention_vec16' if variant=='unsplit' else 'nya_attention_batch_split16')+('_ring' if window else '')+('_split' if variant=='split' else '')
    args=[Q(q),Q(k),Q(v),Q(parts if variant=='split' else dest),Q(scores),U(width),U(heads),U(kh),U(window),U(position),U(n),C.c_float(1),Q(0)]
    if window:args.append(U(slots))
    graph=P();executable=P();check(begin(stream,0))
    for i in range(20):
     call(name,heads,batch*(4 if variant=='split' else 1),256,args)
     if variant=='split':call('nya_attention_combine',heads*batch,1,width,[Q(parts),Q(dest),U(width)])
    check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
    for i in range(3):check(graphlaunch(executable,stream))
    for repeat in range(5):
     check(event_record(a,stream))
     for i in range(10):check(graphlaunch(executable,stream))
     check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b));samples[variant].append(ms.value/200)
    check(execdestroy(executable));check(graphdestroy(graph));check(sync());snapshots.append(environment.snapshot())
  row={'batch':batch,'width':width,'heads':heads,'kvheads':kh,'position':position,'window':window,'slots':slots,'samples_ms':samples,'means_ms':{name:statistics.mean(v) for name,v in samples.items()},'sample_stdev_ms':{name:statistics.stdev(v) for name,v in samples.items()},'environment':snapshots}
  results.append(row);print(width,row['means_ms'],flush=True)
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
result={'accepted':False,'reason':'Synthetic repeated-bank kernel diagnostic with desktop activity; not end-to-end model throughput or an accepted performance score.','source_sha256':hashlib.sha256(source).hexdigest(),'results':results}
Path(__file__).with_name('probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
