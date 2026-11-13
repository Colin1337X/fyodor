from common import *
stream=P();stream_create=bind(cu,'cuStreamCreate',[C.POINTER(P),U]);stream_destroy=bind(cu,'cuStreamDestroy_v2',[P]);check(stream_create(C.byref(stream),1))
begin=bind(cu,'cuStreamBeginCapture_v2',[P,I]);end=bind(cu,'cuStreamEndCapture',[P,C.POINTER(P)])
instantiate=bind(cu,'cuGraphInstantiateWithFlags',[C.POINTER(P),P,Q]);graphlaunch=bind(cu,'cuGraphLaunch',[P,P]);graphdestroy=bind(cu,'cuGraphDestroy',[P]);execdestroy=bind(cu,'cuGraphExecDestroy',[P])
event_create=bind(cu,'cuEventCreate',[C.POINTER(P),U]);event_record=bind(cu,'cuEventRecord',[P,P]);event_sync=bind(cu,'cuEventSynchronize',[P]);elapsed=bind(cu,'cuEventElapsedTime',[C.POINTER(C.c_float),P,P]);event_destroy=bind(cu,'cuEventDestroy_v2',[P])
a=P();b=P();check(event_create(C.byref(a),0));check(event_create(C.byref(b),0))


def graph_call(name,blocks,grid_y,args):
 packed=(P*len(args))(*(C.cast(C.byref(v),P) for v in args))
 check(launch(funcs[name],blocks,grid_y,1,256,1,1,0,stream,packed,None))
import statistics,sys
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
random.seed(713);results=[]
try:
 for dtype,k,rows in [(12,3840,15360),(14,15360,3840)]:
  block_bytes=144 if dtype==12 else 210
  stride=k//256*block_bytes;w=bytearray(random.randbytes(rows*stride))
  for offset in range(0,len(w),block_bytes):
   if dtype==12:w[offset:offset+4]=struct.pack('<ee',.03125,.00390625)
   else:w[offset+208:offset+210]=struct.pack('<e',.03125)
  weights=buffer(w);del w
  x=buffer(array.array('f',(random.uniform(-1,1) for _ in range(k*4))).tobytes());y=buffer(bytes(rows*4*4))
  for batch in (2,4):
   names=['native','bound3','bound4','bound6'];samples={n:[] for n in names};snapshots=[]
   for order in [names,list(reversed(names))]:
    for name in order:
     count=batch
     kernel=f'nya_small_reuse_{dtype}_{count}' if name=='native' else f'nya_reuse_launch_{name[-1]}_{dtype}_{count}'
     graph=P();executable=P();check(begin(stream,0))
     for i in range(10):graph_call(kernel,(rows+7)//8,(batch+count-1)//count,[Q(weights),Q(x),Q(y),Q(k),Q(stride),U(rows),U(batch)])
     check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
     for i in range(3):check(graphlaunch(executable,stream))
     for repeat in range(5):
      check(event_record(a,stream))
      for i in range(10):check(graphlaunch(executable,stream))
      check(event_record(b,stream));check(event_sync(b));ms=C.c_float();check(elapsed(C.byref(ms),a,b));samples[name].append(ms.value/100)
     check(execdestroy(executable));check(graphdestroy(graph));check(sync());snapshots.append(environment.snapshot())
     print(dtype,k,rows,batch,name,statistics.mean(samples[name]),flush=True)
   results.append(dict(dtype=dtype,columns=k,rows=rows,batch=batch,samples_ms=samples,means_ms={n:statistics.mean(v) for n,v in samples.items()},sample_stddev_ms={n:statistics.stdev(v) for n,v in samples.items()},environment=snapshots))
  for address in owned:check(free(address))
  owned.clear()
finally:
 for address in owned:check(free(address))
 check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
Path(__file__).with_name('probe-results.json').write_text(json.dumps(dict(accepted=False,reason='Synthetic repeated weight banks with desktop GPU activity; diagnostic only',source_sha256=hashlib.sha256(source).hexdigest(),results=results),indent=2)+'\n')
