"""Paired synthetic RMSNorm/layout timings; never accepted model throughput."""
import hashlib,json,sys
from pathlib import Path
if not __debug__:raise SystemExit('Assertions required')
out=Path(__file__).resolve().parent;root=out.parents[2];destination=out/'timing-upload-sync.json'
if destination.exists():raise SystemExit('Refusing evidence overwrite')
inputs={Path(__file__).resolve()}
for tool in ('memcheck','racecheck','synccheck'):
    p=out/(tool+'-run.json')
    if not p.exists():raise SystemExit('Complete '+tool+' before timing')
    proof=json.loads(p.read_text());inputs.add(p)
    if proof.get('passed') is not True or proof.get('exit_code')!=0:raise ValueError('Failed sanitizer')
    for name,h in proof['input_sha256'].items():
        path=root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError(name)
        inputs.add(path)
    for name,h in proof['evidence_sha256'].items():
        path=out/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError(name)
        inputs.add(path)
sys.path.insert(0,str(out.parent/'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):raise SystemExit('Another GPU workload remains')
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
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
import statistics
random.seed(713);results=[];F=C.c_float
try:
    for n in (256,512,1024,2048,3840,4096,15360):
        for batch in (2,4):
            x=buffer(array.array('f',(random.uniform(-3,3) for _ in range(n*batch))).tobytes())
            w=buffer(array.array('f',(random.uniform(.5,1.5) for _ in range(n))).tobytes())
            y=buffer(bytes(n*batch*4));packed=buffer(bytes(n*batch*4))
            check(sync()) # Finish default-stream uploads before nonblocking stream use.
            names=['separate','fused'];orders=[names,list(reversed(names)),list(reversed(names)),names]
            samples={name:[] for name in names};snapshots=[];expected=None
            for order in orders:
                for name in order:
                    graph=P();executable=P();check(begin(stream,0))
                    for repeat in range(10):
                        if name=='separate':
                            graph_call('nya_norm',batch,1,[Q(x),Q(y),Q(w),Q(n),U(0),F(1e-6)])
                            graph_call('nya_interleave_two' if batch==2 else 'nya_interleave_four',(n+255)//256,1,[Q(y),Q(packed),Q(n)])
                        else:
                            graph_call('nya_norm_interleaved',batch,1,[Q(x),Q(y),Q(packed),Q(w),Q(n),U(0),F(1e-6),U(batch)])
                    check(end(stream,C.byref(graph)));check(instantiate(C.byref(executable),graph,0))
                    for repeat in range(100):check(graphlaunch(executable,stream))
                    for repeat in range(10):
                        check(event_record(a,stream))
                        for i in range(10):check(graphlaunch(executable,stream))
                        check(event_record(b,stream));check(event_sync(b));ms=F();check(elapsed(C.byref(ms),a,b))
                        assert math.isfinite(ms.value) and ms.value>0
                        samples[name].append(ms.value/100)
                    check(execdestroy(executable));check(graphdestroy(graph));check(sync())
                    actual=(read(y,n*batch*4),read(packed,n*batch*4))
                    values=array.array('f');values.frombytes(actual[0])
                    oracle=array.array('f',(values[t*n+i] for i in range(n) for t in range(batch))).tobytes()
                    assert actual[1]==oracle, 'Graph layout differs from CPU permutation'
                    if expected is None:expected=actual
                    elif actual!=expected:
                        details=[]
                        for layout,(lhs,rhs) in enumerate(zip(expected,actual)):
                            av=array.array('f');av.frombytes(lhs);bv=array.array('f');bv.frombytes(rhs)
                            indices=[i for i,(v,z) in enumerate(zip(av,bv)) if struct.pack('<f',v)!=struct.pack('<f',z)]
                            details.append(dict(layout=layout,mismatches=len(indices),maximum_scaled_error=max((abs(av[i]-bv[i])/(1+abs(av[i])) for i in indices),default=0),examples=[dict(index=i,reference=av[i],candidate=bv[i]) for i in indices[:12]]))
                        (out/'timing-upload-sync-mismatch.json').write_text(json.dumps(dict(accepted_benchmark=False,width=n,batch=batch,mode=name,details=details,completed=results,input_sha256=hashes),indent=2))
                        raise ValueError('Timed outputs differ; diagnostic retained')
                    snapshots.append(environment.snapshot())
            means={name:statistics.mean(values) for name,values in samples.items()}
            results.append(dict(width=n,batch=batch,order=[name for order in orders for name in order],samples_ms=samples,means_ms=means,stdev_ms={name:statistics.stdev(values) for name,values in samples.items()},environment=snapshots))
            print(n,batch,means,flush=True)
            for address in owned:check(free(address))
            owned.clear()
finally:
    for address in owned:check(free(address))
    check(event_destroy(a));check(event_destroy(b));check(stream_destroy(stream));check(unload(module));check(release(device))
for name,h in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
with destination.open('x') as output:
    json.dump(dict(accepted_benchmark=False,reason='Synthetic repeated buffers, no continuous activity acceptance; not model throughput',operations_per_sample=100,warmup_operations_per_order=1000,results=results,input_sha256=hashes),output,indent=2)
