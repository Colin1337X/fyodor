"""Owned CUDA transfer attribution control; no model or performance acceptance."""
import ctypes as C,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
if len(sys.argv)!=2 or not sys.argv[1].isascii() or not sys.argv[1].isalnum():
    raise SystemExit('Supply a unique alphanumeric label')
folder=HERE/sys.argv[1]
folder.mkdir()
source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result=dict(accepted_benchmark=False,passed=False,process_id=os.getpid(),phases=[],
            limitation='Transfer-only correlation does not establish causal System attribution or model throughput.')
lib=C.WinDLL('nvcuda.dll');P=C.c_void_p;I=C.c_int;U=C.c_uint;Z=C.c_size_t;Q=C.c_uint64

def bind(name,args):
    f=getattr(lib,name);f.argtypes=args;f.restype=I;return f

def check(status):
    if status:raise RuntimeError('CUDA error '+str(status))

init=bind('cuInit',[U]);deviceget=bind('cuDeviceGet',[C.POINTER(I),I])
retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(P),I]);release=bind('cuDevicePrimaryCtxRelease_v2',[I])
current=bind('cuCtxSetCurrent',[P]);alloc=bind('cuMemAlloc_v2',[C.POINTER(Q),Z]);free=bind('cuMemFree_v2',[Q])
hostalloc=bind('cuMemAllocHost_v2',[C.POINTER(P),Z]);hostfree=bind('cuMemFreeHost',[P])
upload=bind('cuMemcpyHtoD_v2',[Q,P,Z]);download=bind('cuMemcpyDtoH_v2',[P,Q,Z]);sync=bind('cuCtxSynchronize',[])
device=I();context=P();gpu=Q();pinned=P();size=4*1024*1024
payload=bytes(range(256))*(size//256);original=C.create_string_buffer(payload);pageable=C.create_string_buffer(size)
try:
    check(init(0));check(deviceget(C.byref(device),0));check(retain(C.byref(context),device));check(current(context))
    check(alloc(C.byref(gpu),size));check(hostalloc(C.byref(pinned),size));check(upload(gpu,original,size));check(sync())
    time.sleep(2)
    for round_,order in enumerate((('idle','pageable','pinned'),('pinned','pageable','idle'),('idle','pageable','pinned'))):
        for mode in order:
            phase=dict(round=round_,mode=mode,start=time.perf_counter(),transfers=0,bytes_per_transfer=size)
            deadline=phase['start']+6
            while time.perf_counter()<deadline:
                if mode!='idle':
                    check(download(pinned if mode=='pinned' else C.cast(pageable,P),gpu,size));phase['transfers']+=1
                time.sleep(.02)
            phase['end']=time.perf_counter()
            if mode!='idle':
                actual=C.string_at(pinned,size) if mode=='pinned' else pageable.raw
                if actual!=payload:raise ValueError('Transfer content mismatch')
            phase['content_verified']=mode!='idle'
            result['phases'].append(phase)
            with (folder/'phases.jsonl').open('a') as f:f.write(json.dumps(phase)+'\n')
            print(round_,mode,phase['transfers'],flush=True)
    result['passed']=True
except BaseException as error:
    result['error']=str(error);raise
finally:
    errors=[]
    for address,operation in ((pinned,hostfree),(gpu,free)):
        if address.value:
            status=operation(address)
            if status:errors.append(status)
    if context.value:
        status=release(device)
        if status:errors.append(status)
    result['cleanup_errors']=errors
    result['passed']=result['passed'] and not errors
    result['source_unchanged']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source_hash
    result['sha256']={str(Path(__file__).resolve()):source_hash}
    log=folder/'phases.jsonl'
    if log.exists():result['sha256'][str(log.resolve())]=hashlib.sha256(log.read_bytes()).hexdigest()
    (folder/'result.json').write_text(json.dumps(result,indent=2))
