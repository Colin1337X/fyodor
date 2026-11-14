"""Compare dual-output RMSNorm with the separate validated operations."""
import hashlib,json,sys,itertools
from pathlib import Path
if not __debug__:raise SystemExit('Assertions required')
out=Path(__file__).resolve().parent;root=out.parents[2]
mode=sys.argv[1:] if len(sys.argv)>1 else []
if mode not in ([],['memcheck'],['racecheck'],['synccheck']):raise SystemExit('Invalid sanitizer label')
destination=out/((mode[0]+'-oracle.json') if mode else 'oracle.json')
if destination.exists():raise SystemExit('Refusing to overwrite evidence')
inputs={Path(__file__).resolve(),out/'common.py'}
for path,key in ((out/'compile.json','compile_passed'),(out.parent/'current-decode-profile-20261114/short/result.json','passed')):
    proof=json.loads(path.read_text());inputs.add(path)
    if proof.get(key) is not True:raise ValueError('Failed prerequisite')
    for name,expected in proof['sha256'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(name)
        inputs.add(p)
sys.path.insert(0,str(out.parent/'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):raise SystemExit('Another GPU workload remains')
inputs.update(root/name for name in ('optcpp/matvec/interleaved_reuse.cu','optcpp/matvec/interleaved_two.cu'))
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
from common import *
random.seed(713);F=C.c_float
widths=(1,7,31,32,127,256,257,512,1024,2048,3840,4096,15360)
result=dict(passed=False,accepted_benchmark=False,cases=[],rejections=[],input_sha256=hashes)
try:
    for n,batch,weight_mode,inplace,pattern,epsilon in itertools.product(widths,(2,4),('none','f32','f16','bf16'),(False,True),('random','zero','range'),(1e-6,1e-5)):
        values=array.array('f',(random.uniform(-3,3) if pattern=='random' else 0 if pattern=='zero' else (1e10,-1e10,1e-20,-1e-20)[i%4] for i in range(n*batch)))
        guard=array.array('f',[12345.]*4).tobytes();initial=guard+values.tobytes()+guard
        x=buffer(initial);reference=buffer(guard+bytes(n*batch*4)+guard)
        weights=array.array('f',((1.,-.5,0.,1.5,2**-24)[i%5] for i in range(n)))
        dtype={'none':0,'f32':0,'f16':1,'bf16':30}[weight_mode]
        w=0 if weight_mode=='none' else buffer(weights.tobytes() if weight_mode=='f32' else b''.join(struct.pack('<e',v) if weight_mode=='f16' else struct.pack('<I',struct.unpack('<I',struct.pack('<f',v))[0])[2:] for v in weights))
        packed_reference=buffer(guard+bytes(n*batch*4)+guard)
        call('nya_norm',batch,1,[Q(x+16),Q(reference+16),Q(w),Q(n),U(dtype),F(epsilon)])
        call('nya_interleave_two' if batch==2 else 'nya_interleave_four',(n+255)//256,1,[Q(reference+16),Q(packed_reference+16),Q(n)])
        check(sync());expected=read(reference,len(initial));expected_packed=read(packed_reference,len(initial))
        assert expected[:16]==guard and expected[-16:]==guard
        normalized=array.array('f');normalized.frombytes(expected[16:-16]);assert all(math.isfinite(v) for v in normalized)
        cpu_packed=array.array('f',(normalized[t*n+i] for i in range(n) for t in range(batch))).tobytes()
        assert expected_packed==guard+cpu_packed+guard
        actual=x if inplace else buffer(guard+bytes(n*batch*4)+guard)
        packed=buffer(guard+bytes(n*batch*4)+guard)
        call('nya_norm_interleaved',batch+1,1,[Q(x+16),Q(actual+16),Q(packed+16),Q(w),Q(n),U(dtype),F(epsilon),U(batch)])
        check(sync())
        assert read(actual,len(initial))==expected,(n,batch,weight_mode,inplace,pattern,epsilon,'row-major')
        assert read(packed,len(initial))==expected_packed,(n,batch,weight_mode,inplace,pattern,epsilon,'interleaved')
        if not inplace:assert read(x,len(initial))==initial
        result['cases'].append(dict(width=n,batch=batch,weight=weight_mode,inplace=inplace,pattern=pattern,epsilon=epsilon,byte_exact=True,guards_clear=True))
        for address in owned:check(free(address))
        owned.clear()
        if len(result['cases'])%200==0:print('Passed',len(result['cases']),'of 1248',flush=True)
    for batch in (0,1,3):
        guard=array.array('f',[12345.]*16).tobytes();a=buffer(guard);b=buffer(guard)
        call('nya_norm_interleaved',1,1,[Q(0),Q(a+16),Q(b+16),Q(0),Q(256),U(0),F(1e-6),U(batch)])
        check(sync());assert read(a,len(guard))==guard and read(b,len(guard))==guard
        result['rejections'].append(dict(batch=batch,unchanged=True))
        for address in owned:check(free(address))
        owned.clear()
    for name,h in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    result['passed']=True
finally:
    for address in owned:check(free(address))
    check(unload(module));check(release(device))
    destination.write_text(json.dumps(result,indent=2))
print('1248 exact RMSNorm cases and three rejection cases passed; no timing acceptance')
