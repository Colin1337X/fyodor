"""Exact raw Q4 parity, finite-coefficient edges, and memory guards; no timing."""
import json
import hashlib
from pathlib import Path
import sys

if not __debug__: raise SystemExit('Assertions are required')
out = Path(__file__).resolve().parent
destination = out / ((sys.argv[1][2:]+'-oracle.json') if len(sys.argv)==2 else 'oracle.json')
if sys.argv[1:] not in ([], ['--memcheck'], ['--racecheck'], ['--synccheck']):
    raise SystemExit('Expected no arguments or --memcheck')
if destination.exists():
    raise SystemExit('Refusing to overwrite correctness evidence')
audit_path = out.parent / 'batch-graphs-20261114/boundary-memcheck-c23-512-ring1/result.json'
if not audit_path.exists():
    raise SystemExit('Wait for the completed boundary sanitizer matrix before starting another GPU workload')
for standard in ('c17','c23'):
    for width in ('256','512'):
        for ring in ('0','1'):
            boundary = out.parent / f'batch-graphs-20261114/boundary-memcheck-{standard}-{width}-ring{ring}/result.json'
            if not boundary.exists():
                raise SystemExit('Pending boundary sanitizer: '+str(boundary))
            state=json.loads(boundary.read_text())
            if state.get('passed') is not True or state.get('exit_code')!=0:
                raise ValueError('Failed boundary sanitizer: '+str(boundary))
audit = json.loads(audit_path.read_text())
assert audit['passed'] is True and audit['exit_code'] == 0
root=out.parents[2]
for proof_name in ('compile.json',):
    proof=json.loads((out/proof_name).read_text())
    assert proof.get('compile_passed',proof.get('passed')) is True
    for name, expected in proof['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
sys.path.insert(0,str(out.parent/'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')

# Snapshot every compiled source and prerequisite before creating a context.
input_paths=[root/'CUDA'/name for name in ('matvec.cu','gemm.cu','resident.cu')]
input_paths += [root/'optcpp/matvec'/name for name in
    ('small_batch.cu','small_batch_reuse.cu','interleaved_reuse.cu','interleaved_integer.cu',
     'integer_launch.cu','integer_metadata.cu','integer_metadata_unroll2.cu')]
input_paths += [Path(__file__).resolve(),out/'common.py',out/'compile.json']
input_paths += [out.parent/f'batch-graphs-20261114/boundary-memcheck-{standard}-{width}-ring{ring}/result.json'
    for standard in ('c17','c23') for width in ('256','512') for ring in ('0','1')]
input_hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}

# This module creates the GPU context. All pending-run checks precede its import.
from common import *

random.seed(713)
shapes = [(256,1),(256,7),(256,9),(3840,512),(3840,2048),(3840,4096),
          (3840,8192),(3840,15360),(4096,3840),(8192,3840),(15360,3840)]
bits = [0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
        0x3c00,0xbc00,0x7bff,0xfbff,0x27ae,0xa75a]
cases = [(columns,rows,None) for columns,rows in shapes]
cases += [(256,9,(d,m)) for d in bits for m in bits]
result = dict(accepted_benchmark=False, source_sha256=hashlib.sha256(source).hexdigest(),
              options=[option.decode() for option in opts], cases=[], passed=False)
try:
    for columns,rows,edge in cases:
        stride=columns//256*144
        weights_host=bytearray(random.randbytes(rows*stride))
        for offset in range(0,len(weights_host),144):
            if edge is None:
                weights_host[offset:offset+4]=struct.pack('<ee',(.03125,.03,-.017,.125)[(offset//144)%4],
                    (.00390625,.004,.002,-.001)[(offset//144)%4])
            else:
                weights_host[offset:offset+4]=struct.pack('<HH',*edge)
        weights=buffer(weights_host)
        inputs=array.array('f',(random.uniform(-1,1) for _ in range(columns*4)))
        x=buffer(inputs.tobytes())
        packed_guard=array.array('f',[12345.]*(columns*4+8)).tobytes()
        packed=buffer(packed_guard)
        call('nya_interleave_four',(columns+255)//256,1,[Q(x),Q(packed+16),Q(columns)])
        check(sync())
        assert read(packed,len(packed_guard)) == array.array('f',[12345.]*4+
            [inputs[t*columns+j] for j in range(columns) for t in range(4)]+[12345.]*4).tobytes()
        guard=array.array('f',[12345.]*(rows*4+2)).tobytes()
        reference=buffer(guard)
        call('nya_small_batch_12',(rows+7)//8,4,[Q(weights),Q(x),Q(reference+4),Q(columns),Q(stride),U(rows),U(4)])
        check(sync()); expected=read(reference,len(guard))
        values=array.array('f'); values.frombytes(expected)
        assert all(math.isfinite(value) for value in values)
        assert expected[:4]==guard[:4] and expected[-4:]==guard[-4:]
        max_scaled=0.
        if columns==256:
            for token in range(4):
                for row in range(rows):
                    b=weights_host[row*stride:(row+1)*stride]; products=[]
                    d,m=struct.unpack('<ee',b[:4])
                    for j in range(columns):
                        g=j//32
                        sc=(b[4+g]&63) if g<4 else (b[8+g]&15)|((b[g]>>6)<<4)
                        mn=(b[8+g]&63) if g<4 else (b[8+g]>>4)|((b[4+g]>>6)<<4)
                        q=(b[16+g//2*32+j%32]>>(g%2*4))&15
                        products.append((d*sc*q-m*mn)*inputs[token*columns+j])
                    error=abs(values[1+token*rows+row]-math.fsum(products))
                    bound=2e-6*math.fsum(abs(value) for value in products)+1e-6
                    assert error<=bound, (columns,rows,edge,token,row,error,bound)
                    max_scaled=max(max_scaled,error/bound)
        for name in ('nya_integer_metadata_12_4','nya_integer_metadata_unroll2'):
            actual=buffer(guard)
            tile=4 if name.startswith('nya_integer_launch_4_') else 8
            call(name,(rows+tile-1)//tile,1,[Q(weights),Q(packed+16),Q(actual+4),Q(columns),Q(stride),U(rows),U(4)])
            check(sync())
            assert read(actual,len(guard))==expected, (name,columns,rows,edge)
        result['cases'].append(dict(columns=columns,rows=rows,half_bits=edge,byte_exact=True,
            guards_clear=True,float64_checked=columns==256,maximum_fraction_of_float64_bound=max_scaled))
        for address in owned: check(free(address))
        owned.clear()
        if len(result['cases'])%20==0 or edge is None:
            print('Passed',len(result['cases']),'of',len(cases),flush=True)
    for name,expected_hash in input_hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected_hash, 'Input changed: '+name
    result['passed']=True
except BaseException as error:
    result['failure']=str(error) or type(error).__name__
    raise
finally:
    for address in owned: check(free(address))
    check(unload(module)); check(release(device))
    result['input_sha256']=input_hashes
    result['harness_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__),out/'common.py')}
    with destination.open('x') as output: json.dump(result,output,indent=2)
