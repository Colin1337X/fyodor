"""Exact raw Q4 parity, finite-coefficient edges, and memory guards; no timing."""
import json
import hashlib
import ctypes as C
from pathlib import Path
import sys

out = Path(__file__).resolve().parent
destination = out / ('memcheck-oracle.json' if sys.argv[1:] == ['--memcheck'] else 'oracle.json')
if sys.argv[1:] not in ([], ['--memcheck']):
    raise SystemExit('Expected no arguments or --memcheck')
if destination.exists():
    raise SystemExit('Refusing to overwrite correctness evidence')
audit_path = out.parent / 'interleave-pair-20261114/model-memcheck-short/result.json'
if not audit_path.exists():
    raise SystemExit('Wait for the active FFN-reuse sanitizer to complete before GPU execution')
assert json.loads(audit_path.read_text())['passed'] is True
pack_proof_path = out / 'pack-build.json'
pack_proof = json.loads(pack_proof_path.read_text())
assert pack_proof['passed'] is True
repo = out.parents[2]
for name, expected in pack_proof['sha256'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected, name
packer = C.CDLL(str(repo/'build-cuda/interleaved-byte-integer-20261114/pack-c17.dll'))
packer.nya_test_pack.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
packer.nya_test_pack.restype=C.c_int

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
        packed_stride=columns//256*160
        packed_bytes=rows*packed_stride
        host_source=C.create_string_buffer(bytes(weights_host))
        host_packed=C.create_string_buffer(b'\xcd'*(packed_bytes+32),packed_bytes+32)
        assert packer.nya_test_pack(C.byref(host_packed,16),packed_bytes,host_source,len(weights_host))==0
        assert host_packed.raw[:16]==host_packed.raw[-16:]==b'\xcd'*16
        byte_weights=buffer(host_packed.raw[16:-16])
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
        for name in ('nya_interleaved_byte_integer_12_4',):
            actual=buffer(guard)
            tile=4 if name.startswith('nya_integer_launch_4_') else 8
            call(name,(rows+tile-1)//tile,1,[Q(byte_weights),Q(packed+16),Q(actual+4),Q(columns),Q(packed_stride),U(rows),U(4)])
            check(sync())
            assert read(actual,len(guard))==expected, (name,columns,rows,edge)
        result['cases'].append(dict(columns=columns,rows=rows,half_bits=edge,byte_exact=True,
            guards_clear=True,float64_checked=columns==256,maximum_fraction_of_float64_bound=max_scaled))
        for address in owned: check(free(address))
        owned.clear()
        if len(result['cases'])%20==0 or edge is None:
            print('Passed',len(result['cases']),'of',len(cases),flush=True)
    result['passed']=True
except BaseException as error:
    result['failure']=str(error) or type(error).__name__
    raise
finally:
    for address in owned: check(free(address))
    check(unload(module)); check(release(device))
    result['packer_inputs']=pack_proof['sha256']
    result['harness_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__),out/'common.py',pack_proof_path,out/'pack.c')}
    with destination.open('x') as output: json.dump(result,output,indent=2)
