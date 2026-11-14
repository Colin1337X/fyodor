"""Q6 batch-two/four byte parity and guards; no timing or runtime admission."""
import json
from pathlib import Path
import sys

out = Path(__file__).resolve().parent
if sys.argv[1:] not in (['--memcheck'], ['--racecheck'], ['--synccheck']):
    raise SystemExit('Expected no arguments or --memcheck')
destination = out / (sys.argv[1][2:]+'-oracle.json')
if destination.exists():
    raise SystemExit('Refusing to overwrite correctness evidence')
import hashlib
root = Path.cwd()
if not __debug__:
    raise SystemExit('Assertions required')
input_hashes = {}
for name in ('compile.json', 'assembly.json', 'oracle-control.json'):
    proof = json.loads((out/name).read_text())
    if name=='oracle-control.json' and proof.get('passed') is not True:
        raise ValueError('Host oracle control failed')
    for path, digest in proof['sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed prerequisite: '+path)
        input_hashes[path] = digest
    input_hashes[str((out/name).relative_to(root))] = hashlib.sha256((out/name).read_bytes()).hexdigest()
sys.path.insert(0, str(out.parent/'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
# Import creates a CUDA context; all pending-work checks precede it.
from common import *

paths = [root/'CUDA'/n for n in ('matvec.cu','gemm.cu','resident.cu')]
paths += [root/'optcpp/matvec'/n for n in ('small_batch.cu','small_batch_reuse.cu',
    'interleaved_reuse.cu','interleaved_two.cu','interleaved_q6_integer.cu','q6_pair_rows_four.cu')]
paths += [Path(__file__),out/'common.py',Path(environment.__file__)]
input_hashes.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
library=C.CDLL(str(root/'build-cuda/q6-pair-four-oracle-20261114/oracle-c17.dll'))
oracle=library.q6_oracle
oracle.argtypes=[P,P,Z,Z,P,P];oracle.restype=I
random.seed(713)
shapes = [(256,r) for r in (1,7,9,15,16,17,63,64,65)]
shapes += [(k,17) for k in (512,768,1024,1280,1792)]
shapes += [(3840,2048),(8192,3840),(15360,3840)]
bits = [0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3c00,0xbc00,0x7bff,0xfbff,0x27ae,0xa75a]
cases = [(k,r,None) for k,r in shapes]
cases += [(256,9,(d,sc,q)) for d in bits for sc in (-128,-1,0,1,127) for q in (-32,-1,0,1,31)]
result = dict(accepted_benchmark=False, passed=False, source_sha256=hashlib.sha256(source).hexdigest(),
              options=[option.decode() for option in opts], cases=[])
try:
    for columns,rows,edge in cases:
        stride = columns//256*210
        host = bytearray(random.randbytes(rows*stride))
        for offset in range(0,len(host),210):
            if edge is None:
                host[offset+208:offset+210] = struct.pack('<e',(.03125,.03,-.017,.125)[(offset//210)%4])
            else:
                d,sc,q = edge; encoded = q+32
                host[offset:offset+128] = bytes([(encoded&15)*17])*128
                host[offset+128:offset+192] = bytes([(encoded>>4)*85])*64
                host[offset+192:offset+208] = bytes([sc&255])*16
                host[offset+208:offset+210] = struct.pack('<H',d)
        for batch in (4,):
            weights = buffer(host)
            inputs = array.array('f',(random.uniform(-1,1) for _ in range(columns*batch)))
            x = buffer(inputs.tobytes())
            packed_guard = array.array('f',[12345.]*(columns*batch+8)).tobytes()
            packed = buffer(packed_guard)
            call('nya_interleave_two' if batch==2 else 'nya_interleave_four',(columns+255)//256,1,
                 [Q(x),Q(packed+16),Q(columns)])
            check(sync())
            assert read(packed,len(packed_guard)) == array.array('f',[12345.]*4+
                [inputs[t*columns+j] for j in range(columns) for t in range(batch)]+[12345.]*4).tobytes()
            guard = array.array('f',[12345.]*(rows*batch+2)).tobytes()
            reference = buffer(guard)
            call('nya_small_batch_14',(rows+7)//8,batch,
                 [Q(weights),Q(x),Q(reference+4),Q(columns),Q(stride),U(rows),U(batch)])
            check(sync()); expected = read(reference,len(guard))
            values = array.array('f'); values.frombytes(expected)
            assert all(math.isfinite(value) for value in values)
            assert expected[:4]==guard[:4] and expected[-4:]==guard[-4:]
            host_weights=C.create_string_buffer(bytes(host))
            host_inputs=C.create_string_buffer(inputs.tobytes())
            double_expected=(C.c_double*(rows*batch))()
            double_bounds=(C.c_double*(rows*batch))()
            assert oracle(host_weights,host_inputs,columns,rows,double_expected,double_bounds)==0
            maximum=0.
            for i in range(rows*batch):
                error=abs(values[1+i]-double_expected[i])
                assert error<=double_bounds[i],(columns,rows,edge,i,error,double_bounds[i])
                maximum=max(maximum,error/double_bounds[i])
            if columns==256:
                for token in range(batch):
                    for row in range(rows):
                        b=host[row*stride:(row+1)*stride]; products=[]
                        d=struct.unpack('<e',b[208:210])[0]
                        for j in range(columns):
                            half=j//128; quarter=j%128//32; lane=j%32
                            lo=b[half*64+quarter%2*32+lane]; hi=b[128+half*32+lane]
                            q=(((lo>>(quarter//2*4))&15)|(((hi>>(quarter*2))&3)<<4))-32
                            sc=b[192+j//16]; sc=sc if sc<128 else sc-256
                            products.append(d*sc*q*inputs[token*columns+j])
                        error=abs(values[1+token*rows+row]-math.fsum(products))
                        bound=2e-6*math.fsum(abs(v) for v in products)+1e-6
                        assert error<=bound, (columns,rows,edge,batch,token,row,error,bound)
                        maximum=max(maximum,error/bound)
            names = ('nya_interleaved_two_14_2','nya_interleaved_q6_integer_two_14_2') if batch==2 else (
                'nya_interleaved_q6_integer_four_14_4','nya_q6_pair_rows_four')
            for name in names:
                actual = buffer(guard)
                call(name,(rows+15)//16 if name=='nya_q6_pair_rows_four' else (rows+7)//8,1,[Q(weights),Q(packed+16),Q(actual+4),Q(columns),Q(stride),U(rows),U(batch)])
                check(sync())
                assert read(actual,len(guard))==expected, (name,columns,rows,edge)
            assert read(weights,len(host))==bytes(host)
            assert read(x,len(inputs)*4)==inputs.tobytes()
            result['cases'].append(dict(columns=columns,rows=rows,batch=batch,coefficient_case=edge,
                byte_exact=True,guards_clear=True,float64_checked=True,maximum_fraction_of_float64_bound=maximum))
            for address in owned: check(free(address))
            owned.clear()
        if edge is None or len(result['cases'])%40==0:
            print('Passed',len(result['cases']),'of',len(cases),flush=True)
    for name,digest in input_hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    result['passed']=True
except BaseException as error:
    result['failure']=str(error) or type(error).__name__
    raise
finally:
    for address in owned: check(free(address))
    check(unload(module));check(release(device))
    result['sha256']=input_hashes
    result['harness_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__),out/'common.py')}
    with destination.open('x') as output:json.dump(result,output,indent=2)
