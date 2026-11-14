"""Build and check the production C packer without any CUDA context."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

root=Path.cwd()
out=Path(__file__).resolve().parent
work=root/'build-cuda/interleaved-byte-integer-20261114'
assert not work.exists() and not (out/'pack-build.json').exists()
work.mkdir()
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=os.environ.copy()
env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
commands=[]
inputs=[Path(__file__).resolve(),out/'pack.c',root/'CUDA/q4_byte_pack.h']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
for standard in ('c17','c23'):
    dll=work/f'pack-{standard}.dll'
    command=[str(compiler),f'-std={standard}','-O3','-Wall','-Wextra','-Wpedantic',
        '-Wconversion','-Wshadow','-Werror','-shared','-I',str(root/'CUDA'),
        str(out/'pack.c'),'-o',str(dll)]
    subprocess.run(command,env=env,check=True)
    commands.append(command)
    lib=C.CDLL(str(dll))
    lib.nya_test_pack_size.argtypes=[C.c_size_t]
    lib.nya_test_pack_size.restype=C.c_size_t
    lib.nya_test_pack.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
    lib.nya_test_pack.restype=C.c_int
    rng=random.Random(713)
    raw=rng.randbytes(144*256)
    source=C.create_string_buffer(raw)
    size=lib.nya_test_pack_size(len(raw))
    assert size==160*256 and lib.nya_test_pack_size(143)==0
    destination=C.create_string_buffer(b'\xcd'*(size+32),size+32)
    pointer=C.byref(destination,16)
    assert lib.nya_test_pack(pointer,size-1,source,len(raw))==-1
    assert destination.raw==b'\xcd'*(size+32)
    assert lib.nya_test_pack(pointer,size,source,len(raw))==0
    assert destination.raw[:16]==destination.raw[-16:]==b'\xcd'*16
    packed=destination.raw[16:-16]
    for block in range(256):
        original=raw[block*144:(block+1)*144]
        converted=packed[block*160:(block+1)*160]
        assert converted[:4]==original[:4] and converted[32:]==original[16:]
        assert converted[20:32]==bytes(12)
        for g in range(8):
            sc=(original[4+g]&63) if g<4 else (original[8+g]&15)|((original[g]>>6)<<4)
            mn=(original[8+g]&63) if g<4 else (original[8+g]>>4)|((original[4+g]>>6)<<4)
            assert converted[4+g]==sc and converted[12+g]==mn
    hashes[str(dll.relative_to(root))]=hashlib.sha256(dll.read_bytes()).hexdigest()
for p in inputs: assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(root))]
(out/'pack-build.json').write_text(json.dumps(dict(passed=True,gpu_execution=False,
    accepted_benchmark=False,blocks_per_standard=256,commands=commands,sha256=hashes),indent=2))
print('C17/C23 production C packer: 256 guarded metadata blocks each passed')
