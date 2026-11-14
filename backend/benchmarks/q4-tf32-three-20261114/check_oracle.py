"""Validate the C double oracle against Python fsum, without any GPU work."""
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'oracle-control.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite oracle validation')
work = root / 'build-cuda/q4-tf32-three-oracle-20261114'
work.mkdir()
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ['PATH'])
inputs = {Path(__file__).resolve(), out / 'oracle.c', compiler}
bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
        0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x27ae, 0xa75a]
cases = [(256, 3, None), (512, 2, None)]
cases += [(256, 1, (d, m)) for d in bits for m in bits]
result = dict(passed=False, gpu_execution=False, accepted_benchmark=False, standards=[])
for standard in ('c17', 'c23'):
    binary = work / f'oracle-{standard}.dll'
    command = [str(compiler), '-std=' + standard, '-O3', '-Wall', '-Wextra', '-Werror',
               '-pedantic', '-shared', str(out / 'oracle.c'), '-o', str(binary), '-lm']
    subprocess.run(command, env=env, check=True)
    inputs.add(binary)
    library = C.CDLL(str(binary))
    oracle = library.q4_oracle
    oracle.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p, C.c_void_p]
    oracle.restype = C.c_int
    rng = random.Random(713)
    for columns, rows, edge in cases:
        stride = columns // 256 * 144
        data = bytearray(rng.randbytes(rows * stride))
        for offset in range(0, len(data), 144):
            data[offset:offset + 4] = struct.pack('<HH', *edge) if edge else struct.pack('<ee', .03, -.004)
        weights = C.create_string_buffer(bytes(data))
        x = (C.c_float * (4 * columns))(*(rng.uniform(-1, 1) for _ in range(4 * columns)))
        expected, bounds = (C.c_double * (4 * rows))(), (C.c_double * (4 * rows))()
        if oracle(weights, x, columns, rows, expected, bounds):
            raise ValueError('Oracle rejected a finite fixture')
        for token in range(4):
            for row in range(rows):
                products = []
                for column in range(columns):
                    start = row * stride + column // 256 * 144
                    b = data[start:start + 144]
                    d, m = struct.unpack('<ee', b[:4])
                    group = column % 256 // 32
                    sc = b[4 + group] & 63 if group < 4 else (b[8 + group] & 15) | ((b[group] >> 6) << 4)
                    mn = b[8 + group] & 63 if group < 4 else (b[8 + group] >> 4) | ((b[4 + group] >> 6) << 4)
                    q = (b[16 + group // 2 * 32 + column % 32] >> (group % 2 * 4)) & 15
                    products.append((d * sc * q - m * mn) * x[token * columns + column])
                total = math.fsum(products)
                absolute = math.fsum(abs(p) for p in products)
                index = token * rows + row
                if abs(expected[index] - total) > 1e-12 * (1 + absolute):
                    raise ValueError((standard, columns, rows, edge, 'sum'))
                if abs(bounds[index] - (2e-6 * absolute + 1e-6)) > 1e-12 * (1 + bounds[index]):
                    raise ValueError((standard, columns, rows, edge, 'bound'))
    for half in (0x7c00, 0xfc00, 0x7e00):
        weights = C.create_string_buffer(struct.pack('<HH', half, 0) + bytes(140))
        if oracle(weights, x, 256, 1, expected, bounds) != -2:
            raise ValueError('Nonfinite coefficient accepted')
    if oracle(None, None, 0, 1, None, None) != -1:
        raise ValueError('Invalid arguments accepted')
    result['standards'].append(dict(standard=standard, command=command, finite_cases=len(cases),
                                    nonfinite_rejections=3, invalid_argument_rejections=1))
result['passed'] = True
result['sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
destination.write_text(json.dumps(result, indent=2))
print('C17/C23 double oracle validated; 198 finite fixtures per standard; no GPU work')
