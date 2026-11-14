"""CPU-only PTX instruction census; no CUDA driver or performance claim."""
import collections
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'ptx-inspection.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite PTX evidence')
prior = json.loads((out / 'compile.json').read_text())
for name, expected in prior['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
paths = [root / name for name in ('CUDA/matvec.cu', 'optcpp/matvec/integer_launch.cu',
                                  'optcpp/matvec/llama_q4_metadata.cu')]
source = '\n'.join(path.read_text() for path in paths).encode()
directory = root / '.tools/cuda-sdk/cuda_nvrtc-windows-x86_64-13.1.115-archive/bin/x64'
with os.add_dll_directory(str(directory)):
    builtins = C.CDLL(str(directory / 'nvrtc-builtins64_131.dll'))
    nv = C.CDLL(str(directory / 'nvrtc64_130_0.dll'))
    P, Z, I = C.c_void_p, C.c_size_t, C.c_int
    def bind(name, args):
        f = getattr(nv, name); f.argtypes = args; f.restype = I; return f
    def check(status):
        if status: raise RuntimeError(f'NVRTC status {status}')
    create = bind('nvrtcCreateProgram', [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P])
    compile_ = bind('nvrtcCompileProgram', [P, I, C.POINTER(C.c_char_p)])
    size = bind('nvrtcGetPTXSize', [P, C.POINTER(Z)])
    get = bind('nvrtcGetPTX', [P, P])
    destroy = bind('nvrtcDestroyProgram', [C.POINTER(P)])
    options = [option.encode() for option in prior['options']]
    program = P(); check(create(C.byref(program), source, b'interleaved_integer.cu', 0, None, None))
    try:
        check(compile_(program, len(options), (C.c_char_p * len(options))(*options)))
        n = Z(); check(size(program, C.byref(n)))
        ptx = C.create_string_buffer(n.value); check(get(program, ptx))
    finally:
        check(destroy(C.byref(program)))
actual_ptx_sha256 = hashlib.sha256(ptx.value).hexdigest()
text = ptx.value.decode()
entries = {}
for name in ('nya_integer_launch_8_3', 'nya_llama_q4_metadata_12_4'):
    start = text.index('.entry ' + name + '(')
    opening = text.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    body = text[opening:end]
    instructions = collections.Counter()
    for line in body.splitlines():
        match = re.match(r'^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.]*)(?:\s+[^;]*)?;\s*$', line)
        if match:
            instructions[match[1]] += 1
    entries[name] = dict(static_instruction_counts=dict(sorted(instructions.items())),
                         virtual_register_declarations=re.findall(r'\.reg\s+[^;]+;', body),
                         body_sha256=hashlib.sha256(body.encode()).hexdigest())
result = dict(gpu_execution=False, accepted_benchmark=False, ptx_sha256=actual_ptx_sha256,
              options=prior['options'], entries=entries,
              sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              note='Static PTX instruction sites, not dynamic counts, SASS, physical registers, occupancy, or speed. Both kernels compiled in one module with the same options.')
with destination.open('x') as output:
    json.dump(result, output, indent=2)
for name, entry in entries.items():
    print(name, entry['static_instruction_counts'])
