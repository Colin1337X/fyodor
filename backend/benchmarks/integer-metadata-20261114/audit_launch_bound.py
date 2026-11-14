"""Offline SM120 assembly census; never loads a CUDA driver or runs kernels."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'launch-bound-assembly.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite assembly evidence')
prior = json.loads((out / 'ptx-inspection.json').read_text())
for name, expected in prior['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
paths = [root / name for name in prior['sha256']]
source = '\n'.join(path.read_text() for path in paths).encode()
original = paths[-1].read_text()
assert original.count('__launch_bounds__(256, 3)') == 1
assert original.count('nya_integer_metadata_12_4') == 1
variant = original.replace('__launch_bounds__(256, 3)', '__launch_bounds__(256, 4)').replace('nya_integer_metadata_12_4', 'nya_integer_metadata_bound4')
source += b'\n' + variant.encode()

directory = root / '.tools/cuda-sdk/cuda_nvrtc-windows-x86_64-13.1.115-archive/bin/x64'
assembler = root / '.tools/cuda-sdk/cuda_nvcc-windows-x86_64-13.1.115-archive/bin/ptxas.exe'
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
ptx_hash = hashlib.sha256(ptx.value).hexdigest()
# The extra bound-four entry intentionally changes this module's PTX hash.
with tempfile.TemporaryDirectory(prefix='fyodor-offline-assembly-') as temp:
    source_path = Path(temp) / 'candidate.ptx'
    source_path.write_bytes(ptx.value)
    command = [str(assembler), '-arch=sm_120', '-O3', '-v', str(source_path),
               '-o', str(Path(temp) / 'candidate.cubin')]
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    log = run.stdout + run.stderr
    (out / 'launch-bound-assembly.log').write_text(log)
    assert run.returncode == 0, log
    cubin_hash = hashlib.sha256((Path(temp) / 'candidate.cubin').read_bytes()).hexdigest()
entries = {}
for name in ('nya_integer_launch_8_3', 'nya_integer_metadata_12_4', 'nya_integer_metadata_bound4'):
    match = re.search(r'Function properties for ' + name + r'\s+.*?(?=ptxas info\s+: Compiling|\Z)', log, re.S)
    assert match, name
    detail = match.group()
    stack = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', detail)
    registers = re.search(r'Used (\d+) registers', detail)
    assert stack and registers, detail
    entries[name] = dict(registers=int(registers[1]), stack_bytes=int(stack[1]),
                         spill_store_bytes=int(stack[2]), spill_load_bytes=int(stack[3]))
result = dict(gpu_execution=False, accepted_benchmark=False, assembly_passed=True,
              architecture='sm_120', experiment='Additional entry changes only launch bound 3 to 4 and symbol name', source_sha256=hashlib.sha256(source).hexdigest(), assembler_options=['-arch=sm_120', '-O3', '-v'],
              ptx_sha256=ptx_hash, cubin_sha256=cubin_hash, entries=entries,
              assembler_version=subprocess.check_output([str(assembler), '--version'], text=True),
              sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths + [Path(__file__), assembler, out / 'launch-bound-assembly.log', out / 'ptx-inspection.json']},
              note='Offline assembler allocation only; runtime driver JIT may differ. No measured occupancy or speed.')
with destination.open('x') as output:
    json.dump(result, output, indent=2)
print(json.dumps(entries, indent=2))
