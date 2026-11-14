"""Offline SM120 compilation only: never load the CUDA driver or create a context."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
proof = out / 'result.json'
if proof.exists():
    raise SystemExit('Refusing to overwrite evidence')
sources = [root / name for name in (
    'CUDA/matvec.cu', 'CUDA/gemm.cu', 'CUDA/resident.cu',
    'optcpp/attention/wide_value16.cu',
    'optcpp/attention/wide_batch_split16.cu',
    'optcpp/attention/wide_batch_cached_query.cu')]
nvdir = root / '.tools/cuda-sdk/cuda_nvrtc-windows-x86_64-13.1.115-archive/bin/x64'
assembler = root / '.tools/cuda-sdk/cuda_nvcc-windows-x86_64-13.1.115-archive/bin/ptxas.exe'
inputs = sources + [Path(__file__), assembler, nvdir / 'nvrtc64_130_0.dll', nvdir / 'nvrtc-builtins64_131.dll']
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def key(path):
    return str(path.relative_to(root))
before = {key(p): digest(p) for p in inputs}
loader = os.add_dll_directory(str(nvdir))
builtins = C.CDLL(str(nvdir / 'nvrtc-builtins64_131.dll'))
nv = C.CDLL(str(nvdir / 'nvrtc64_130_0.dll'))
P, I, Z = C.c_void_p, C.c_int, C.c_size_t
def bind(name, args):
    f = getattr(nv, name)
    f.argtypes, f.restype = args, I
    return f
def check(status):
    if status:
        raise RuntimeError('NVRTC status ' + str(status))
create = bind('nvrtcCreateProgram', [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P])
compile_program = bind('nvrtcCompileProgram', [P, I, C.POINTER(C.c_char_p)])
destroy = bind('nvrtcDestroyProgram', [C.POINTER(P)])
def output(program, kind):
    size = bind('nvrtcGet' + kind + 'Size', [P, C.POINTER(Z)])
    get = bind('nvrtcGet' + kind, [P, P])
    count = Z()
    check(size(program, C.byref(count)))
    data = C.create_string_buffer(count.value)
    check(get(program, data))
    return data.value
options = [b'--gpu-architecture=compute_120', b'--std=c++11', b'--fmad=true', b'-DNYA_CUDA_FAST_MATH=1']
program = P()
check(create(C.byref(program), b'\n'.join(p.read_bytes() for p in sources), b'attention_resources.cu', 0, None, None))
try:
    status = compile_program(program, len(options), (C.c_char_p * len(options))(*options))
    (out / 'compile.log').write_bytes(output(program, 'ProgramLog'))
    check(status)
    (out / 'kernels.ptx').write_bytes(output(program, 'PTX'))
finally:
    check(destroy(C.byref(program)))
command = [str(assembler), '-arch=sm_120', '-v', str(out / 'kernels.ptx'), '-o', str(out / 'kernels.cubin')]
completed = subprocess.run(command, capture_output=True)
(out / 'assembly.stdout').write_bytes(completed.stdout)
(out / 'assembly.stderr').write_bytes(completed.stderr)
if completed.returncode:
    raise SystemExit('Assembly failed')
trace = completed.stdout.decode() + completed.stderr.decode()
entries = {}
for name, body in re.findall(r"Compiling entry function '([^']+)'(.*?)(?=ptxas info\s*: Compiling entry function|\Z)", trace, re.S):
    if name.startswith(('nya_attention_cached_query', 'nya_attention_batch_split16')):
        registers = re.search(r'Used (\d+) registers', body)
        spills = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', body)
        if not registers or not spills:
            raise RuntimeError('Missing resource inventory: ' + name)
        entries[name] = dict(registers=int(registers[1]), stack_bytes=int(spills[1]), spill_store_bytes=int(spills[2]), spill_load_bytes=int(spills[3]))
required = {'nya_attention_cached_query_split', 'nya_attention_cached_query_ring_split', 'nya_attention_batch_split16_split', 'nya_attention_batch_split16_ring_split'}
if not required.issubset(entries):
    raise RuntimeError('Missing split kernels')
changed = [name for name, expected in before.items() if digest(root / name) != expected]
artifacts = {key(p): digest(p) for p in out.iterdir() if p.is_file()}
proof.write_text(json.dumps(dict(passed=not changed, gpu_execution=False, accepted_benchmark=False, options=[v.decode() for v in options], command=command, entries=entries, changed_inputs=changed, sha256={**before, **artifacts}, limitation='Offline compiler allocation, not driver-JIT resources, achieved occupancy, runtime safety, or throughput.'), indent=2))
print(json.dumps(entries, indent=2))
if changed:
    raise SystemExit('Inputs changed')
