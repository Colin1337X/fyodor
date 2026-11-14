"""NVRTC compile without loading CUDA or creating any GPU context."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

root = Path.cwd()
out = Path(__file__).resolve().parent
destination = out / 'compile.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite compilation evidence')
paths = [root / 'CUDA/matvec.cu', root / 'optcpp/matvec/q4_tf32_three.cu', root / 'optcpp/matvec/q4_tf32_split.cu']
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
    log_size = bind('nvrtcGetProgramLogSize', [P, C.POINTER(Z)])
    log_get = bind('nvrtcGetProgramLog', [P, P])
    ptx_size = bind('nvrtcGetPTXSize', [P, C.POINTER(Z)])
    ptx_get = bind('nvrtcGetPTX', [P, P])
    destroy = bind('nvrtcDestroyProgram', [C.POINTER(P)])
    options = [b'--gpu-architecture=compute_120', b'--std=c++11', b'--fmad=true', b'-DNYA_CUDA_FAST_MATH=1']
    program = P(); check(create(C.byref(program), source, b'q4_tf32_three.cu', 0, None, None))
    try:
        status = compile_(program, len(options), (C.c_char_p*len(options))(*options))
        n = Z(); check(log_size(program, C.byref(n)))
        log = C.create_string_buffer(n.value); check(log_get(program, log))
        (out / 'compile.log').write_bytes(log.value)
        check(status)
        check(ptx_size(program, C.byref(n)))
        ptx = C.create_string_buffer(n.value); check(ptx_get(program, ptx))
        assert b'.entry nya_q4_tf32_split(' in ptx.value
    finally:
        check(destroy(C.byref(program)))
work=root/'build-cuda/q4-tf32-split-20261114'
work.mkdir()
ptx_path=work/'module.ptx';ptx_path.write_bytes(ptx.value)
paths += [ptx_path, out/'compile.log', directory/'nvrtc64_130_0.dll', directory/'nvrtc-builtins64_131.dll']
metadata = dict(accepted_benchmark=False, gpu_execution=False, compile_passed=True,
                options=[option.decode() for option in options], ptx_sha256=hashlib.sha256(ptx.value).hexdigest(),
                sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths + [Path(__file__)]},
                pending=['independent numerical bounds and guards', 'memory sanitization', 'real-model parity',
                         'paired decode performance', 'runtime integration only after evidence'])
destination.write_text(json.dumps(metadata, indent=2))
print('Q4 three-product TF32 candidate compiled; no CUDA driver/context/kernel used')
