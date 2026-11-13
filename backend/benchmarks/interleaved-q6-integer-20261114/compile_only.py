"""Compile batch-two/four Q6 candidates without creating a CUDA context."""
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
paths = [root / name for name in ('CUDA/matvec.cu', 'optcpp/matvec/interleaved_two.cu',
         'optcpp/matvec/interleaved_reuse.cu', 'optcpp/matvec/interleaved_q6_integer.cu')]
parts = []
for path, old_name, new_name, batch in ((paths[1], 'nya_interleaved_two', 'nya_interleaved_q6_integer_two', 2),
                                      (paths[2], 'nya_interleaved_reuse', 'nya_interleaved_q6_integer_four', 4)):
    text = path.read_text().replace(old_name, new_name).replace('nya_interleave_two', 'nya_q6_integer_interleave_two').replace('nya_interleave_four', 'nya_q6_integer_interleave_four')
    assert text.count('weight = d * (float)sc * (float)q;') == 1
    text = text.replace('weight = d * (float)sc * (float)q;',
        'float product = __int_as_float(__float_as_int((float)(sc*q)) | ((sc ^ q) & (-2147483647-1)));\n                weight = d * product;')
    parts.append(text.replace(f'NYA_SMALL_REUSE(12, {batch})', ''))
assert paths[3].read_text().endswith('\n'.join(parts)), 'Unexpected candidate transformation'
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
    program = P(); check(create(C.byref(program), source, b'interleaved_q6_integer.cu', 0, None, None))
    try:
        status = compile_(program, len(options), (C.c_char_p*len(options))(*options))
        n = Z(); check(log_size(program, C.byref(n)))
        log = C.create_string_buffer(n.value); check(log_get(program, log))
        (out / 'compile.log').write_bytes(log.value)
        check(status)
        check(ptx_size(program, C.byref(n)))
        ptx = C.create_string_buffer(n.value); check(ptx_get(program, ptx))
        for suffix in ('two_14_2', 'four_14_4'):
            assert ('.entry nya_interleaved_q6_integer_' + suffix + '(').encode() in ptx.value
        for suffix in ('two_12_2', 'four_12_4'):
            assert ('.entry nya_interleaved_q6_integer_' + suffix + '(').encode() not in ptx.value
    finally:
        check(destroy(C.byref(program)))
metadata = dict(accepted_benchmark=False, gpu_execution=False, compile_passed=True,
                options=[option.decode() for option in options], ptx_sha256=hashlib.sha256(ptx.value).hexdigest(),
                sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths + [Path(__file__), root / 'optcpp/matvec/integer_scale_q6.cu', root / 'backend/benchmarks/decode-q6-integer-20261028/coefficient-check.json']},
                pending=['raw exact parity and guards', 'memory sanitization', 'real-model parity',
                         'paired decode performance', 'runtime integration only after evidence'])
destination.write_text(json.dumps(metadata, indent=2))
print('Q6 batch-two/four integer products compiled; no CUDA driver/context/kernel used')
