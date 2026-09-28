"""Inspect JIT resource counts after benchmarks finish; no performance timing."""
import ctypes as c
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
baseline = 'b15e04d68455f8d1e6cebc140fadb04a0508bf3c'
dll_dir = root / '.tools/cuda-sdk/cuda_nvrtc-windows-x86_64-13.1.115-archive/bin/x64'
builtins = c.CDLL(str(dll_dir / 'nvrtc-builtins64_131.dll'))
nvrtc = c.CDLL(str(dll_dir / 'nvrtc64_130_0.dll'))
driver = c.WinDLL('nvcuda.dll')

def bind(lib, name, *args):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = c.c_int
    return fn

def check(status):
    if status:
        raise RuntimeError(f'CUDA/NVRTC status {status}')

ptr = c.c_void_p
create = bind(nvrtc, 'nvrtcCreateProgram', c.POINTER(ptr), c.c_char_p, c.c_char_p, c.c_int, ptr, ptr)
compile_program = bind(nvrtc, 'nvrtcCompileProgram', ptr, c.c_int, c.POINTER(c.c_char_p))
destroy = bind(nvrtc, 'nvrtcDestroyProgram', c.POINTER(ptr))
get_size = bind(nvrtc, 'nvrtcGetPTXSize', ptr, c.POINTER(c.c_size_t))
get_ptx = bind(nvrtc, 'nvrtcGetPTX', ptr, ptr)
load = bind(driver, 'cuModuleLoadData', c.POINTER(ptr), ptr)
unload = bind(driver, 'cuModuleUnload', ptr)
get_function = bind(driver, 'cuModuleGetFunction', c.POINTER(ptr), ptr, c.c_char_p)
attribute = bind(driver, 'cuFuncGetAttribute', c.POINTER(c.c_int), c.c_int, ptr)
check(bind(driver, 'cuInit', c.c_uint)(0))
context = ptr()
check(bind(driver, 'cuDevicePrimaryCtxRetain', c.POINTER(ptr), c.c_int)(c.byref(context), 0))
check(bind(driver, 'cuCtxPushCurrent_v2', ptr)(context))
result = {'baseline': baseline, 'options': ['--gpu-architecture=compute_120', '--std=c++11', '--fmad=true', '-DNYA_CUDA_FAST_MATH=1'], 'kernels': {}}
try:
    for variant in ('before', 'two-chains'):
        sources = []
        for path in ('CUDA/matvec.cu', 'CUDA/gemm.cu', 'CUDA/resident.cu'):
            sources.append(subprocess.check_output(['git', 'show', f'{baseline}:{path}'], cwd=root) if variant == 'before' else (root / path).read_bytes())
        source = b'\n'.join(sources)
        program = ptr()
        module = ptr()
        check(create(c.byref(program), source, b'fyodor_matvec.cu', 0, None, None))
        try:
            options = (c.c_char_p * len(result['options']))(*(s.encode() for s in result['options']))
            check(compile_program(program, len(options), options))
            size = c.c_size_t()
            check(get_size(program, c.byref(size)))
            ptx = c.create_string_buffer(size.value)
            check(get_ptx(program, ptx))
            check(load(c.byref(module), ptx))
            record = {'source_sha256': hashlib.sha256(source).hexdigest(), 'ptx_sha256': hashlib.sha256(ptx.raw).hexdigest()}
            for name in ('nya_dot_12', 'nya_dot_14'):
                function = ptr()
                check(get_function(c.byref(function), module, name.encode()))
                record[name] = {}
                for label, key in (('registers_per_thread', 4), ('local_bytes', 3), ('shared_bytes', 1), ('max_threads', 0)):
                    value = c.c_int()
                    check(attribute(c.byref(value), key, function))
                    record[name][label] = value.value
            result['kernels'][variant] = record
        finally:
            if module:
                check(unload(module))
            check(destroy(c.byref(program)))
finally:
    previous = ptr()
    check(bind(driver, 'cuCtxPopCurrent_v2', c.POINTER(ptr))(c.byref(previous)))
    check(bind(driver, 'cuDevicePrimaryCtxRelease_v2', c.c_int)(0))
(out / 'registers.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
