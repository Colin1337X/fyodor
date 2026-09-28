"""Inspect JIT resource counts after benchmarks finish; no performance timing."""
import ctypes as c
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
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
import re
options_list=['--gpu-architecture=compute_120','--std=c++11','--fmad=true','-DNYA_CUDA_FAST_MATH=1']
options=(c.c_char_p*len(options_list))(*(s.encode() for s in options_list))
bundles={'before':('matvec','gemm','resident','training'),'inference':('matvec','gemm','resident'),'training':('matvec','gemm','training')}
source_names={'before':'fyodor_matvec.cu','inference':'fyodor_inference.cu','training':'fyodor_training.cu'}
result={'options':options_list,'bundles':{}}
bodies={}
try:
    for variant,files in bundles.items():
        source=b'\n'.join((root/'CUDA'/f'{name}.cu').read_bytes() for name in files)
        program=ptr(); module=ptr()
        check(create(c.byref(program),source,source_names[variant].encode(),0,None,None))
        try:
            check(compile_program(program,len(options),options))
            size=c.c_size_t(); check(get_size(program,c.byref(size)))
            ptx=c.create_string_buffer(size.value); check(get_ptx(program,ptx))
            check(load(c.byref(module),ptx))
            record={'source_sha256':hashlib.sha256(source).hexdigest(),'ptx_sha256':hashlib.sha256(ptx.raw).hexdigest(),'entries':{}}
            text=ptx.value.decode(); bodies[variant]={}
            for match in re.finditer(r'\.visible \.entry (\w+)\(',text):
                name=match[1]; opening=text.index('{',match.end()); depth=1; end=opening+1
                while depth:
                    depth+=(text[end]=='{')-(text[end]=='}'); end+=1
                body=text[match.start():end]
                bodies[variant][name]=body
                function=ptr(); check(get_function(c.byref(function),module,name.encode()))
                entry={'ptx_entry_sha256':hashlib.sha256(body.encode()).hexdigest()}
                # NVRTC's module-global function ordinal prefixes local labels;
                # retain each local block number and all instructions verbatim.
                normalized=re.sub(r'\$L__BB\d+_', '$L__BB_', body)
                entry['normalized_ptx_sha256']=hashlib.sha256(normalized.encode()).hexdigest()
                for label,key in [('registers',4),('local_bytes',3),('shared_bytes',1)]:
                    value=c.c_int(); check(attribute(c.byref(value),key,function)); entry[label]=value.value
                record['entries'][name]=entry
            result['bundles'][variant]=record
        finally:
            if module: check(unload(module))
            check(destroy(c.byref(program)))
finally:
    previous=ptr(); check(bind(driver,'cuCtxPopCurrent_v2',c.POINTER(ptr))(c.byref(previous)))
    check(bind(driver,'cuDevicePrimaryCtxRelease_v2',c.c_int)(0))
before=result['bundles']['before']['entries']
result['differences']={variant:[name for name,entry in result['bundles'][variant]['entries'].items() if entry!=before[name]] for variant in ['inference','training']}
result['normalized_differences']={variant:[name for name,entry in result['bundles'][variant]['entries'].items() if any(entry[key]!=before[name][key] for key in ['normalized_ptx_sha256','registers','local_bytes','shared_bytes'])] for variant in ['inference','training']}
import difflib
diff=[]
for variant,names in result['differences'].items():
    for name in names:
        diff.extend(difflib.unified_diff(bodies['before'][name].splitlines(True),bodies[variant][name].splitlines(True),fromfile='before/'+name,tofile=variant+'/'+name))
(out/'kernel-code-diff.log').write_text(''.join(diff))
(out/'kernel-code.json').write_text(json.dumps(result,indent=2)+'\n')
print({variant:len(r['entries']) for variant,r in result['bundles'].items()},result['differences'])
assert not any(result['normalized_differences'].values()),result['normalized_differences']
