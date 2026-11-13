"""Real Gemma lifecycle checks for opt-in Q4 integer batch-four coefficients."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
memcheck = '--memcheck' in sys.argv
if memcheck:
    sys.argv.remove('--memcheck')
if len(sys.argv) not in (4, 5):
    raise SystemExit('Usage: run_verify.py PREFIX RING COUNT [RUN_LABEL] [--memcheck]')
prefix, ring, count = map(int, sys.argv[1:4])
if prefix not in (31, 253, 255, 256, 257, 513, 2048, 8192) or ring not in (0, 1) or not 1 <= count <= 17:
    raise SystemExit('Unsupported case')
tag = f'verify-prefix{prefix}-ring{ring}-count{count}'
if len(sys.argv) == 5:
    if not sys.argv[4].isascii() or not sys.argv[4].isalnum():
        raise SystemExit('RUN_LABEL must be ASCII alphanumeric')
    tag += '-' + sys.argv[4]
if memcheck:
    tag += '-memcheck'
paths = {ext: out / f'{tag}.{ext}' for ext in ('json', 'log', 'stderr.log', 'c')}
exe = root / f'build-cuda/integer-runtime-{tag}.exe'
if any(path.exists() for path in (*paths.values(), exe)):
    raise SystemExit('Refusing to overwrite evidence')
flags = (root / 'build-cuda/CMakeFiles/nya-engine.dir/flags.make').read_text()
if '-DNYA_ALIGNED_PREFIX=1' not in flags or '-DNYA_INTERLEAVED_INTEGER=1' not in flags:
    raise SystemExit('Build must enable aligned prefix and interleaved integer')
source = (root / 'backend/tests/test_verify.c').read_text()
assert source.count('size_t count=17,') == 1
source=source.replace('size_t count=17,', f'size_t count={count},')
marker='    s=nya_llm_session_create(m,total,error,sizeof(error));CHECK(s);'
assert source.count(marker)==2
source=source.replace(marker, '    CHECK(!interleaved_setting("NYA_CUDA_INTERLEAVED_INTEGER","0"));\n'+marker, 1)
position=source.rindex(marker)
source=source[:position]+'    CHECK(!interleaved_setting("NYA_CUDA_INTERLEAVED_INTEGER","1"));\n'+source[position:]
paths['c'].write_text(source)
library = root / 'build-cuda/libnya-engine.a'
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
baseline = json.loads((out.parent / 'fdpp-batch-20261114/aligned-prefix-ctx8192.json').read_text())
settings = {k: v for k, v in baseline['environment'].items() if not k.startswith('NYA_TEST_')}
settings.update(NYA_CUDA_RING_KV=str(ring), NYA_TEST_VERIFY_RESIDENT='1',
                NYA_CUDA_ALIGNED_PREFIX='1', NYA_MTP_TRACE='1',
                NYA_CUDA_INTERLEAVED_REUSE='1', NYA_CUDA_INTERLEAVED_TWO='1')
env.update(settings)
build = [str(compiler), '-std=c17', '-O3', '-Wall', '-Wextra', '-Wpedantic',
         '-Wconversion', '-Wshadow', '-Werror', '-I', 'backend/include', '-I', 'backend/core',
         str(paths['c']), str(library), '-lbcrypt', 'C:/Windows/System32/vulkan-1.dll', '-o', str(exe)]
subprocess.run(build, env=env, check=True)
model = Path('E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf')
command = [str(exe), str(model), str(prefix)]
if memcheck:
    sanitizer = root / '.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    command = [str(sanitizer), '--tool', 'memcheck', '--error-exitcode', '99', *command]
inputs = [paths['c'], library, exe, Path(__file__), root / 'CUDA/resident.inc',
          root / 'CUDA/CMakeLists.txt', root / 'optcpp/attention/flash_decoding_pp_batch.cu', root / 'CUDA/cuda.c',
          root / 'optcpp/matvec/interleaved_integer.cu']
hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
with paths['log'].open('x') as stdout, paths['stderr.log'].open('x') as stderr:
    result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
trace = paths['stderr.log'].read_text()
sanitizer_clear = 'ERROR SUMMARY: 0 errors' in (trace + paths['log'].read_text()) if memcheck else None
# test_verify admits capacity=prefix+count+1. Small plans intentionally retain
# normal execution and must never report an aligned plan.
expected = prefix + count + 1 > 256
admitted = 'CUDA aligned prefix:' in trace
integer_selected = 'CUDA Q4 integer batch-four: enabled' in trace
integer_calls = [int(value) for value in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)', trace)]
metadata = dict(accepted_benchmark=False, prefix=prefix, ring=ring, count=count,
                build=build, command=command, environment=settings, sha256=hashes,
                exit_code=result.returncode, expected_aligned=expected, aligned_admitted=admitted,
                integer_selected=integer_selected, integer_calls=integer_calls, memcheck=memcheck, sanitizer_clear=sanitizer_clear,
                model_stat=dict(bytes=model.stat().st_size, mtime_ns=model.stat().st_mtime_ns),
                note='Original lifecycle assertions and 0.0005 bound retained; only token count specialized.')
with paths['json'].open('x') as output:
    json.dump(metadata, output, indent=2)
if result.returncode or not integer_selected or not integer_calls or min(integer_calls)<=0 or admitted != expected or (memcheck and not sanitizer_clear):
    raise SystemExit(result.returncode or 'Admission or sanitizer validation failed')
print(tag, 'passed', flush=True)
