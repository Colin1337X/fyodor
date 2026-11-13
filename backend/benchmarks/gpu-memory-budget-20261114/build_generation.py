"""Build, but do not run, the in-process memory-budget diagnostic."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
build = root / 'build-cuda'
destination = out / 'generation-build.json'
work = build / 'budget-generation-20261114'
model, patch = out / 'generation.c', out / 'generation.patch'
if destination.exists() or work.exists() or model.exists() or patch.exists():
    raise SystemExit('Refusing to overwrite diagnostic build evidence')
work.mkdir()
library = build / 'libnya-engine.a'
base_hash = hashlib.sha256(library.read_bytes()).hexdigest()
production = (root / 'backend/core/llm_cpu.c').read_text()
needle = 'if (request->soft_token_spans != 0) {'
assert production.count(needle) == 1
source = production.replace(needle, 'if (1) { /* Diagnostic matched B512 prompt ingestion. */')
source = 'extern int nya_diag_memory(const char *phase);\n' + source
needle = 'if (profile && position + 1 == prompt_count) decode_start = nya_runtime_monotonic_seconds();'
assert source.count(needle)==1
source = source.replace(needle, 'if (profile && position + 1 == prompt_count) { (void)nya_diag_memory("ordinary-before"); decode_start = nya_runtime_monotonic_seconds(); }')
needle = 'if (profile) decode_start = nya_runtime_monotonic_seconds();'
assert source.count(needle)==1
source = source.replace(needle, 'if (profile) { (void)nya_diag_memory("mtp-before"); decode_start = nya_runtime_monotonic_seconds(); }')
needle = 'if (profile) nya_llm_decode_profile("ordinary", decode_start, prompt_count, generated);'
assert source.count(needle)==1
source = source.replace(needle, needle + '\n    if(profile) (void)nya_diag_memory("ordinary-after");')
needle = 'nya_llm_decode_profile(draft->is_assistant ? "mtp" : "speculative", decode_start, prompt_count, out->generated_tokens);'
assert source.count(needle)==1
source = source.replace(needle, needle + '\n        (void)nya_diag_memory("mtp-after");')
core = work / 'core.c'
core.write_text(source)
patch.write_text(''.join(difflib.unified_diff(production.splitlines(True), source.splitlines(True),
                                           fromfile='production/llm_cpu.c', tofile='diagnostic/llm_cpu.c')))
model.write_bytes((out.parent / 'mtp-cost-20261113/model.c').read_bytes())
flags = (build / 'CMakeFiles/nya-engine.dir/flags.make').read_text()
defines = re.search(r'^C_DEFINES = (.*)$', flags, re.M)[1].split()
cflags = re.search(r'^C_FLAGS = (.*)$', flags, re.M)[1].split()
assert '-DNYA_ALIGNED_PREFIX=1' in defines and '-DNYA_INTERLEAVED_INTEGER=1' in defines
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = os.environ.copy()
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
commands = []
for standard in ('c17', 'c23'):
    command = [str(compiler), *defines,
               *[f'-std={standard}' if flag.startswith('-std=') else flag for flag in cflags],
               '-I', str(root / 'backend/core'), '@' + str(build / 'CMakeFiles/nya-engine.dir/includes_C.rsp'),
               '-c', str(core), '-o', str(work / f'core-{standard}.o')]
    commands.append(command)
exe = work / 'generation.exe'
commands.append([str(compiler), '-std=c17', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-Wconversion',
                 '-Wshadow', '-Werror', '-I', 'backend/include', '-I', 'backend/core', str(model),
                 str(work / 'core-c17.o'), str(out / 'budget.c'), str(library), '-ldxgi', '-ldxguid', '-lbcrypt', 'C:/Windows/System32/vulkan-1.dll', '-o', str(exe)])
for command in commands:
    subprocess.run(command, env=env, check=True)
assert b'Generation interval:' in exe.read_bytes()
assert hashlib.sha256(library.read_bytes()).hexdigest() == base_hash
inputs = [core, model, patch, library, exe, work / 'core-c17.o', work / 'core-c23.o',
          root / 'backend/core/llm_cpu.c', root / 'CUDA/resident.inc', root / 'CUDA/cuda.c', root / 'CUDA/CMakeLists.txt',
          root / 'optcpp/matvec/interleaved_integer.cu', out / 'budget.c', Path(__file__)]
metadata = dict(accepted_benchmark=False, executed_generation=False,
                revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                commands=commands, sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                note='Isolated B512 prompt override plus memory budget queries before/after decode; production library unchanged. Budget calls are outside decode endpoints; no timing acceptance. C17/C23 strict compilation. Real-model sanitizer and monitored execution remain separate gates.')
with destination.open('x') as result:
    json.dump(metadata, result, indent=2)
print('Built', exe, 'without launching GPU work')
