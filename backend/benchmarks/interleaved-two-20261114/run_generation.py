"""Measure batch-two Q6 interleaving in full generation, without promoting scores."""
import difflib, hashlib, json, os, re, subprocess, sys
from pathlib import Path

root = Path.cwd()
out = Path(__file__).resolve().parent
build = root / 'build-cuda'
validation = json.loads((out/'validation.json').read_text())
assert len(validation['cases']) == 8 and all(r['exit_code'] == 0 for r in validation['cases'])
memory=json.loads((out/'integrated-memcheck.json').read_text())
assert memory['exit_code']==0 and 'ERROR SUMMARY: 0 errors' in (out/'integrated-memcheck.log').read_text()
assert '-- Verified without non-c:' in (out/'physical-removal.log').read_text()
if (out / 'build.json').exists():
    raise SystemExit('Refusing to overwrite evidence')
compiler = root / '.tools/w64devkit/bin/gcc.exe'
env = os.environ.copy()
env['PATH'] = str(compiler.parent) + os.pathsep + env['PATH']
production = (root / 'backend/core/llm_cpu.c').read_text()
needle = 'if (request->soft_token_spans != 0) {'
assert production.count(needle) == 1
source = production.replace(needle, 'if (1) { /* Diagnostic matched B512 prompt ingestion. */', 1)
core = build / 'interleaved-two-20261114-core.c'
obj = core.with_suffix('.o')
exe = build / 'interleaved-two-20261114.exe'
core.write_text(source)
model = out / 'generation.c'
model.write_bytes((out.parent / 'mtp-cost-20261113/model.c').read_bytes())
flags = (build / 'CMakeFiles/nya-engine.dir/flags.make').read_text()
defines = re.search(r'^C_DEFINES = (.*)$', flags, re.M)[1].split()
cflags = re.search(r'^C_FLAGS = (.*)$', flags, re.M)[1].split()
commands = [
    [str(compiler), *defines, *cflags, '-I', str(root/'backend/core'),
     '@'+str(build/'CMakeFiles/nya-engine.dir/includes_C.rsp'), '-c', str(core), '-o', str(obj)],
    [str(compiler), '-std=c17', '-O3', '-Wall', '-Wextra', '-Werror',
     '-I', 'backend/include', '-I', 'backend/core', str(model), str(obj),
     str(build/'libnya-engine.a'), '-lbcrypt', 'C:/Windows/System32/vulkan-1.dll', '-o', str(exe)]
]
for command in commands:
    subprocess.run(command, env=env, check=True)
(out/'candidate.patch').write_text(''.join(difflib.unified_diff(
    production.splitlines(True), source.splitlines(True),
    fromfile='production/llm_cpu.c', tofile='diagnostic/llm_cpu.c')))
meta = {'accepted_llama_score': False,
        'revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'commands': commands,
        'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (core, model, obj, exe, build/'libnya-engine.a')}}
(out/'build.json').write_text(json.dumps(meta, indent=2)+'\n')
env.update(NYA_CUDA_PROFILE='0', NYA_CUDA_BATCH_LOGITS='1', NYA_CUDA_ASSISTED_FFN='1', NYA_CUDA_SMALL_BATCH_REUSE='1',
           NYA_CUDA_INTERLEAVED_REUSE='1', NYA_CUDA_WIDE_BATCH='1', NYA_SPECULATIVE_EARLY_REJECT='1', NYA_CUDA_MTP_WIDE_VALUE='1')
for label, enabled in [('runtime-off','0'), ('runtime-on','1')]:
    env['NYA_CUDA_INTERLEAVED_TWO'] = enabled
    subprocess.run([sys.executable, 'backend/benchmarks/mtp-cost-20261113/run.py',
                    '../interleaved-two-20261114/'+label,
                    str(exe.relative_to(root)), 'short', 'long'], env=env, check=True)
subprocess.run([sys.executable, str(out/'summarize_generation.py')], check=True)
