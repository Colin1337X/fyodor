"""Build benchmark-only variants without changing production prefill dispatch."""
import difflib,hashlib,json,re,subprocess,os
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;build=root/'build-cuda'
compiler=root/'.tools/w64devkit/bin/gcc.exe'
os.environ['PATH']=str(compiler.parent)+os.pathsep+os.environ['PATH']
flags=(build/'CMakeFiles/nya-engine.dir/flags.make').read_text()
defines=re.search(r'^C_DEFINES = (.*)$',flags,re.M)[1].split()
cflags=re.search(r'^C_FLAGS = (.*)$',flags,re.M)[1].split()
production=(root/'backend/core/llm_cpu.c').read_text()
needle='if (request->soft_token_spans != 0) {'
assert production.count(needle)==1
prepared=production.replace(needle,'if (1) { /* Standalone probe: use existing batched prefill for all prompts. */',1)
manifest={'production_source_sha256':hashlib.sha256(production.encode()).hexdigest(),'library_sha256':hashlib.sha256((build/'libnya-engine.a').read_bytes()).hexdigest(),'variants':[]}
for variant in ('baseline','candidate'):
    source=prepared
    if variant=='baseline':
        needle='if (proposed + 1 < maximum) {'
        assert source.count(needle)==1
        source=source.replace(needle,'if (1) { /* Restore the discarded final assistant forward for baseline. */',1)
    path=build/f'mtp-cost-{variant}-core.c';obj=path.with_suffix('.o');exe=build/f'mtp-cost-{variant}-b512.exe'
    path.write_text(source)
    patch=''.join(difflib.unified_diff(production.splitlines(True),source.splitlines(True),fromfile='production/llm_cpu.c',tofile=f'probe/{variant}/llm_cpu.c'))
    (out/f'{variant}-probe.patch').write_text(patch)
    commands=[[str(compiler),*defines,*cflags,'-I',str(root/'backend/core'),'@'+str(build/'CMakeFiles/nya-engine.dir/includes_C.rsp'),'-c',str(path),'-o',str(obj)],
        [str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(out/'model.c'),str(obj),str(build/'libnya-engine.a'),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]]
    for command in commands:subprocess.run(command,check=True)
    manifest['variants'].append({'variant':variant,'commands':commands,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()})
(out/'probe-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
