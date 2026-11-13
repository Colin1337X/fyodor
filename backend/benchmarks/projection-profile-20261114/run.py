"""Reprofile the integrated decode pipeline; no graph-disabled throughput claims."""
import difflib,hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;build=root/'build-cuda'
if (out/'profile.json').exists():raise SystemExit('Refusing to overwrite evidence')
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy()
env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
production=(root/'backend/core/llm_cpu.c').read_text()
needle='if (request->soft_token_spans != 0) {'
assert production.count(needle)==1
source=production.replace(needle,'if (1) { /* Profile only: matched batched prompt ingestion. */',1)
path=build/'projection-profile-20261114-core.c';obj=path.with_suffix('.o');exe=build/'projection-profile-20261114.exe'
path.write_text(source)
flags=(build/'CMakeFiles/nya-engine.dir/flags.make').read_text()
defines=re.search(r'^C_DEFINES = (.*)$',flags,re.M)[1].split()
cflags=re.search(r'^C_FLAGS = (.*)$',flags,re.M)[1].split()
commands=[[str(compiler),*defines,*cflags,'-I',str(root/'backend/core'),'@'+str(build/'CMakeFiles/nya-engine.dir/includes_C.rsp'),'-c',str(path),'-o',str(obj)],
 [str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(out/'model.c'),str(obj),str(build/'libnya-engine.a'),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]]
for command in commands:subprocess.run(command,env=env,check=True)
settings=json.loads((out.parent/'mtp-early-reject-20261113/production-short.json').read_text())['environment']
settings.update(NYA_CUDA_PROFILE='1',NYA_SPECULATIVE_EARLY_REJECT='1',NYA_CUDA_WIDE_BATCH='1',NYA_CUDA_SMALL_BATCH_REUSE='1',NYA_CUDA_MTP_WIDE_VALUE='1',NYA_CUDA_BATCH_LOGITS='1',NYA_CUDA_ASSISTED_FFN='1')
meta={'accepted_llama_score':False,'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'commands':commands,'environment':settings,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'library_sha256':hashlib.sha256((build/'libnya-engine.a').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'runs':[]}
(out/'candidate.patch').write_text(''.join(difflib.unified_diff(production.splitlines(True),source.splitlines(True),fromfile='production/llm_cpu.c',tofile='profile/llm_cpu.c')))
for context,threshold in [('short','25'),('long','8205')]:
 env.update(settings);env['NYA_CUDA_PROFILE_FROM']=threshold
 with (out/(context+'.stdout')).open('w') as stdout,(out/(context+'.stderr')).open('w') as stderr:
  result=subprocess.run([str(exe),context],env=env,stdout=stdout,stderr=stderr)
 meta['runs'].append({'context':context,'threshold':threshold,'exit_code':result.returncode})
 (out/'profile.json').write_text(json.dumps(meta,indent=2)+'\n')
 if result.returncode:raise SystemExit(result.returncode)
 rows=[json.loads(line) for line in (out/(context+'.stdout')).read_text().splitlines() if line.startswith('{"round":')]
 assert len(rows)==4 and all(row['parity'] and row['generated']==32 for row in rows)
 print(context,'parity passed',flush=True)
subprocess.run([sys.executable,str(out/'summarize.py')],check=True)
