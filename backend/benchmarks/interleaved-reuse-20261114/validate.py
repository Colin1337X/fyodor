"""Real Gemma full-logit parity for interleaved verification weight reuse."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if (out/'validation.json').exists():raise SystemExit('Refusing to overwrite evidence')
source=(root/'backend/tests/test_verify.c').read_text()
assert 'size_t count=17,' in source
source=source.replace('size_t count=17,','size_t count=getenv("NYA_TEST_VERIFY_TWO")?2:4,',1)
model=out/'model.c';model.write_text(source)
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy()
env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
exe=root/'build-cuda/interleaved-verify.exe'
command=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(model),'build-cuda/libnya-engine.a','-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(command,env=env,check=True)
settings=json.loads((out.parent/'mtp-profile-20261114/profile.json').read_text())['environment']
settings.update(NYA_CUDA_PROFILE='0',NYA_TEST_VERIFY_INTERLEAVED='1',NYA_TEST_VERIFY_RESIDENT='1')
env.update(settings)
meta={'command':command,'environment':settings,'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model,exe,root/'build-cuda/libnya-engine.a',root/'CUDA/resident.inc')},'cases':[]}
for ring in (0,1):
 for batch in (2,4):
  env['NYA_CUDA_RING_KV']=str(ring)
  if batch==2:env['NYA_TEST_VERIFY_TWO']='1'
  else:env.pop('NYA_TEST_VERIFY_TWO',None)
  path=out/f'real-ring{ring}-batch{batch}.log'
  with path.open('w') as log:
   result=subprocess.run([str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','257'],env=env,stdout=log,stderr=log)
  meta['cases'].append({'ring':ring,'batch':batch,'prefix':257,'exit_code':result.returncode})
  (out/'validation.json').write_text(json.dumps(meta,indent=2)+'\n')
  if result.returncode:raise SystemExit(result.returncode)
  if batch==4:assert 'CUDA interleaved weight-reuse calls:' in path.read_text()
  print(ring,batch,'passed',flush=True)
