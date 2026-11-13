"""Real Gemma verification against scalar logits, reset and rejected suffix replay."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
env=os.environ.copy();compiler=root/'.tools/w64devkit/bin/gcc.exe'
env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
settings=json.loads((out.parent/'mtp-target-profile-20261113/profile.json').read_text())['environment']
settings.update(NYA_CUDA_PROFILE='0',NYA_CUDA_WIDE_BATCH='1',NYA_TEST_VERIFY_RESIDENT='1')
env.update(settings)
exe=root/'build-cuda/wide-batch-verify.exe'
command=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(out/'model.c'),'build-cuda/libnya-engine.a','-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(command,env=env,check=True)
meta={'command':command,'environment':settings,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'cases':[]}
for ring in (0,1):
 for batch in (2,4):
  for prefix in (257,1025):
   env['NYA_CUDA_RING_KV']=str(ring)
   if batch==2:env['NYA_TEST_VERIFY_TWO']='1'
   else:env.pop('NYA_TEST_VERIFY_TWO',None)
   path=out/f'real-ring{ring}-batch{batch}-prefix{prefix}.log'
   if path.exists():raise SystemExit('Refusing to overwrite '+str(path))
   with path.open('w') as log:
    result=subprocess.run([str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf',str(prefix)],env=env,stdout=log,stderr=log)
   meta['cases'].append({'ring':ring,'batch':batch,'prefix':prefix,'exit_code':result.returncode})
   (out/'validation.json').write_text(json.dumps(meta,indent=2)+'\n')
   if result.returncode:raise SystemExit(result.returncode)
   assert 'CUDA batched wide-value attention calls:' in path.read_text()
   print(ring,batch,prefix,'passed',flush=True)
