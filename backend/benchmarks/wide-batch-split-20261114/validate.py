"""Full Gemma logits and lifecycle checks for selective batched split attention."""
import hashlib,json,os,re,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if (out/'validation.json').exists():raise SystemExit('Refusing to overwrite evidence')
source=(root/'backend/tests/test_verify.c').read_text()
assert 'size_t count=17,' in source
source=source.replace('size_t count=17,','size_t count=getenv("NYA_TEST_VERIFY_TWO")?2:4,',1)
model=out/'model.c';model.write_text(source)
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy()
env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
exe=root/'build-cuda/wide-batch-split-verify.exe'
command=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(model),'build-cuda/libnya-engine.a','-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(command,env=env,check=True)
settings=json.loads((out.parent/'mtp-profile-20261114/profile.json').read_text())['environment']
settings.update(NYA_CUDA_PROFILE='0',NYA_TEST_VERIFY_BATCH_SPLIT='1',NYA_TEST_VERIFY_RESIDENT='1',
                NYA_CUDA_WIDE_BATCH='1',NYA_CUDA_WIDE_SPLIT='1',NYA_CUDA_SPLIT_ATTENTION='1',
                NYA_CUDA_INTERLEAVED_REUSE='1',NYA_CUDA_INTERLEAVED_TWO='1',NYA_CUDA_BATCH_LOGITS='1')
env.update(settings)
paths=(model,exe,root/'build-cuda/libnya-engine.a',root/'CUDA/resident.inc',root/'CUDA/cuda.c',root/'optcpp/attention/wide_batch_split16.cu')
meta={'command':command,'environment':settings,'base_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'uncommitted_runtime_changes':True,'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'cases':[]}
for prefix in (257,8192):
 for ring in (0,1):
  for batch in (2,4):
   env['NYA_CUDA_RING_KV']=str(ring)
   if batch==2:env['NYA_TEST_VERIFY_TWO']='1'
   else:env.pop('NYA_TEST_VERIFY_TWO',None)
   path=out/f'real-prefix{prefix}-ring{ring}-batch{batch}.log'
   with path.open('x') as log:
    result=subprocess.run([str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf',str(prefix)],env=env,stdout=log,stderr=log)
   text=path.read_text();calls=[int(n) for n in re.findall(r'CUDA batched split attention calls: (\d+)',text)]
   expected=batch==2 or prefix>=8192
   case={'ring':ring,'batch':batch,'prefix':prefix,'exit_code':result.returncode,'split_calls':calls,'expected_dispatch':expected}
   meta['cases'].append(case)
   (out/'validation.json').write_text(json.dumps(meta,indent=2)+'\n')
   if result.returncode:raise SystemExit(result.returncode)
   assert bool(calls)==expected and all(n>0 for n in calls),'Dispatch mismatch'
   print(prefix,ring,batch,'passed',flush=True)
