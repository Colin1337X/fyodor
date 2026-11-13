"""Validate the standalone experiment after the preceding GPU jobs finish."""
import json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
build_name=sys.argv[1] if len(sys.argv)>1 else 'build-cuda'
assert build_name in ('build-cuda','build-cuda-c23')
suffix='-c23' if build_name.endswith('c23') else ''
prior=json.loads((out.parent/'wide-batch-20261113/memcheck.json').read_text())
assert len(prior)==2 and all(case['exit_code']==0 for case in prior)
if (out/f'validation{suffix}.json').exists():raise SystemExit('Refusing to overwrite validation evidence')
subprocess.run([sys.executable,str(out/'build_probe.py'),build_name],check=True)
env=os.environ.copy()
env.update(NYA_CPU_THREADS='6',NYA_CUDA_BATCH='8',NYA_CUDA_RING_KV='1',NYA_CUDA_SMALL_BATCH='1',NYA_MTP_RESIDENT_TARGET='1',NYA_MTP_TRACE='1',NYA_TEST_MTP_COMPARE='1',NYA_SPECULATIVE_BATCH='1')
results=[]
for provider in ('cpu','cuda'):
 for enabled in ('0','1'):
  env.update(NYA_COMPUTE=provider,NYA_SPECULATIVE_EARLY_REJECT=enabled)
  for test in ('generation','architectures'):
   label=f'{provider}-{enabled}-{test}{suffix}'
   command=[str(root/f'{build_name}/mtp-early-reject-{test}.exe'),str(root/f'{build_name}/test_models/early-{label}.gguf')]
   with (out/(label+'.log')).open('w') as log:
    result=subprocess.run(command,env=env,stdout=log,stderr=log)
   results.append({'provider':provider,'enabled':enabled,'test':test,'command':command,'exit_code':result.returncode})
   (out/f'validation{suffix}.json').write_text(json.dumps(results,indent=2)+'\n')
   if result.returncode:raise SystemExit(result.returncode)
   print(label,'passed',flush=True)
