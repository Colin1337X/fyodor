"""Run integration checks before same-binary diagnostic generation timings."""
import json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
validated=json.loads((out/'validation.json').read_text())['cases']
assert len(validated)==8 and all(row['exit_code']==0 for row in validated)
env=os.environ.copy();env['PATH']=str(root/'.tools/w64devkit/bin')+os.pathsep+env['PATH']
env.update(NYA_CUDA_WIDE_BATCH='1',NYA_CUDA_WIDE_VALUE='1',NYA_MTP_TRACE='1')
for build in ('build-cuda','build-cuda-c23'):
 with (out/(build+'-tests.log')).open('w') as log:
  subprocess.run([str(root/'.tools/w64devkit/bin/ctest.exe'),'--test-dir',build,'-R','inference-verify','--output-on-failure'],env=env,stdout=log,stderr=log,check=True)
 print(build,'verification tests passed',flush=True)
subprocess.run([sys.executable,'backend/benchmarks/mtp-cost-20261113/build_probe.py',str(out/'build')],env=env,check=True)
env.update(NYA_CUDA_SMALL_BATCH_REUSE='1',NYA_CUDA_MTP_WIDE_VALUE='0')
for label,enabled in [('runtime-off','0'),('runtime-on','1')]:
 env['NYA_CUDA_WIDE_BATCH']=enabled
 subprocess.run([sys.executable,'backend/benchmarks/mtp-cost-20261113/run.py','../wide-batch-20261113/'+label,'build-cuda/mtp-cost-candidate-b512.exe'],env=env,check=True)
subprocess.run([sys.executable,str(out/'summarize_runtime.py')],check=True)
