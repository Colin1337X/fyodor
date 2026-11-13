"""Run only after the timing pair finishes; never overlap GPU evaluation."""
import json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
summary=json.loads((out/'runtime-summary.json').read_text())
assert summary['accepted_llama_score'] is False
for label in ('runtime-off','runtime-on'):
 runs=json.loads((out/label/'timing.json').read_text())['runs']
 assert len(runs)==2 and all(row['exit_code']==0 for row in runs)
env=os.environ.copy()
env.update(NYA_COMPUTE='cuda',NYA_CUDA_WIDE_VALUE='1',NYA_CUDA_WIDE_BATCH='1',
           NYA_TEST_VERIFY_RESIDENT='1',NYA_MTP_TRACE='1')
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
results=[]
for width,ring,batch in ((256,0,2),(512,1,4)):
 env.update(NYA_TEST_VERIFY_WIDE=str(width),NYA_CUDA_RING_KV=str(ring),NYA_CUDA_BATCH=str(batch))
 if ring:env['NYA_TEST_VERIFY_RING']='1'
 else:env.pop('NYA_TEST_VERIFY_RING',None)
 path=out/f'memcheck-{width}-ring{ring}-batch{batch}.log'
 if path.exists():raise SystemExit('Refusing to overwrite '+str(path))
 command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99',str(root/'build-cuda/nya-test-verify.exe'),str(root/'build-cuda/test_models/pretraining.trained.gguf'),'257']
 with path.open('w') as log:
  result=subprocess.run(command,env=env,stdout=log,stderr=log)
 results.append({'command':command,'width':width,'ring':ring,'batch':batch,'exit_code':result.returncode})
 (out/'memcheck.json').write_text(json.dumps(results,indent=2)+'\n')
 if result.returncode:raise SystemExit(result.returncode)
 text=path.read_text()
 assert 'ERROR SUMMARY: 0 errors' in text and 'CUDA batched wide-value attention calls:' in text
 print(width,ring,batch,'memcheck passed',flush=True)
