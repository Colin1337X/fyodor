"""Memory-check admitted partial buffers through real native session lifecycle."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if (out/'integrated-memcheck.json').exists():raise SystemExit('Refusing to overwrite evidence')
exe=root/'build-cuda/nya-test-verify.exe'
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
settings=dict(NYA_COMPUTE='cuda',NYA_CUDA_WIDE_VALUE='1',NYA_CUDA_WIDE_BATCH='1',
              NYA_CUDA_WIDE_SPLIT='1',NYA_CUDA_SPLIT_ATTENTION='1',NYA_TEST_VERIFY_RESIDENT='1',
              NYA_TEST_VERIFY_BATCH_SPLIT='1',NYA_MTP_TRACE='1')
meta={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'synthetic_integrated_models':True,'cases':[]}
for width,batch,ring,prefix in ((256,2,1,257),(512,4,0,8192)):
 env=os.environ.copy();env.update(settings)
 env.update(NYA_TEST_VERIFY_WIDE=str(width),NYA_CUDA_BATCH=str(batch),NYA_CUDA_RING_KV=str(ring))
 if ring:env['NYA_TEST_VERIFY_RING']='1'
 else:env.pop('NYA_TEST_VERIFY_RING',None)
 command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99',str(exe),str(root/'build-cuda/test_models/pretraining.trained.gguf'),str(prefix)]
 path=out/f'integrated-memcheck-{width}-batch{batch}.log'
 with path.open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
 text=path.read_text()
 case={'width':width,'batch':batch,'ring':ring,'prefix':prefix,'command':command,
       'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'exit_code':result.returncode,
       'zero_errors':'ERROR SUMMARY: 0 errors' in text,'dispatch_observed':'CUDA batched split attention calls:' in text}
 meta['cases'].append(case);(out/'integrated-memcheck.json').write_text(json.dumps(meta,indent=2)+'\n')
 assert result.returncode==0 and case['zero_errors'] and case['dispatch_observed']
 print(width,batch,'integrated memcheck passed',flush=True)
