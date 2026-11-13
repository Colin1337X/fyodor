import json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
validation=json.loads((out/'validation.json').read_text())
assert len(validation['cases'])==4 and all(r['exit_code']==0 for r in validation['cases'])
if (out/'memcheck.json').exists():raise SystemExit('Refusing to overwrite evidence')
env=os.environ.copy();env.update(validation['environment'])
env.update(NYA_CUDA_RING_KV='1',NYA_MTP_TRACE='1')
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
cases=[]
for label,exe,model,prefix in [
 ('tiny','build-cuda/nya-test-verify.exe','build-cuda/test_models/pretraining.trained.gguf','7'),
 ('gemma','build-cuda/batch-logits-verify.exe','E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','257')]:
 env['NYA_CUDA_BATCH']='4' if label=='tiny' else '512'
 command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99',str(root/exe),model,prefix]
 with (out/(label+'-memcheck.log')).open('w') as log:
  result=subprocess.run(command,env=env,stdout=log,stderr=log)
 cases.append({'command':command,'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'exit_code':result.returncode})
 (out/'memcheck.json').write_text(json.dumps(cases,indent=2)+'\n')
 assert result.returncode==0
 text=(out/(label+'-memcheck.log')).read_text()
 assert 'ERROR SUMMARY: 0 errors' in text and 'CUDA batched output projection calls:' in text
 print(label,'memory check passed',flush=True)
