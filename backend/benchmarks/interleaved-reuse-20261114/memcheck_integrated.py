import json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if (out/'integrated-memcheck.json').exists():raise SystemExit('Refusing to overwrite evidence')
settings=json.loads((out/'validation.json').read_text())['environment']
settings.update(NYA_CUDA_RING_KV='1')
env=os.environ.copy();env.update(settings)
command=[str(root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'),'--tool','memcheck','--error-exitcode','99',str(root/'build-cuda/interleaved-verify.exe'),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','7']
with (out/'integrated-memcheck.log').open('w') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
(out/'integrated-memcheck.json').write_text(json.dumps({'command':command,'environment':settings,'exit_code':result.returncode},indent=2)+'\n')
assert result.returncode==0
text=(out/'integrated-memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in text and 'CUDA interleaved weight-reuse calls:' in text
print('Integrated Gemma ring-KV memcheck passed',flush=True)
