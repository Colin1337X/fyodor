from pathlib import Path
import os,subprocess,json,hashlib
root=Path.cwd();out=root/'backend/benchmarks/gemma4-ring-runtime-20261113'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(NYA_COMPUTE='cuda',NYA_TEST_GRAPH='1',NYA_TEST_RING_KV='1',NYA_CUDA_RING_KV='1',NYA_CUDA_BATCH='16',NYA_CUDA_GRAPHS='1')
exe=root/'build-cuda-c23/nya-test-inference.exe'
san=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
cmd=[str(san),'--tool','memcheck','--error-exitcode','99',str(exe),'--attention-shapes',str(root/'build-cuda-c23/test_models/ring-memcheck')]
p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(out/'memcheck.log').write_bytes(p.stdout+p.stderr)
(out/'memcheck-results.json').write_text(json.dumps(dict(exit_code=p.returncode,command=cmd,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),environment={k:v for k,v in env.items() if k.startswith('NYA_')}),indent=2)+'\n')
print('memcheck exit',p.returncode,flush=True)
raise SystemExit(p.returncode)
