"""Run one explicit precision-path isolation, retaining failures and exact binary."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=2 or sys.argv[1] not in ('native-matmul','unsplit-attention'):
 raise SystemExit('Choose native-matmul or unsplit-attention')
mode=sys.argv[1];result_path=out/f'replay-{mode}.json'
if result_path.exists():raise SystemExit('Refusing to overwrite evidence')
control=json.loads((out/'replay-control.json').read_text())
command=control['command'];exe=Path(command[0])
assert hashlib.sha256(exe.read_bytes()).hexdigest()==control['sha256'][str(exe.relative_to(root))]
settings=control['environment'].copy()
if mode=='native-matmul':settings['NYA_CUDA_BLAS']='0'
else:settings.update(NYA_CUDA_WIDE_SPLIT='0',NYA_CUDA_FLASH_DECODING_PP='0')
env=os.environ.copy();env.update(settings);env.pop('NYA_TEST_VERIFY_BATCH_SPLIT',None)
with (out/f'replay-{mode}.log').open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
meta={'mode':mode,'command':command,'environment':settings,'exit_code':result.returncode,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
result_path.write_text(json.dumps(meta,indent=2)+'\n')
print((out/f'replay-{mode}.log').read_text(),flush=True)
raise SystemExit(result.returncode)
