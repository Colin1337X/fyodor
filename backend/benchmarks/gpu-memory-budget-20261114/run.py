"""Capture in-process budgets around an otherwise matched generation diagnostic."""
import hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;folder=out/'short'
if folder.exists():raise SystemExit('Refusing to overwrite budget evidence')
build=json.loads((out/'generation-build.json').read_text())
for name,expected in build['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
spec=importlib.util.spec_from_file_location('gate',out.parent/'gemma4-qkv-20261029/environment.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(PYTHONDONTWRITEBYTECODE='1',NYA_BENCH_GPU_MONITOR='1',NYA_CUDA_PROFILE='0',NYA_CUDA_ALIGNED_PREFIX='1',
 NYA_CUDA_WIDE_BATCH_SPLIT='1',NYA_CUDA_WIDE_BATCH='1',NYA_CUDA_SMALL_BATCH_REUSE='1',NYA_CUDA_INTERLEAVED_REUSE='1',
 NYA_CUDA_INTERLEAVED_TWO='1',NYA_CUDA_INTERLEAVED_INTEGER='0',NYA_CUDA_BATCH_LOGITS='1',NYA_CUDA_ASSISTED_FFN='1',
 NYA_CUDA_MTP_WIDE_VALUE='1',NYA_SPECULATIVE_EARLY_REJECT='1')
command=[sys.executable,str(out.parent/'mtp-cost-20261113/run.py'),str(folder),'build-cuda/budget-generation-20261114/generation.exe','short']
result=subprocess.run(command,env=env)
raise SystemExit(result.returncode)
