"""Full-model joint-base correctness diagnostic, never accepted timing."""
import hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=2 or sys.argv[1] not in ('short','long'):raise SystemExit('Choose short or long')
context=sys.argv[1];folder=out/f'{context}-correctness'
if folder.exists():raise SystemExit('Refusing to overwrite model evidence')
for mode in ('cpu','ring','linear','failure'):
    result=json.loads((out/f'fixture-{mode}/result.json').read_text())
    assert result['passed'] is True
    for name,expected in result['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
build=json.loads((out/'generation-build.json').read_text())
for name,expected in build['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
spec=importlib.util.spec_from_file_location('gate',out.parent/'gemma4-qkv-20261029/environment.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(PYTHONDONTWRITEBYTECODE='1',NYA_CUDA_PROFILE='0',NYA_BENCH_GPU_MONITOR='0',
    NYA_CUDA_ALIGNED_PREFIX='1',NYA_CUDA_WIDE_BATCH_SPLIT='1',NYA_CUDA_WIDE_BATCH='1',
    NYA_CUDA_SMALL_BATCH_REUSE='1',NYA_CUDA_INTERLEAVED_REUSE='1',NYA_CUDA_INTERLEAVED_TWO='1',
    NYA_CUDA_INTERLEAVED_INTEGER='0',NYA_CUDA_BATCH_LOGITS='1',NYA_CUDA_ASSISTED_FFN='1',
    NYA_CUDA_MTP_WIDE_VALUE='1',NYA_SPECULATIVE_EARLY_REJECT='1')
command=[sys.executable,str(out.parent/'mtp-cost-20261113/run.py'),str(folder),
    'build-cuda/joint-base-generation-v2-20261114/generation.exe',context]
subprocess.run(command,env=env,check=True)
trace=(folder/f'{context}.stderr').read_text()
assert trace.count('MTP joint base: enabled; proposal window=3')==3
rows=[json.loads(line) for line in (folder/f'{context}.stdout').read_text().splitlines() if line.startswith('{"round":')]
assert len(rows)==12 and all(r['parity'] is True and r['generated']==32 for r in rows)
(folder/'correctness.json').write_text(json.dumps(dict(accepted_benchmark=False,passed=True,cases=len(rows),joint_rounds=3,
    note='Output parity only. No continuous coverage; timings are not accepted.',
    sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'generation-build.json',Path(__file__),folder/f'{context}.stdout',folder/f'{context}.stderr',folder/'timing.json']}),indent=2))
print('Full-model',context,'output parity passed; three joint-base rounds')
