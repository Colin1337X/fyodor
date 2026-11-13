"""Run the existing architecture suite; CPU fallback or exclusive GPU controls."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=2 or sys.argv[1] not in ('cpu','ring','linear','failure'):
    raise SystemExit('Choose cpu, ring, linear or failure')
mode=sys.argv[1]
if mode!='cpu':
    required=out.parent/'interleaved-integer-20261114/verify-prefix8192-ring1-count4.json'
    if not required.exists():raise SystemExit('Wait for the existing 8K validation')
    assert json.loads(required.read_text())['exit_code']==0
build=json.loads((out/'fixture-build.json').read_text())
for name,expected in build['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
folder=out/f'fixture-{mode}';folder.mkdir()
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cpu' if mode=='cpu' else 'cuda', NYA_TEST_MTP_COMPARE='1',
           NYA_CUDA_BATCH='8', NYA_CUDA_RING_KV='0' if mode=='linear' else '1',
           NYA_MTP_RESIDENT_TARGET='1', NYA_MTP_TRACE='1',NYA_SPECULATIVE_BATCH='1',NYA_CUDA_SMALL_BATCH='1')
if mode=='failure':env['NYA_CUDA_FAIL_AFTER']='0'
work=root/'build-cuda/joint-base-generation-20261114'
result=dict(accepted_benchmark=False,environment={k:v for k,v in env.items() if k.startswith('NYA_')},runs=[],passed=False)
try:
    for standard in ('c17','c23'):
        command=[str(work/f'architecture-{standard}.exe'), str(work/f'fixture-{mode}-{standard}.gguf')]
        with (folder/f'{standard}.stdout').open('xb') as stdout,(folder/f'{standard}.stderr').open('xb') as stderr:
            child=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
        trace=(folder/f'{standard}.stderr').read_text()
        selected='MTP joint base: enabled' in trace
        result['runs'].append(dict(standard=standard,command=command,exit_code=child.returncode,joint_base_selected=selected))
        assert child.returncode==0
        assert selected==(mode in ('ring','linear'))
    result['passed']=True
finally:
    result['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'fixture-build.json',Path(__file__),*folder.glob('*.stdout'),*folder.glob('*.stderr')]}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
print(mode,'architecture tests passed for C17 and C23')
