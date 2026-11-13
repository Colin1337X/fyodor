"""Actual sanitizer validation of the joint-base architecture and edge suite."""
import hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path

def validate(code,stdout,stderr):
    if type(code) is not int or code!=0:raise ValueError('Sanitizer child failed')
    if re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$',stdout+'\n'+stderr,re.M)!=['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    matches=re.findall(r'^Joint edge cases: (\d+); matched_errors=(\d+); eos=(\d+); context=(\d+)\s*$',stdout,re.M)
    if len(matches)!=1 or matches[0][0]!='22' or any(int(v)<=0 for v in matches[0][1:]):
        raise ValueError('Missing edge-case coverage')
    if 'MTP joint base: enabled' not in stderr:raise ValueError('Joint verification was not selected')

def main():
    root=Path.cwd();out=Path(__file__).resolve().parent
    if len(sys.argv)!=2 or sys.argv[1] not in ('ring','linear'):raise SystemExit('Choose ring or linear')
    mode=sys.argv[1];folder=out/f'edge-memcheck-{mode}'
    if folder.exists():raise SystemExit('Refusing to overwrite sanitizer evidence')
    spec=importlib.util.spec_from_file_location('gate',out.parent/'gemma4-qkv-20261029/environment.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
        raise SystemExit('Wait for other GPU workloads or unknown desktop contexts')
    build=json.loads((out/'edge-fixture-build.json').read_text())
    for name,expected in build['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    ordinary=json.loads((out/f'edge-fixture-{mode}/result.json').read_text())
    assert ordinary['passed'] is True
    for name,expected in ordinary['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    folder.mkdir();work=root/'build-cuda/joint-base-generation-v2-20261114'
    env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
    env.update(NYA_COMPUTE='cuda',NYA_TEST_MTP_COMPARE='1',NYA_CUDA_BATCH='8',NYA_CUDA_RING_KV='1' if mode=='ring' else '0',NYA_MTP_RESIDENT_TARGET='1',NYA_MTP_TRACE='1',NYA_SPECULATIVE_BATCH='1',NYA_CUDA_SMALL_BATCH='1')
    result=dict(accepted_benchmark=False,passed=False,runs=[],errors=[],environment={k:v for k,v in env.items() if k.startswith('NYA_')},sanitizer_sha256=hashlib.sha256(sanitizer.read_bytes()).hexdigest())
    try:
        for standard in ('c17','c23'):
            command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99',str(work/f'architecture-edge-{standard}.exe'),str(work/f'edge-memcheck-{mode}-{standard}.gguf')]
            with (folder/f'{standard}.stdout').open('xb') as stdout,(folder/f'{standard}.stderr').open('xb') as stderr:
                child=subprocess.run(command,stdout=stdout,stderr=stderr,env=env)
            result['runs'].append(dict(standard=standard,command=command,exit_code=child.returncode))
            validate(child.returncode,(folder/f'{standard}.stdout').read_text(),(folder/f'{standard}.stderr').read_text())
        result['passed']=True
    except BaseException as error:
        result['errors'].append(str(error) or type(error).__name__);raise
    finally:
        inputs=[out/'edge-fixture-build.json',out/f'edge-fixture-{mode}/result.json',Path(__file__),*folder.glob('*.stdout'),*folder.glob('*.stderr')]
        result['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        (folder/'result.json').write_text(json.dumps(result,indent=2))
    print(mode,'edge sanitizer passed for C17 and C23')
if __name__=='__main__':main()
