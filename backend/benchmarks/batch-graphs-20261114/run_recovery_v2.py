"""Run one isolated recovery case under Compute Sanitizer after model validation."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def main():
    if len(sys.argv)!=6:raise SystemExit('Usage: run_recovery.py c17|c23 256|512 0|1 begin|end|instantiate|launch cold|cached')
    standard,width,ring,site,cache=sys.argv[1:]
    if standard not in ('c17','c23') or width not in ('256','512') or ring not in ('0','1') or site not in ('begin','end','instantiate','launch','request','lowering','sync','readback') or cache not in ('cold','cached') or (cache=='cached' and site not in ('launch','request','sync','readback')):
        raise SystemExit('Invalid recovery case')
    root=Path.cwd();out=Path(__file__).resolve().parent
    label=f'{standard}-{width}-ring{ring}-{site}-{cache}'
    folder=out/f'recovery-v2-{label}'
    fixture=root/f'build-cuda/batch-graph-faults-v2-20261114/recovery-{label}.gguf'
    if folder.exists() or fixture.exists():raise SystemExit('Refusing to overwrite evidence')
    inputs={Path(__file__).resolve()}
    for path,key in ((out/'model-memcheck-short/result.json','passed'),(out/'recovery-build-v2.json','build_passed')):
        if not path.exists():raise SystemExit('Wait for completed prerequisite: '+str(path))
        proof=json.loads(path.read_text());inputs.add(path)
        if proof.get(key) is not True:raise ValueError('Failed prerequisite: '+str(path))
        for name,expected in proof['sha256'].items():
            p=root/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(name)
            inputs.add(p)
    settings_path=out/'lifecycle-c17-257/result.json'
    settings=json.loads(settings_path.read_text())['environment'];inputs.add(settings_path)
    sys.path.insert(0,str(out.parent/'gemma4-qkv-20261029'))
    import environment
    if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):raise SystemExit('Another GPU workload remains')
    sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    binary=root/f'build-cuda/batch-graph-faults-v2-20261114/recovery-{standard}.exe'
    inputs.update((sanitizer,binary,Path(environment.__file__)))
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99',str(binary),str(fixture),width,ring,site,cache]
    env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(settings)
    folder.mkdir();result=dict(passed=False,accepted_benchmark=False,command=command,environment=settings,errors=[])
    try:
        with (folder/'stdout').open('xb') as stdout,(folder/'stderr').open('xb') as stderr:
            child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
            result['process_id']=child.pid
            (folder/'running.json').write_text(json.dumps(result,indent=2))
            try:result['exit_code']=child.wait()
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=15)
        output=(folder/'stdout').read_text();trace=(folder/'stderr').read_text()
        if result['exit_code']!=0:raise ValueError('Recovery child failed')
        if re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$',output+'\n'+trace,re.M)!=['0']:raise ValueError('Sanitizer summary failed')
        if re.findall(r'^INJECTED graph failure: (\w+)$',trace,re.M)!=[site]:raise ValueError('Injection did not execute exactly once')
        marker=f'recovery=1 site={site} cache={cache} width={width} ring={ring}'
        if output.splitlines().count(marker)!=1:raise ValueError('Missing recovery proof')
        resources=re.findall(r'^GRAPH resources: graphs=(\d+)/(\d+) execs=(\d+)/(\d+) live=(\d+) errors=(\d+)$',trace,re.M)
        if len(resources)!=1:raise ValueError('Missing resource accounting')
        created,freed,exec_created,exec_freed,live,errors=map(int,resources[0])
        if created<2 or exec_created<2 or created!=freed or exec_created!=exec_freed or live or errors:
            raise ValueError('Unbalanced graph ownership')
        for name,expected in hashes.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed input: '+name)
        result['passed']=True
    except BaseException as error:
        result['errors'].append(str(error) or type(error).__name__);raise
    finally:
        paths=[p for p in (folder/'stdout',folder/'stderr',fixture) if p.exists()]
        result['sha256']={**hashes,**{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        (folder/'result.json').write_text(json.dumps(result,indent=2))
    print(label,'recovery passed; no timing acceptance')

if __name__=='__main__':main()
