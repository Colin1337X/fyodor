"""Sanitize topology transitions after the real-model run; never accept timing."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def validate(code, stdout, stderr):
    if type(code) is not int or code != 0:
        raise ValueError('Boundary sanitizer failed')
    summaries = re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$', stdout+'\n'+stderr, re.M)
    if summaries != ['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    cases = re.findall(r'^reset=(\d+) position=(\d+) count=(\d+) trial=(\d+) parity=1$', stdout, re.M)
    expected = [(str(reset),str(pos),str(count),str(trial)) for reset in range(2)
                for pos,count in ((250,4),(254,2),(256,2),(258,4),(8190,2),(8192,4),(8196,2))
                for trial in range(3)]
    if cases != expected:
        raise ValueError('Incomplete or reordered boundary cases')
    if re.findall(r'CUDA batched graph captures: (\d+) replays: (\d+)', stderr) != [('6','42')]:
        raise ValueError('Topology capture/reuse failed')


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ('c17','c23') or sys.argv[2] not in ('256','512') or sys.argv[3] not in ('0','1'):
        raise SystemExit('Usage: run_boundary_memcheck.py c17|c23 256|512 0|1')
    standard,width,ring = sys.argv[1:]
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    folder = out/f'boundary-memcheck-{standard}-{width}-ring{ring}'
    fixture = root/f'build-cuda/batch-graphs-20261114/memcheck-{standard}-{width}-ring{ring}.gguf'
    if folder.exists() or fixture.exists():
        raise SystemExit('Refusing to overwrite evidence')
    inputs = {Path(__file__).resolve()}
    for path in (out/'model-memcheck-short/result.json', out/f'boundaries-{standard}-{width}-ring{ring}/result.json'):
        if not path.exists():
            raise SystemExit('Wait for completed prerequisite: '+str(path))
        proof = json.loads(path.read_text())
        if proof.get('passed') is not True:
            raise ValueError('Failed prerequisite: '+str(path))
        inputs.add(path)
        for name,expected in proof['sha256'].items():
            p = root/name
            if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise ValueError('Changed prerequisite: '+name)
            inputs.add(p)
    settings = proof['environment']
    sys.path.insert(0,str(out.parent/'gemma4-qkv-20261029'))
    import environment
    if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
        raise SystemExit('Another GPU workload remains')
    sanitizer = root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
    binary = root/f'build-cuda/batch-graphs-20261114/boundaries-{standard}.exe'
    inputs.update((sanitizer,binary,Path(environment.__file__)))
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    command = [str(sanitizer),'--tool','memcheck','--error-exitcode','99',str(binary),str(fixture),width,ring]
    env = {k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
    env.update(settings)
    folder.mkdir()
    result = dict(passed=False,accepted_benchmark=False,command=command,environment=settings,errors=[])
    try:
        with (folder/'stdout').open('xb') as stdout,(folder/'stderr').open('xb') as stderr:
            child = subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
            result['process_id'] = child.pid
            (folder/'running.json').write_text(json.dumps(result,indent=2))
            try:
                result['exit_code'] = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=15)
        validate(result['exit_code'],(folder/'stdout').read_text(),(folder/'stderr').read_text())
        for name,expected in hashes.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Input changed: '+name)
        result['passed'] = True
    except BaseException as error:
        result['errors'].append(str(error) or type(error).__name__)
        raise
    finally:
        paths = [p for p in (folder/'stdout',folder/'stderr',fixture) if p.exists()]
        result['sha256'] = {**hashes,**{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        (folder/'result.json').write_text(json.dumps(result,indent=2))
    print('42 sanitized boundary cases passed; no timing acceptance')


if __name__ == '__main__':
    main()
