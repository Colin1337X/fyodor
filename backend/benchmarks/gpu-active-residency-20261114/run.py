"""Retain PDH controls without treating System activity as known workload."""
import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
folder=out/'controls'
if folder.exists():raise SystemExit('Refusing to overwrite control evidence')
prerequisite=out.parent/'batch-graphs-20261114/model-memcheck-short/result.json'
if not prerequisite.exists():raise SystemExit('Wait for the completed batched-graph model sanitizer')
proof=json.loads(prerequisite.read_text())
if proof.get('passed') is not True or type(proof.get('exit_code')) is not int or proof['exit_code']!=0:
    raise SystemExit('Batched-graph model sanitizer did not pass')
inputs={Path(__file__).resolve(),out/'worker.py',prerequisite,
        out.parent/'gemma4-qkv-20261029/environment.py'}
inputs.update((out.parent/'gpu-monitor-20261114').glob('*.py'))
for name,expected in proof['sha256'].items():
    p=root/name
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Changed prerequisite: '+name)
    inputs.add(p)
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
spec=importlib.util.spec_from_file_location('gate',out.parent/'gemma4-qkv-20261029/environment.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
    raise SystemExit('Wait for other GPU workloads or unknown desktop contexts')
sys.path.insert(0,str(out.parent/'gpu-monitor-20261114'))
from monitor_capture import capture
folder.mkdir()
result=dict(accepted_benchmark=False,errors=[],command=[sys.executable,str(out/'worker.py')],
            input_sha256=hashes,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),out/'worker.py')})
try:
    with capture(folder) as monitor:
        with (folder/'worker.stdout').open('xb') as stdout,(folder/'worker.stderr').open('xb') as stderr:
            result['spawn_before_seconds']=time.perf_counter()
            child=subprocess.Popen(result['command'],stdout=stdout,stderr=stderr,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            result['process_id']=child.pid
            try:
                while child.poll() is None:
                    monitor.check_alive();time.sleep(.2)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=15)
                result.update(exit_code=child.returncode,exit_observed_seconds=time.perf_counter())
        if child.returncode:raise RuntimeError('Transfer control failed')
    for name,expected in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Input changed during controls: '+name)
except BaseException as error:
    result['errors'].append(str(error) or type(error).__name__)
    raise
finally:
    result['evidence_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
print('Transfer controls complete; no benchmark acceptance or System attribution inferred')
