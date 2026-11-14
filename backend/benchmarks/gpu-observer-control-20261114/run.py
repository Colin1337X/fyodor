"""Diagnostic idle PDH versus PDH plus owned nvidia-smi polling; no acceptance."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

base = Path(__file__).resolve().parent
result_path = base / 'result.json'
if result_path.exists():
    raise SystemExit('Refusing to overwrite evidence')
sys.path.insert(0, str(base.parent / 'gpu-monitor-20261114'))
from pdh_counter import Counter

identity_file = base.parent / 'gpu-activity-20261114/cuda_identity.py'
spec = importlib.util.spec_from_file_location('identity', identity_file)
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
devices = identity.identities()
if len(devices['devices']) != 1:
    raise SystemExit('Requires one CUDA adapter')
luid = devices['devices'][0]['counter_luid']
result = dict(accepted_benchmark=False, identity=devices, samples=[], errors=[],
              limitation='Idle observer control only; correlation cannot attribute System work.')
query = None
try:
    query = Counter()
    for phase in ('pdh_only', 'pdh_and_smi', 'pdh_only_after'):
        for index in range(10):
            row = dict(phase=phase, index=index, start=time.perf_counter())
            child = None
            try:
                if phase == 'pdh_and_smi':
                    child = subprocess.Popen(['nvidia-smi', '-q', '-x'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    row['helper_pid'] = child.pid
                    stdout, stderr = child.communicate(timeout=10)
                    row.update(helper_exit=child.returncode,
                               helper_end=time.perf_counter(),
                               helper_stdout=stdout.decode(), helper_stderr=stderr.decode())
                time.sleep(max(0, 1 - (time.perf_counter() - row['start'])))
                row['counters'] = [c for c in query.collect() if luid in c['instance']]
                row['observed'] = time.perf_counter()
                result['samples'].append(row)
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    child.communicate(timeout=10)
except BaseException as error:
    result['errors'].append(str(error) or type(error).__name__)
    raise
finally:
    if query is not None:
        query.close()
    result['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), identity_file,
                                 base.parent / 'gpu-monitor-20261114/pdh_counter.py')}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print('Observer control complete; diagnostic only')
