"""Live collector ownership checks using CPU-only controls, not benchmarks."""
import json
from pathlib import Path
import subprocess
import sys
import time
from coverage import audit
from monitor_capture import capture, read_complete

out = Path(__file__).resolve().parent / 'capture-cpu-control'
out.mkdir(exist_ok=False)
normal = out / 'normal'; normal.mkdir()
metadata = dict(accepted_benchmark=False, control='CPU-only sleep while unrelated GPU sanitizer is active')
with capture(normal) as monitor:
    start = time.perf_counter()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'])
    status = child.wait(timeout=15)
    end = time.perf_counter()
    metadata.update(process_id=child.pid, start=start, end=end, exit_code=status)
assert status == 0 and monitor['successful_stop'] and not monitor['errors']
rows = read_complete(normal / 'gpu-pdh.jsonl')
metadata['coverage'] = audit(rows, luid=monitor['adapter']['counter_luid'],
                             start=start, end=end, lifetimes={child.pid: (start, end)})
metadata['normal_lifecycle_passed'] = True
exception = out / 'exception'; exception.mkdir()
try:
    with capture(exception):
        raise RuntimeError('injected workload failure')
except RuntimeError as error:
    assert str(error) == 'injected workload failure'
else:
    raise AssertionError('Workload error was swallowed')
failed = json.loads((exception / 'gpu-monitor.json').read_text())
assert failed['successful_stop'] and failed['errors'] == ['injected workload failure']
metadata['exception_cleanup_passed'] = True
with (out / 'validation.json').open('x') as result:
    json.dump(metadata, result, indent=2)
print('Collector lifecycle and exception cleanup pass; sampled coverage clear:',
      metadata['coverage']['sampled_coverage_clear'])
