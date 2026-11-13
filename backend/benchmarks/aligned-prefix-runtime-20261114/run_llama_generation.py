"""Run the independent reference with retained environment and PDH evidence."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from llama_intervals import parse

root = Path.cwd()
out = Path(__file__).resolve().parent
monitor_directory = out.parent / 'gpu-monitor-20261114'
sys.path.insert(0, str(monitor_directory))
from monitor_capture import capture
from coverage import audit
from audit_generation import lifetime

audit_path = out / 'verify-prefix513-ring1-count4-memcheck-audit.json'
if not audit_path.exists():
    raise SystemExit('Wait for completed integrated sanitizer audit')
sanitizer = json.loads(audit_path.read_text())
assert sanitizer['sanitizer_zero_errors'] is True and sanitizer['numerical_lifecycle_passed'] is True
native_path = out / 'monitored-generation/timing.json'
if not native_path.exists():
    raise SystemExit('Complete the native diagnostic first; never overlap GPU workloads')
native = json.loads(native_path.read_text())
assert [row['context'] for row in native['runs']] == ['short', 'long']
for row in native['runs']:
    lifetime(row)
build = json.loads((out / 'llama-generation-build.json').read_text())
tokens = json.loads((out / 'prompt-token-parity/result.json').read_text())
for metadata in (build, tokens):
    for name, expected in metadata['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Input changed: {name}'
for context in ('short', 'long'):
    assert tokens[context]['identical'] is True
folder = out / 'monitored-llama-generation'
folder.mkdir()  # Existing evidence must never be overwritten.
env = {key: value for key, value in os.environ.items() if not key.startswith(('NYA_', 'GGML_', 'LLAMA_'))}
env.pop('CUDA_VISIBLE_DEVICES', None)
env['PATH'] = str(root / '.tools/llama-b10809-cuda') + os.pathsep + env['PATH']
spec = importlib.util.spec_from_file_location('environment', out.parent / 'gemma4-qkv-20261029/environment.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
gate.configure(folder)
gate.idle()  # A failed idle gate stops this launch; it is not an accepted measurement.
metadata = dict(accepted_benchmark=False, gate_passed=True, runs=[], errors=[],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                build_manifest_sha256=hashlib.sha256((out / 'llama-generation-build.json').read_bytes()).hexdigest())
result = dict(accepted_benchmark=False, runs=[], errors=[], sampled_coverage_clear=False)
def save():
    (folder / 'timing.json').write_text(json.dumps(metadata, indent=2))

try:
    with capture(folder) as monitor:
        for context in ('short', 'long'):
            monitor.check_alive()
            command = [str(root / 'build-cuda/aligned-generation-20261114/llama-generation-c17.exe'), tokens['model'], context]
            row = dict(context=context, command=command, observer_clock='qpc', samples=[])
            metadata['runs'].append(row)
            with (folder / f'{context}.stdout').open('xb') as stdout, (folder / f'{context}.stderr').open('xb') as stderr:
                row['spawn_before_seconds'] = time.perf_counter()
                process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                row.update(spawn_after_seconds=time.perf_counter(), process_id=process.pid)
                try:
                    while process.poll() is None:
                        monitor.check_alive()
                        row['samples'].append(gate.snapshot()); save(); time.sleep(1)
                except BaseException as error:
                    row['observer_error'] = str(error) or type(error).__name__
                    raise
                finally:
                    if process.poll() is None:
                        process.terminate(); process.wait(timeout=15)
                        row['terminated_by_observer'] = True
                    row.update(exit_observed_seconds=time.perf_counter(), exit_code=process.returncode)
                    save()
            lifetime(row)
            print(context, row['exit_code'], flush=True)
    samples = [json.loads(line) for line in (folder / 'gpu-pdh.jsonl').read_text().splitlines()]
    spans = dict(lifetime(row) for row in metadata['runs'])
    assert len(spans) == len(metadata['runs']), 'PID reuse requires independent attribution'
    for row in metadata['runs']:
        context = row['context']
        expected = [int(value) for value in (out / f'prompt-token-parity/{context}-fyodor.stdout').read_text().split()]
        observations = parse((folder / f'{context}.stdout').read_text(), expected, spans[row['process_id']])
        for observation in observations:
            observation['coverage'] = audit(samples, luid=monitor['adapter']['counter_luid'],
                start=observation['start'], end=observation['end'], lifetimes=spans)
        result['runs'].append(dict(context=context, observations=observations))
    result['sampled_coverage_clear'] = all(observation['coverage']['sampled_coverage_clear']
        for run in result['runs'] for observation in run['observations'])
except BaseException as error:
    result['errors'].append(str(error) or type(error).__name__)
    raise
finally:
    result['sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in folder.iterdir() if path.is_file()}
    (folder / 'gpu-coverage.json').write_text(json.dumps(result, indent=2))
if not result['sampled_coverage_clear']:
    raise SystemExit('Reference coverage incomplete or contaminated; retained as diagnostic only')
