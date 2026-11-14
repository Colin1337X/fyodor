"""Run the independent reference with retained environment and PDH evidence."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
reference = Path(__file__).resolve().parent.parent / 'aligned-prefix-runtime-20261114'
sys.path.insert(0, str(reference))
from llama_intervals import parse
from compare_generation import native_text
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'mtp-joint-base-v2-20261114'))
from model_identity import verify_gemma

root = Path.cwd()
out = Path(__file__).resolve().parent
monitor_directory = out.parent / 'gpu-monitor-20261114'
sys.path.insert(0, str(monitor_directory))
from monitor_capture import capture
from coverage import audit
from audit_generation import lifetime

if len(sys.argv) not in (3, 4) or sys.argv[1] not in ('short', 'long') or (len(sys.argv)==4 and sys.argv[3]!='guarded'):
    raise SystemExit('Usage: run_reference.py short|long UNIQUE_LABEL [guarded]')
context, label = sys.argv[1:3]
guarded = len(sys.argv)==4
build_manifest = out.parent / 'mtp-joint-base-v2-20261114/guarded-reference-build.json' if guarded else reference / 'llama-generation-build.json'
binary_directory = 'guarded-reference-20261114' if guarded else 'aligned-generation-20261114'
if not label.isascii() or not label.isalnum():
    raise SystemExit('Label must be ASCII alphanumeric')
folder = out / f'reference-{"guarded-" if guarded else ""}{context}-{label}'
if folder.exists():
    raise SystemExit('Refusing to overwrite reference evidence')
spec = importlib.util.spec_from_file_location('preflight', out.parent / 'gemma4-qkv-20261029/environment.py')
preflight = importlib.util.module_from_spec(spec); spec.loader.exec_module(preflight)
if any(not preflight.desktop_identity_allowed(p) for p in preflight.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains; no reference launched')
inputs = {Path(__file__).resolve()}
proof_paths = [build_manifest, out.parent / 'batch-graphs-20261114/prompt-token-parity/result.json', out / 'build.json',
               out / 'model-memcheck-short/result.json', out / f'generation-{context}-c17/result.json']
for path in proof_paths:
    inputs.add(path)
    if not path.exists():
        raise SystemExit('Wait for completed prerequisite: ' + str(path))
    proof = json.loads(path.read_text())
    if path == build_manifest and proof.get('upstream_revision') != '5266f24da75dc449bd56cbed7addb9c8e4a6a73e':
        raise SystemExit('Reference revision differs from pinned baseline')
    if path == out / 'build.json' and proof.get('build_passed') is not True:
        raise SystemExit('Candidate build did not pass')
    if path.parent.name.startswith(('model-memcheck-', 'generation-')) and proof.get('passed') is not True:
        raise SystemExit(f'Prerequisite did not pass: {path}')
    for name, expected in proof['sha256'].items():
        inputs.add(root / name)
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Changed prerequisite: {name}')
tokens = json.loads(proof_paths[1].read_text())
if tokens[context]['identical'] is not True:
    raise SystemExit('Prompt token IDs differ')
expected_text = native_text((out / f'generation-{context}-c17/stdout').read_text())
model_identity = verify_gemma()
folder.mkdir()

env = {key: value for key, value in os.environ.items() if not key.startswith(('NYA_', 'GGML_', 'LLAMA_'))}
env.pop('CUDA_VISIBLE_DEVICES', None)
library_directory = root / '.tools/llama-b10809-cuda'
env['PATH'] = str(library_directory) + os.pathsep + env['PATH']
spec = importlib.util.spec_from_file_location('environment', out.parent / 'gemma4-qkv-20261029/environment.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
inputs.update(library_directory.glob('*.dll'))
inputs.update(monitor_directory.glob('*.py'))
inputs.update((reference/'llama_intervals.py', reference/'compare_generation.py',
               out.parent/'mtp-joint-base-v2-20261114/model_identity.py', Path(gate.__file__)))
input_hashes = {(str(p.relative_to(root)) if p.is_relative_to(root) else str(p)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
(folder/'input-sha256.json').write_text(json.dumps(input_hashes,indent=2))
gate.configure(folder)
gate.idle()  # A failed idle gate stops this launch; it is not an accepted measurement.
metadata = dict(accepted_benchmark=False, gate_passed=True, runs=[], errors=[],
                quiet_gap_seconds=2 if guarded else 0,
                model_identity=model_identity,
                child_cwd=str(library_directory),
                library_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in library_directory.glob('*.dll')},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                build_manifest_sha256=hashlib.sha256(build_manifest.read_bytes()).hexdigest())
result = dict(accepted_benchmark=False, runs=[], errors=[], sampled_coverage_clear=False,
              input_sha256=input_hashes,
              prerequisite_sha256={(str(p.relative_to(root)) if p.is_relative_to(root) else str(p)): hashlib.sha256(p.read_bytes()).hexdigest() for p in proof_paths},
              validator_sha256={(str(p.relative_to(root)) if p.is_relative_to(root) else str(p)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [out.parent / 'mtp-joint-base-v2-20261114/model_identity.py', reference / 'llama_intervals.py', reference / 'compare_generation.py',
                            monitor_directory / 'coverage.py', monitor_directory / 'audit_generation.py',
                            monitor_directory / 'monitor_capture.py']},
              expected_output_sha256=hashlib.sha256(expected_text).hexdigest())
def save():
    (folder / 'timing.json').write_text(json.dumps(metadata, indent=2))

try:
    with capture(folder) as monitor:
        for context in (context,):
            monitor.check_alive()
            command = [str(root / 'build-cuda' / binary_directory / 'llama-generation-c17.exe'), tokens['model'], context]
            row = dict(context=context, command=command, observer_clock='qpc', samples=[])
            metadata['runs'].append(row)
            with (folder / f'{context}.stdout').open('xb') as stdout, (folder / f'{context}.stderr').open('xb') as stderr:
                row['spawn_before_seconds'] = time.perf_counter()
                process = subprocess.Popen(command, env=env, cwd=library_directory, stdout=stdout, stderr=stderr)
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
        expected = [int(value) for value in (out.parent / 'batch-graphs-20261114/prompt-token-parity' / f'{context}-fyodor.stdout').read_text().split()]
        observations = parse((folder / f'{context}.stdout').read_text(), expected, spans[row['process_id']])
        if any(bytes.fromhex(observation['text_hex']) != expected_text for observation in observations):
            raise ValueError('Reference output bytes differ from validated native output')
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
    changed = [name for name,h in input_hashes.items() if not (root/name).is_file()
               or hashlib.sha256((root/name).read_bytes()).hexdigest()!=h]
    result['changed_inputs'] = changed
    result['inputs_unchanged'] = not changed
    if changed:
        result['errors'].append('Reference inputs changed: ' + ', '.join(changed))
        result['sampled_coverage_clear'] = False
    result['sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in folder.iterdir() if path.is_file()}
    (folder / 'gpu-coverage.json').write_text(json.dumps(result, indent=2))
if not result['sampled_coverage_clear']:
    raise SystemExit('Reference coverage incomplete or contaminated; retained as diagnostic only')
