"""Validate isolated batched graphs with reset/rewind and one-bank readback."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path.cwd()
def key(path):return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
out = Path(__file__).resolve().parent
if not __debug__:raise SystemExit('Lifecycle assertions required')
pending=out/'generation-long-c17/result.json'
if not pending.exists():raise SystemExit('Wait for completed cached-query long generation before filtered memcheck')
if json.loads(pending.read_text()).get('passed') is not True:raise ValueError('Long generation failed')
if len(sys.argv) != 3 or sys.argv[1] not in ('c17', 'c23') or sys.argv[2] not in ('257', '8192'):
    raise SystemExit('Usage: run_lifecycle.py c17|c23 257|8192')
standard, prefix = sys.argv[1:]
if prefix != "8192": raise SystemExit("Filtered candidate memcheck requires prefix 8192")
folder = out / f'model-cached-memcheck-{standard}-{prefix}'
if folder.exists():
    raise SystemExit('Refusing to overwrite evidence')
proof_path = out / 'build.json'
proof = json.loads(proof_path.read_text())
assert proof['build_passed'] is True
inputs = {root / name for name in proof['sha256']}
inputs.update((proof_path, Path(__file__).resolve(), pending))
generation=json.loads(pending.read_text())
if not generation.get('candidate_exercised') or len(generation.get('cached_query_calls',[]))!=6:
    raise ValueError('Long generation lacks cached-query execution evidence')
for name,expected in generation['sha256'].items():
    path=root/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Generation input changed: '+name)
    inputs.add(path)

control_path=out/'control.json';control=json.loads(control_path.read_text())
if control.get('passed') is not True:raise ValueError('Control checks failed')
inputs.add(control_path)
for name,h in control['sha256'].items():
    path=root/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError(name)
    inputs.add(path)
for name, expected in proof['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(out.parent / 'mtp-joint-base-v2-20261114'))
from model_identity import verify_gemma
identity = verify_gemma()
sys.path.insert(0, str(out.parent / 'gemma4-qkv-20261029'))
import environment
initial = environment.snapshot()
if any(not environment.desktop_identity_allowed(p) for p in initial['processes']):
    raise SystemExit('Another GPU workload remains')
settings_path = out.parent / 'integer-launch-20261114/verify-prefix257-ring1-count4.json'
settings = json.loads(settings_path.read_text())['environment']
inputs.add(settings_path)
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
inputs.add(sanitizer)
hashes = {key(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
settings = dict(settings, NYA_CUDA_BATCH_GRAPHS='1', NYA_CUDA_QKV_INPUT_REUSE='1', NYA_CUDA_PAIR_ROWS='1', NYA_CUDA_Q6_PAIR_ROWS='1', NYA_CUDA_CACHED_QUERY='1')
env.update(settings)
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['PATH'] = str(root / '.tools/w64devkit/bin') + os.pathsep + env['PATH']
exe = root / f'build-cuda/model-attention-query-cache-v2-20261115/lifecycle-{standard}.exe'
command = [str(sanitizer), '--tool', 'memcheck', '--kernel-name', 'regex=nya_attention_cached_query', '--error-exitcode', '99', str(exe), 'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf', prefix]
folder.mkdir()
with (folder / 'stdout').open('x') as stdout, (folder / 'stderr').open('x') as stderr:
    child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
    (folder/'running.json').write_text(json.dumps(dict(process_id=child.pid,command=command),indent=2))
    try:child.wait()
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=15)
trace = (folder / 'stderr').read_text()
calls = [int(x) for x in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)', trace)]
changed = [name for name, expected in hashes.items()
           if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
selected = "CUDA native metadata Q4: selected" in trace
graphs = [(int(a),int(b)) for a,b in re.findall(r'CUDA batched graph captures: (\d+) replays: (\d+)',trace)]
graph_reused = any(captures > 0 and replays > captures for captures,replays in graphs)
hits=[int(v) for v in re.findall(r'CUDA QKV input reuse capture/direct hits: (\d+)',trace)]
rows=re.findall(r'^reset=(\d+) prefix=(\d+) tokens=(\d+) max_scaled=([\deE.+-]+) execution=(\S+)$',(folder/'stdout').read_text(),re.M)
case_inventory=[(r,p,t,e) for r,p,t,error,e in rows]==[(str(r),prefix,'4','resident') for r in range(2)]
numerics=len(rows)==2 and all(0<=float(error)<=0.0005 for r,p,t,error,e in rows)
pair_calls=[int(v) for v in re.findall(r'CUDA paired rows capture/direct calls: (\d+)',trace)]
q6_lines=[line for line in trace.splitlines() if 'CUDA Q6 paired rows capture/direct calls:' in line]
q6_matches=[re.fullmatch(r'CUDA Q6 paired rows capture/direct calls: (\d+)',line) for line in q6_lines]
q6_calls=[int(m[1]) for m in q6_matches if m]
q6_valid=len(q6_lines)==1 and len(q6_calls)==1 and min(q6_calls)>0
cached_lines=[line for line in trace.splitlines() if 'CUDA cached query capture/direct calls:' in line]
cached_matches=[re.fullmatch(r'CUDA cached query capture/direct calls: (\d+)',line) for line in cached_lines]
cached_calls=[int(m[1]) for m in cached_matches if m]
cached_valid=len(cached_lines)==1 and len(cached_calls)==1 and min(cached_calls)>0
logs=(folder/'stdout').read_text()+trace
summary=re.findall(r'^========= ERROR SUMMARY:.*$',logs,re.M)
memcheck_ok=summary==['========= ERROR SUMMARY: 0 errors']
passed = memcheck_ok and cached_valid and q6_valid and bool(pair_calls) and min(pair_calls)>0 and case_inventory and numerics and bool(hits) and min(hits)>0 and selected and graph_reused and child.returncode == 0 and bool(calls) and min(calls) > 0 and not changed
outputs = {key(p): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in folder.iterdir() if p.is_file()}
(folder / 'result.json').write_text(json.dumps(dict(passed=passed,
    accepted_benchmark=False, sanitizer_scope='Only nya_attention_cached_query kernels; all other kernels excluded', memcheck_passed=memcheck_ok, cached_query_calls=cached_calls, cached_query_inventory_valid=cached_valid, single_download_assertions_passed=passed, command=command,
    reset_inventory_valid=case_inventory, numerical_summary_valid=numerics, process_id=child.pid,
    environment=settings, initial=initial, model_identity=identity, graph_counts=graphs, graph_reused=graph_reused,
    exit_code=child.returncode, metadata_selected=selected, integer_calls=calls, qkv_reuse_hits=hits, pair_rows_calls=pair_calls, q6_pair_rows_calls=q6_calls, q6_inventory_valid=q6_valid, changed_inputs=changed,
    sha256={**hashes, **outputs},
    note='Filtered memcheck, not full-model sanitizer coverage. Original lifecycle/numerical guards plus one graph replay and one download per four-row verification; no throughput acceptance.'), indent=2))
print('Lifecycle', standard, prefix, 'passed' if passed else 'FAILED', flush=True)
raise SystemExit(0 if passed else 1)
