"""Monitor validated QKV reuse; coverage alone is not performance acceptance."""
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys

def snapshot(root, paths):
    return {(str(p.relative_to(root)) if p.is_relative_to(root) else str(p)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def changed_inputs(root, hashes):
    changed = []
    for name, expected in hashes.items():
        try:
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        except OSError:
            actual = None
        if actual != expected:
            changed.append(name)
    return changed


def graph_inventory(trace):
    prefix = 'CUDA batched graph captures:'
    lines = [line for line in trace.splitlines() if prefix in line]
    matches = [re.fullmatch(re.escape(prefix) + r' (\d+) replays: (\d+)', line)
               for line in lines]
    counts = [(int(m[1]), int(m[2])) for m in matches if m]
    valid = (len(lines) == 6 and all(matches) and
             all(captures > 0 and replays > captures for captures, replays in counts))
    return counts, valid



def reuse_inventory(trace, enabled, prefix='CUDA QKV input reuse capture/direct hits:', expected_count=3):
    lines = [line for line in trace.splitlines() if prefix in line]
    matches = [re.fullmatch(re.escape(prefix) + r' (\d+)', line) for line in lines]
    hits = [int(match[1]) for match in matches if match]
    valid = len(hits) == expected_count and all(v > 0 for v in hits) if enabled else not hits
    return hits, valid and all(matches)


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    diagnostic = Path(__file__).resolve().parent
    if len(sys.argv) not in (3, 4) or sys.argv[1] != 'long':
        raise SystemExit('Usage: run_monitored_v2.py long UNIQUE_LABEL [enabled|disabled]')
    context, label = sys.argv[1:3]
    variant = sys.argv[3] if len(sys.argv) == 4 else 'enabled'
    if variant not in ('enabled', 'disabled'):
        raise SystemExit('Choose enabled or disabled paired-row dispatch')
    enabled = variant == 'enabled'
    if not label.isascii() or not label.isalnum():
        raise SystemExit('Label must be ASCII alphanumeric')
    folder = diagnostic / f'monitored-{context}-{label}'
    if folder.exists():
        raise SystemExit('Refusing to overwrite evidence')
    proofs = [(out / 'build.json', 'build_passed'),
              (out / 'control.json', 'passed'),
              (out / 'lifecycle-fallback-c23-257/result.json', 'passed'),
              (out / 'model-cached-memcheck-c23-8192/result.json', 'passed'),
              (out / 'lifecycle-fallback-c17-257/result.json', 'passed'),
              (out / 'lifecycle-c17-8192/result.json', 'passed')]
    proofs += [(out / f'generation-{context}-c17/result.json', 'passed'),
               (out / 'generation-short-c23/result.json', 'passed')]
    proofs += [(out.parent/'batch-graphs-20261114'/f'boundaries-{standard}-{width}-ring{ring}/result.json','passed')
               for standard in ('c17','c23') for width in (256,512) for ring in (0,1)]
    inputs = {path for path, key in proofs}
    for path, key in proofs:
        if not path.exists():
            raise SystemExit(f'Wait for completed prerequisite: {path}')
        proof = json.loads(path.read_text())
        if proof.get(key) is not True:
            raise SystemExit(f'Prerequisite failed: {path}')
        for name, expected in proof['sha256'].items():
            inputs.add(root / name)
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise SystemExit(f'Prerequisite input changed: {name}')
    identity_path = out.parent / 'mtp-joint-base-v2-20261114/model_identity.py'
    sys.path.insert(0,str(identity_path.parent))
    from model_identity import verify_gemma
    identity = verify_gemma()
    spec = importlib.util.spec_from_file_location('gate', out.parent / 'gemma4-qkv-20261029/environment.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
        raise SystemExit('Another GPU workload remains')
    folder.mkdir()
    preflight = folder / 'preflight'
    preflight.mkdir()
    gate.configure(preflight)
    gate.idle()
    env = {k: v for k, v in os.environ.items() if not k.startswith('NYA_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', NYA_CUDA_PROFILE='0', NYA_CUDA_BATCH_GRAPHS='1', NYA_BENCH_GPU_MONITOR='1',
        NYA_BENCH_MODE_SET='joint-base', NYA_CUDA_QKV_INPUT_REUSE='1', NYA_CUDA_PAIR_ROWS='1', NYA_CUDA_Q6_PAIR_ROWS='1', NYA_CUDA_CACHED_QUERY='1' if enabled else '0', NYA_CUDA_ALIGNED_PREFIX='1', NYA_CUDA_WIDE_BATCH_SPLIT='1',
        NYA_CUDA_WIDE_BATCH='1', NYA_CUDA_SMALL_BATCH_REUSE='1', NYA_CUDA_INTERLEAVED_REUSE='1',
        NYA_CUDA_INTERLEAVED_TWO='1', NYA_CUDA_INTERLEAVED_INTEGER='1', NYA_CUDA_BATCH_LOGITS='1',
        NYA_CUDA_ASSISTED_FFN='1', NYA_CUDA_MTP_WIDE_VALUE='1', NYA_SPECULATIVE_EARLY_REJECT='1')
    runner = out.parent / 'pdh-recovery-20261115/run_cost.py'
    command = [sys.executable, str(runner), str(folder),
               'build-cuda/model-attention-query-cache-v2-20261115/generation-c17.exe', context]
    inputs.update([Path(__file__).resolve(), runner, identity_path,
                   out.parent / 'gemma4-qkv-20261029/environment.py',
                   out.parent / 'gemma4-gate-20261113/model-oracle-exit.json',
                   out.parent / 'gpu-activity-20261114/cuda_identity.py'])
    inputs.update((out.parent / 'gpu-monitor-20261114').glob('*.py'))
    inputs.update(diagnostic.glob('*.py'))
    inputs.update((out.parent / 'pdh-recovery-20261115').glob('*.py'))
    before = snapshot(root, sorted(inputs))
    (folder / 'input-sha256.json').write_text(json.dumps(before, indent=2))
    completed = subprocess.run(command, env=env)
    # The monitor can reject cooperative completion after retained recovery errors.
    # Still audit the finished trace with the unchanged conservative auditor.
    audit_exit = None
    manifest = folder / 'gpu-monitor.json'
    if manifest.exists() and (folder / 'timing.json').exists() and not (folder / 'gpu-coverage.json').exists():
        metadata = json.loads(manifest.read_text())
        if metadata.get('adapter', {}).get('counter_luid'):
            audit_command = [sys.executable, str(out.parent / 'gpu-monitor-20261114/audit_generation.py'),
                str(folder), '--samples', str(folder / 'gpu-pdh.jsonl'),
                '--luid', metadata['adapter']['counter_luid'], '--output', str(folder / 'gpu-coverage.json'),
                '--mode-set', 'joint-base']
            audit_exit = subprocess.run(audit_command, env=env).returncode
    changed = changed_inputs(root, before)
    trace_path = folder / f'{context}.stderr'
    trace = trace_path.read_text() if trace_path.exists() else ''
    metadata_selected = 'CUDA native metadata Q4: selected' in trace
    graph_counts, graphs_reused = graph_inventory(trace)
    hits, reuse_valid = reuse_inventory(trace, True)
    pair_calls, pair_valid = reuse_inventory(trace, True, 'CUDA paired rows capture/direct calls:')
    q6_calls, q6_valid = reuse_inventory(trace, True, 'CUDA Q6 paired rows capture/direct calls:')
    cached_query_calls, cached_query_valid = reuse_inventory(trace, enabled, 'CUDA cached query capture/direct calls:', expected_count=6)
    selected = trace.count('MTP joint base: enabled; proposal window=3')
    evidence = [p for p in folder.iterdir() if p.is_file()]
    calls = [int(v) for v in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)',trace)]
    stdout_path=folder/f'{context}.stdout'
    rows = [json.loads(line) for line in (stdout_path.read_text() if stdout_path.exists() else '').splitlines() if line.startswith('{"round":')]
    modes = ('ordinary','mtp-serial2','mtp-batch2','mtp-joint3')
    expected = [(r,m) for r in range(3) for m in (reversed(modes) if r==1 else modes)]
    correct = ([(r['round'],r['mode']) for r in rows]==expected and
               all(r.get('parity') is True and r.get('generated')==32 and
                   r.get('prompt')==(25 if context=='short' else 8205) for r in rows))
    result = dict(accepted_benchmark=False, diagnostic_recovery=True, recovery_audit_exit=audit_exit, metadata_selected=metadata_selected, model_identity=identity, integer_calls=calls, qkv_reuse_hits=hits, qkv_reuse_enabled=True, pair_rows_enabled=True, q6_pair_rows_enabled=True, cached_query_enabled=enabled, cached_query_calls=cached_query_calls, cached_query_inventory_valid=cached_query_valid, q6_pair_rows_calls=q6_calls, q6_inventory_valid=q6_valid, pair_rows_calls=pair_calls, pair_inventory_valid=pair_valid, qkv_reuse_inventory_valid=reuse_valid, output_parity=correct, runner_exit=completed.returncode, joint_rounds=selected,
        command=command, environment_overrides={k:v for k,v in env.items() if k.startswith('NYA_')}, changed_inputs=changed, inputs_unchanged=not changed, graph_counts=graph_counts, graphs_reused=graphs_reused,
        sha256={**before, **snapshot(root, evidence)})
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    if changed:
        raise SystemExit('Inputs changed during monitored run: ' + ', '.join(changed))
    if not cached_query_valid or not q6_valid or not pair_valid or not reuse_valid or not graphs_reused or not metadata_selected or selected != 3 or len(calls)!=3 or min(calls)<=0 or not correct:
        raise SystemExit('Output parity or joint/integer execution inventory failed')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
