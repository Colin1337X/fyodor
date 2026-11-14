"""Monitor validated FFN input reuse; coverage alone is not performance acceptance."""
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys

def snapshot(root, paths):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
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


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    if len(sys.argv) != 3 or sys.argv[1] not in ('short', 'long'):
        raise SystemExit('Usage: run_monitored.py short|long UNIQUE_LABEL')
    context, label = sys.argv[1:]
    if not label.isascii() or not label.isalnum():
        raise SystemExit('Label must be ASCII alphanumeric')
    folder = out / f'monitored-{context}-{label}'
    if folder.exists():
        raise SystemExit('Refusing to overwrite evidence')
    proofs = [(out / 'executables.json', 'link_passed'),
              (out / 'build.json', 'compile_passed'),
              (out / 'model-memcheck-short/result.json', 'passed'),
              (out / 'inventory-257/result.json', 'passed')]
    proofs += [(out / f'generation-{mode}-c17/result.json', 'passed') for mode in ('short','long')]
    inputs = {path for path, key in proofs}
    for path, key in proofs:
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
    env.update(PYTHONDONTWRITEBYTECODE='1', NYA_CUDA_PROFILE='0', NYA_BENCH_GPU_MONITOR='1',
        NYA_BENCH_MODE_SET='joint-base', NYA_CUDA_ALIGNED_PREFIX='1', NYA_CUDA_WIDE_BATCH_SPLIT='1',
        NYA_CUDA_WIDE_BATCH='1', NYA_CUDA_SMALL_BATCH_REUSE='1', NYA_CUDA_INTERLEAVED_REUSE='1',
        NYA_CUDA_INTERLEAVED_TWO='1', NYA_CUDA_INTERLEAVED_INTEGER='1', NYA_CUDA_BATCH_LOGITS='1',
        NYA_CUDA_ASSISTED_FFN='1', NYA_CUDA_MTP_WIDE_VALUE='1', NYA_SPECULATIVE_EARLY_REJECT='1')
    runner = out.parent / 'mtp-cost-20261113/run.py'
    command = [sys.executable, str(runner), str(folder),
               'build-cuda/interleave-pair-20261114/generation-c17.exe', context]
    inputs.update([Path(__file__).resolve(), runner, identity_path,
                   out.parent / 'gemma4-qkv-20261029/environment.py',
                   out.parent / 'gemma4-gate-20261113/model-oracle-exit.json',
                   out.parent / 'gpu-activity-20261114/cuda_identity.py'])
    inputs.update((out.parent / 'gpu-monitor-20261114').glob('*.py'))
    before = snapshot(root, sorted(inputs))
    (folder / 'input-sha256.json').write_text(json.dumps(before, indent=2))
    completed = subprocess.run(command, env=env)
    changed = changed_inputs(root, before)
    trace = (folder / f'{context}.stderr').read_text()
    selected = trace.count('MTP joint base: enabled; proposal window=3')
    evidence = [p for p in folder.iterdir() if p.is_file()]
    calls = [int(v) for v in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)',trace)]
    rows = [json.loads(line) for line in (folder/f'{context}.stdout').read_text().splitlines() if line.startswith('{"round":')]
    modes = ('ordinary','mtp-serial2','mtp-batch2','mtp-joint3')
    expected = [(r,m) for r in range(3) for m in (reversed(modes) if r==1 else modes)]
    correct = ([(r['round'],r['mode']) for r in rows]==expected and
               all(r.get('parity') is True and r.get('generated')==32 and
                   r.get('prompt')==(25 if context=='short' else 8205) for r in rows))
    result = dict(accepted_benchmark=False, model_identity=identity, integer_calls=calls, output_parity=correct, runner_exit=completed.returncode, joint_rounds=selected,
        command=command, changed_inputs=changed, inputs_unchanged=not changed,
        sha256={**before, **snapshot(root, evidence)})
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    if changed:
        raise SystemExit('Inputs changed during monitored run: ' + ', '.join(changed))
    if selected != 3 or len(calls)!=3 or min(calls)<=0 or not correct:
        raise SystemExit('Output parity or joint/integer execution inventory failed')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
