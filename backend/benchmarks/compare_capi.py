"""Serial, reversed-order comparisons with matched F32 KV and token IDs.

The external executable is built from llama_capi_bench.c and matching upstream
headers/libraries. This script never integrates upstream execution into Fyodor.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gpu():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,memory.used,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True,
                           timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fyodor', required=True, type=Path)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--llama-capi', required=True, type=Path)
    parser.add_argument('--llama-commit', required=True)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--prompt', type=int, default=512)
    parser.add_argument('--generate', type=int, default=128)
    parser.add_argument('--context', type=int, default=0)
    args = parser.parse_args()
    if (not 1 <= args.repetitions <= 1000 or not 0 <= args.warmup <= 1000 or
            min(args.threads, args.batch) <= 0 or min(args.prompt, args.generate, args.context) < 0 or
            not args.prompt + args.generate):
        parser.error('invalid workload size')
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'NYA_CPU_THREADS': str(args.threads),
           'NYA_CUDA_BATCH': str(args.batch), 'NYA_CPU_BATCH': str(args.batch)}
    common = ['-m', str(args.model.resolve()), '-b', 'cuda', '-p', str(args.prompt),
              '-n', str(args.generate), '--context', str(args.context), '-r', str(args.repetitions),
              '--warmup', str(args.warmup), '--json']
    engines = ([('before', args.before)] if args.before else []) + [
        ('fyodor', args.fyodor), ('llama-capi', args.llama_capi)]
    metadata = {
        'machine': platform.platform(), 'model_sha256': sha(args.model),
        'llama_commit': args.llama_commit,
        'harness_sha256': sha(Path(__file__).with_name('llama_capi_bench.c')),
        'environment': {k: v for k, v in env.items() if k.startswith(('NYA_', 'GGML_', 'CUDA_'))},
        'caveats': ['Both engines use F32 KV, identical deterministic token IDs and warmup workloads.',
                    'No tokenizer, sampling, model loading or prefix preparation is timed.',
                    'Prefill returns final-position logits; decode returns host logits every token.',
                    'llama.cpp may round cache capacity to its allocator alignment; see allocated_context.',
                    'Numerical implementations and intermediate precision remain engine-specific.',
                    'Whole-device telemetry includes other applications. Contention is not isolated.',
                    'This is the upstream C API harness, not the official llama-bench executable.'],
        'runs': []}
    dependencies = [Path(env[k]) for k in ('NYA_CUDA_CUTLASS_LIBRARY', 'NYA_CUDA_BLAS_LIBRARY') if env.get(k)]
    dependencies += [args.llama_capi.resolve().parent / name for name in ('llama.dll', 'ggml-base.dll', 'ggml-cuda.dll')]
    metadata['dependency_sha256'] = {str(p.resolve()): sha(p) for p in dependencies if p.is_file()}
    for phase, order in [('a', engines), ('b', list(reversed(engines)))]:
        for name, exe in order:
            command = [str(exe.resolve()), *common]
            if name == 'llama-capi':
                command += ['--batch', str(args.batch), '--threads', str(args.threads)]
            run = {'name': f'{name}-{phase}', 'command': command, 'binary_sha256': sha(exe),
                   'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'gpu_before': gpu()}
            p = subprocess.run(command, env=env, capture_output=True,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            run.update(exit_code=p.returncode, gpu_after=gpu())
            (args.output / f'{name}-{phase}.json').write_bytes(p.stdout)
            (args.output / f'{name}-{phase}.log').write_bytes(p.stderr)
            metadata['runs'].append(run)
            (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            if p.returncode:
                raise RuntimeError(f'{name} failed; inspect saved stderr')
            result = json.loads(p.stdout)
            if result['kv_type'] != 'f32' or result['backend'] != 'cuda':
                raise RuntimeError('backend/precision mismatch; discard comparison')
            if name != 'llama-capi' and result['execution'] != 'resident':
                raise RuntimeError('Fyodor left the resident path; discard comparison')
            if name == 'llama-capi':
                offload = re.search(rb'offloaded (\d+)/(\d+) layers to GPU', p.stderr)
                if not offload or offload[1] != offload[2]:
                    raise RuntimeError('upstream full GPU offload not confirmed; discard comparison')
            print(run['name'], [(r['test'], r['tokens_per_second'], r['stddev'])
                                for r in result['results']], flush=True)


if __name__ == '__main__':
    main()
