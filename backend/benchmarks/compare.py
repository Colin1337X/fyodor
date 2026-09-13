"""Run both independent engines serially on one GGUF; preserve raw evidence.

Python is tooling only, never a dependency of the C engine or benchmark binary.
This helper neither downloads models nor sends model content over the network.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import threading
import time
import shutil


def sample_resources(process, stop, samples):
    """Best-effort process RAM and whole-device telemetry, including startup.

    Telemetry is separate from the engine's timed sections. GPU figures include
    other applications; they are never labeled as this process's allocation.
    Missing platform APIs are reported as unavailable, not zero consumption.
    """
    reader = None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set',
                'peak_paged_pool', 'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool',
                'pagefile', 'peak_pagefile', 'private_bytes')]
        psapi = ctypes.WinDLL('psapi')
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        def reader():
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(int(process._handle), ctypes.byref(counters), counters.cb):
                return {key: getattr(counters, key) for key in ('peak_working_set', 'working_set', 'private_bytes')}
            return None
    smi = shutil.which('nvidia-smi')
    started = time.monotonic()
    while not stop.is_set():
        row = {'seconds_since_launch': time.monotonic() - started, 'process_ram_bytes': reader() if reader else None}
        if smi:
            try:
                query = subprocess.run([smi, '--query-gpu=index,utilization.gpu,memory.used,memory.free,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if query.returncode == 0:
                    row['whole_device_index_util_percent_used_mib_free_mib_pstate_sm_mhz_mem_mhz_watts_celsius'] = query.stdout.strip().splitlines()
            except (OSError, subprocess.TimeoutExpired):
                pass
        samples.append(row)
        stop.wait(1)

p = argparse.ArgumentParser()
p.add_argument('--fyodor', required=True)
p.add_argument('--llama', required=True)
p.add_argument('--model', required=True)
p.add_argument('--backend', choices=('cpu', 'cuda', 'vulkan'), default='cpu')
p.add_argument('--threads', type=int, default=6)
p.add_argument('--prompt', type=int, default=512)
p.add_argument('--generate', type=int, default=128)
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--warmup', type=int, default=1, help='Fyodor warmup repetitions; llama-bench uses its own warmup policy')
p.add_argument('--context', type=int, default=0)
p.add_argument('--batch', type=int, default=512)
p.add_argument('--output', required=True)
a = p.parse_args()
root = pathlib.Path(a.output)
root.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env['NYA_CPU_THREADS'] = str(a.threads)
env['NYA_CUDA_BATCH'] = str(a.batch)
env['NYA_CPU_BATCH'] = str(a.batch)
commands = {
    'fyodor': [a.fyodor, '-m', a.model, '-b', a.backend, '-p', str(a.prompt), '-n', str(a.generate),
               '-r', str(a.repetitions), '--context', str(a.context), '--warmup', str(a.warmup), '--json'],
    'llama': [a.llama, '-m', a.model, '-p', str(a.prompt), '-n', str(a.generate), '-r', str(a.repetitions),
              '-d', str(a.context), '-t', str(a.threads), '-b', str(a.batch), '-ub', str(a.batch),
              '-ngl', '0' if a.backend == 'cpu' else '99', '-ctk', 'f16', '-ctv', 'f16', '-fa', 'off', '-o', 'json'],
}
digest = hashlib.sha256()
with open(a.model, 'rb') as model:
    for chunk in iter(lambda: model.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
metadata = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), platform=platform.platform(),
                model_sha256=digest.hexdigest(), commands=commands,
                environment={k: v for k, v in env.items() if k.startswith(('NYA_CPU_', 'NYA_CUDA_', 'NYA_INFERENCE_'))},
                caveats=['Fyodor KV is F32; llama-bench b10809 rejects F32 KV, so it uses F16.',
                         'Both use synthetic token IDs, but their deterministic token sequences differ.',
                         'llama warmup policy differs; use enough repetitions and report variance.',
                         'Select a llama build containing the requested backend; inspect its reported backend.'])
(root / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
for name, command in commands.items():
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    stop = threading.Event()
    samples = []
    sampler = threading.Thread(target=sample_resources, args=(process, stop, samples), daemon=True)
    sampler.start()
    try:
        stdout, stderr = process.communicate()
    finally:
        stop.set()
        sampler.join()
    (root / (name + '.json')).write_bytes(stdout)
    (root / (name + '.log')).write_bytes(stderr)
    (root / (name + '-resources.json')).write_text(json.dumps(samples, indent=2), encoding='utf-8')
    if process.returncode:
        raise SystemExit(f'{name} failed ({process.returncode}); inspect {root}/{name}.log')
    json.loads(stdout)
print(root)
