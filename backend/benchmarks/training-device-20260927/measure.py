"""Run only after validation exits: paired F32-KV workloads on an idle desktop.

Pass a new directory name for each attempt; raw measurements are never replaced.
"""
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent / sys.argv[1]
out.mkdir(exist_ok=False)

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv,noheader'], text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW).strip()

def gpu_processes():
    xml=subprocess.check_output(['nvidia-smi','-q','-x'],timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    return [{'pid':p.findtext('pid'),'type':p.findtext('type'),'name':p.findtext('process_name')} for p in ET.fromstring(xml).findall('.//process_info')]

# WDDM's query-compute-apps includes graphics-only desktop processes. Keep the
# observed desktop identities, reject pure compute or newly appearing GPU apps.
desktop_processes=gpu_processes()
desktop_identities={(p['pid'],p['name']) for p in desktop_processes if p['type']=='C+G'}
def compute_processes(own_pid=None):
    return [p for p in gpu_processes() if p['pid']!=str(own_pid) and (p['type']=='C' or (p['pid'],p['name']) not in desktop_identities)]

def idle():
    time.sleep(2)
    readings=[]
    consecutive=0
    for _ in range(20):
        state=gpu(); readings.append(state)
        utilization=int(state.split('%')[0].strip())
        consecutive=consecutive+1 if 7<=utilization<=12 else 0
        if consecutive==2:
            competitors=compute_processes()
            if competitors:
                (out/'idle-failure.json').write_text(json.dumps({'readings':readings,'competing_compute_processes':competitors},indent=2))
                raise RuntimeError('Competing CUDA process: '+str(competitors))
            return readings
        time.sleep(1)
    (out/'idle-failure.json').write_text(json.dumps({'readings':readings},indent=2))
    raise RuntimeError('Desktop idle gate failed: '+str(readings))

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(key,ctypes.c_size_t) for key in ('peak_working_set','working_set','peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool','pagefile','peak_pagefile','private_bytes')]

psapi=ctypes.WinDLL('psapi')
psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype=wintypes.BOOL

def sample(process,stop,rows):
    start=time.monotonic()
    while not stop.is_set():
        c=Counters();c.cb=ctypes.sizeof(c)
        row={'elapsed_seconds':time.monotonic()-start}
        if psapi.GetProcessMemoryInfo(int(process._handle),ctypes.byref(c),c.cb):
            row.update({key:getattr(c,key) for key in ('peak_working_set','working_set','private_bytes')})
        try:
            row['whole_gpu']=gpu()
            row['competing_compute_processes']=compute_processes(process.pid)
        except (OSError,subprocess.SubprocessError):row['whole_gpu']=None
        rows.append(row)
        stop.wait(1)

env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0',NYA_COMPUTE='cpu')
exe=root/'build-cuda/nya-bench-training-device.exe'
model=root/'.tools/tinyllama-q4_k_m.gguf'
meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'model_sha256':sha(model),'binary_sha256':sha(exe),'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},
      'source_sha256':{p:sha(root/p) for p in ['CUDA/training.cu','CUDA/training.inc','CUDA/gemm.cu','CUDA/cuda.c','backend/core/training_device.c','backend/benchmarks/training_device_bench.c']},
      'runs':[]}
(out/'processes-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
for order in ['forward','reverse']:
    before=idle()
    command=[str(exe),str(model),order]
    rows=[]; stop=threading.Event()
    with (out/f'{order}.out').open('wb') as stdout, (out/f'{order}.log').open('wb') as stderr:
        p=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
        sampler=threading.Thread(target=sample,args=(p,stop,rows));sampler.start()
        code=p.wait(); stop.set(); sampler.join()
    record={'order':order,'command':command,'exit_code':code,'before':before,'resources':rows}
    meta['runs'].append(record)
    (out/'measurements.json').write_text(json.dumps(meta,indent=2))
    if code: raise RuntimeError(f'{order} failed: {code}')
    if any(r.get('competing_compute_processes') for r in rows): raise RuntimeError('New GPU workload; reject complete attempt')
    record['after']=idle()
    (out/'measurements.json').write_text(json.dumps(meta,indent=2))
    print(order,'passed',flush=True)
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
