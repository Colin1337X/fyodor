"""Run only after validation exits: paired CUDA device initialization.

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
        consecutive=consecutive+1 if 0<=utilization<=12 else 0
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

env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_','CUDA_CACHE_'))}
env.update(NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0',NYA_COMPUTE='cuda')
executables={'before':root/'.tools/fyodor-device-before-module-split.exe','after':root/'build-cuda/nya-bench-training-device.exe'}
meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'binary_sha256':{k:sha(v) for k,v in executables.items()},'source_sha256':{p:sha(root/p) for p in ['CUDA/CMakeLists.txt','CUDA/cuda.c','CUDA/training.inc','backend/benchmarks/training_device_bench.c']},
      'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'blocks':[]}
(out/'processes-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
for cache in ['disabled','enabled']:
    env['CUDA_CACHE_DISABLE']='1' if cache=='disabled' else '0'
    for mode in ['inference','training']:
        for order in [('before','after'),('after','before')]:
            block={'cache':cache,'mode':mode,'order':order,'before':idle(),'runs':[]}
            meta['blocks'].append(block)
            for repeat in range(7):
                for version in order:
                    command=[str(executables[version]),'--startup',mode]
                    rows=[]; stop=threading.Event()
                    p=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    sampler=threading.Thread(target=sample,args=(p,stop,rows));sampler.start()
                    stdout,stderr=p.communicate(); stop.set(); sampler.join()
                    record={'version':version,'repeat':repeat,'warmup':repeat<2,'exit_code':p.returncode,
                            'stdout':stdout.decode(),'stderr':stderr.decode(),'resources':rows}
                    block['runs'].append(record)
                    (out/'measurements.json').write_text(json.dumps(meta,indent=2))
                    if p.returncode: raise RuntimeError(f'{mode} {version} failed')
                    record['result']=json.loads(record['stdout'])
                    if any(r.get('competing_compute_processes') for r in rows): raise RuntimeError('New GPU workload; reject attempt')
            block['after']=idle()
            (out/'measurements.json').write_text(json.dumps(meta,indent=2))
            print(cache,mode,order,'passed',flush=True)
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
import statistics
groups={}
for b in meta['blocks']:
    for r in b['runs']:
        if not r['warmup']:
            groups.setdefault((b['cache'],b['mode'],r['version']),[]).append(r['result']['create_ms'])
summary=[{'cache':k[0],'mode':k[1],'version':k[2],'samples':len(v),'mean_ms':statistics.mean(v),'stddev_ms':statistics.stdev(v)} for k,v in groups.items()]
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
