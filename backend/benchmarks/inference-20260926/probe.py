"""Exploratory two-chain dot-product probe, not performance acceptance.

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
env.update(NYA_CPU_THREADS='6',NYA_CPU_BATCH='512',NYA_CUDA_BATCH='512',NYA_CUDA_BLAS='1',NYA_CUDA_CUTLASS='0',NYA_CUDA_BLAS_LIBRARY=str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'))
model=root/'.tools/tinyllama-q4_k_m.gguf'
executables={'before':root/'.tools/fyodor-bench-before-dot-ilp.exe','after':root/'build-cuda/fyodor-bench.exe','llama':root/'.tools/llama-b10809-cuda/fyodor-llama-capi-bench.exe'}
meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'model_sha256':sha(model),'llama_commit':'5266f24da75dc449bd56cbed7addb9c8e4a6a73e','harness_sha256':sha(root/'backend/benchmarks/llama_capi_bench.c'),'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'source_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'CUDA/matvec.cu',root/'CUDA/resident.cu',root/'CUDA/resident.inc',root/'CUDA/cuda.c')},'binary_sha256':{name:sha(p) for name,p in executables.items()},'dependency_sha256':{str(p.relative_to(root)):sha(p) for p in [Path(env['NYA_CUDA_BLAS_LIBRARY']),*(executables['llama'].parent/name for name in ('llama.dll','ggml-base.dll','ggml-cuda.dll'))]},'caveats':['Windows interactive desktop; GPU counters identified DWM and ChatGPT as active at idle. Other apps remain open. Not a headless measurement.','Require two consecutive 7..12% whole-GPU utilization readings before and after each process. Preserve failure, never discard individual timed samples.','F32 KV, deterministic identical tokens, 512 batch/ubatch, six threads, FA off in llama C API harness; intermediate math differs.','Model loading, tokenization, sampling, prefix preparation are untimed. Host logits read every decoded token.','Resource sampling includes startup and model load; whole-device readings include other apps. Driver graph storage is not in explicit tensor byte totals.'],'runs':[]}
(out/'processes-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
meta['observed_desktop_processes']=desktop_processes
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
for prefix in (0,512,1024):
    engines=['before','after']
    for phase,order in [('a',engines),('b',list(reversed(engines)))]:
        for engine in order:
            name=f'{prefix}-{engine}-{phase}'
            cmd=[str(executables[engine]),'-m',str(model),'-b','cuda','-p','512' if not prefix else '0','-n','128','--context',str(prefix),'-r','5','--warmup','2','--json']
            if engine=='llama':cmd+=['--batch','512','--threads','6']
            row={'name':name,'command':cmd,'gpu_idle_before':idle()}
            meta['runs'].append(row)
            process=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
            stop=threading.Event(); samples=[]
            sampler=threading.Thread(target=sample,args=(process,stop,samples));sampler.start()
            try:
                stdout,stderr=process.communicate(timeout=240)
            except subprocess.TimeoutExpired:
                process.kill();stdout,stderr=process.communicate()
                row['timeout']=True
            finally:
                stop.set();sampler.join()
            row.update(exit_code=process.returncode,gpu_after=gpu())
            (out/(name+'.json')).write_bytes(stdout);(out/(name+'.log')).write_bytes(stderr)
            (out/(name+'-resources.json')).write_text(json.dumps(samples,indent=2))
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            if process.returncode:raise RuntimeError(row)
            if any(s.get('competing_compute_processes') for s in samples):
                row['contaminated']=True
                (out/'metadata.json').write_text(json.dumps(meta,indent=2))
                raise RuntimeError('Competing CUDA process during '+name)
            data=json.loads(stdout)
            if data['kv_type']!='f32' or data['backend']!='cuda' or (engine!='llama' and data['execution']!='resident'):raise RuntimeError(data)
            if engine=='llama':
                offload=re.search(rb'offloaded (\d+)/(\d+) layers to GPU',stderr)
                if not offload or offload[1]!=offload[2]:raise RuntimeError('Upstream full offload unconfirmed')
            try:row['gpu_idle_after']=idle()
            except RuntimeError as error:
                row['idle_failure']=str(error)
                (out/'metadata.json').write_text(json.dumps(meta,indent=2))
                raise
            (out/'metadata.json').write_text(json.dumps(meta,indent=2))
            print(name,[(r['test'],r['tokens_per_second'],r['stddev']) for r in data['results']],flush=True)
(out/'processes-after.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
