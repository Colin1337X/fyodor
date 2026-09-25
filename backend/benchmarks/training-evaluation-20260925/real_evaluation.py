"""Real mapped-weight evaluation with exact resume/export and memory sampling."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
model = root / 'build-train-profile/evaluated-real.gguf'
checkpoint = root / 'build-train-profile/evaluated-real.ckpt'
held = out / 'held-out.txt'
held.write_text('A separate held-out passage measures validation loss without an optimizer update.\n',encoding='utf-8')
command = [str(root/'build-cpu/fyodor-train.exe'),'--mode','cpt','--base',str(root/'.tools/tinyllama-q4_k_m.gguf'),
           '--rank','2','--context','8','--steps','1','--threads','3','--memory-mib','512',
           '--data',str(root/'backend/benchmarks/training-20260923/real-final/corpus.txt'),
           '--resume',str(root/'build-train-profile/executor-baseline.ckpt'),
           '--output',str(model),'--checkpoint',str(checkpoint),'--metrics',str(out/'real-evaluation.csv'),
           '--eval-data',str(held),'--eval-records','1','--eval-every','1']

class Counters(ctypes.Structure):
    _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
        (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
info = ctypes.WinDLL('kernel32',use_last_error=True).K32GetProcessMemoryInfo
info.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
info.restype = wintypes.BOOL
peak_working = peak_commit = samples = 0
with (out/'real-evaluation.log').open('wb') as log:
    child = subprocess.Popen(command,env=dict(os.environ,NYA_COMPUTE='cpu'),stdout=log,stderr=subprocess.STDOUT)
    while True:
        c = Counters(); c.cb = ctypes.sizeof(c)
        if info(int(child._handle),ctypes.byref(c),c.cb):
            samples += 1; peak_working = max(peak_working,c.PeakWorkingSetSize); peak_commit = max(peak_commit,c.PrivateUsage)
        try:
            code = child.wait(timeout=0.05); break
        except subprocess.TimeoutExpired:
            pass
assert code == 0

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

reference = json.loads((out.parent/'training-execution-20260925/real-resume.json').read_text())
record = {'command':command,'exit_code':code,'model_sha256':sha(model),'checkpoint_sha256':sha(checkpoint),
          'trainer_sha256':sha(Path(command[0])),'peak_working_set_bytes':peak_working,
          'sampled_peak_private_commit_bytes':peak_commit,'memory_samples':samples,
          'memory_method':'K32GetProcessMemoryInfo every 50 ms over startup, resume, evaluation, training and export. OS peak working set includes mapped/shared pages; sampled private commit is not physical RSS.'}
for key in ('model_sha256','checkpoint_sha256'):
    assert record[key] == reference[key],key
lines = (out/'real-evaluation.log').read_text().splitlines()
record['evaluations'] = [dict(pair.split('=',1) for pair in line.split()) for line in lines if line.startswith('eval_step=')]
assert [r['eval_step'] for r in record['evaluations']] == ['3','4']
(out/'real-evaluation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
assert model.resolve().is_relative_to((root/'build-train-profile').resolve())
model.unlink()
print(json.dumps(record,indent=2))
