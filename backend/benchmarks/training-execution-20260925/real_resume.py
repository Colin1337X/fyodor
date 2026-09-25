"""Resume a pre-executor real-model checkpoint using three CPU workers."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
model = root / "build-train-profile/executor-resumed.gguf"
checkpoint = root / "build-train-profile/executor-resumed.ckpt"
command = [str(root / "build-cpu/fyodor-train.exe"), "--mode", "cpt", "--base", str(root / ".tools/tinyllama-q4_k_m.gguf"),
           "--rank", "2", "--context", "8", "--steps", "1", "--threads", "3", "--memory-mib", "512",
           "--data", str(root / "backend/benchmarks/training-20260923/real-final/corpus.txt"),
           "--resume", str(root / "build-train-profile/executor-baseline.ckpt"), "--output", str(model),
           "--checkpoint", str(checkpoint), "--metrics", str(out / "real-resume.csv")]

class Counters(ctypes.Structure):
    _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)] + [
        (name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]

info = ctypes.WinDLL("kernel32",use_last_error=True).K32GetProcessMemoryInfo
info.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
info.restype = wintypes.BOOL
peak_working = peak_commit = samples = 0
with (out / "real-resume.log").open("wb") as log:
    child = subprocess.Popen(command,env=dict(os.environ,NYA_COMPUTE="cpu"),stdout=log,stderr=subprocess.STDOUT)
    while True:
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        if info(int(child._handle),ctypes.byref(counters),counters.cb):
            samples += 1
            peak_working = max(peak_working,counters.PeakWorkingSetSize)
            peak_commit = max(peak_commit,counters.PrivateUsage)
        try:
            code = child.wait(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            pass
assert code == 0

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

record = {"command":command,"exit_code":code,"model_sha256":sha(model),"checkpoint_sha256":sha(checkpoint),
          "peak_working_set_bytes":peak_working,"sampled_peak_private_commit_bytes":peak_commit,"samples":samples,
          "memory_method":"K32GetProcessMemoryInfo polled every 50 ms; observed OS peak working set includes mapped/shared pages. Private commit is sampled, not physical RSS. Includes initialization, resume, training and export."}
reference = json.loads((out / "real-final/metadata.json").read_text(encoding="utf-8"))["runs"][0]
for key in ("model_sha256","checkpoint_sha256"):
    assert record[key] == reference[key], key
(out / "real-resume.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
assert model.resolve().is_relative_to((root / "build-train-profile").resolve())
model.unlink()
print(json.dumps(record,indent=2))
