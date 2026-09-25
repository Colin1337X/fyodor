"""Real-model accumulation, process restart, and Windows peak RAM evidence.
This validates execution and resume, not useful language-model convergence.
"""
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
work = root / "build-train-profile/real-accumulation"
work.mkdir(exist_ok=False)
corpus = evidence / "accumulation-corpus.txt"
corpus.write_text("The cat sat on the mat. The dog ran away.\n", encoding="utf-8")
env = dict(os.environ, NYA_COMPUTE="cpu")
env.pop("NYA_TRAIN_PROFILE", None)
base = root / ".tools/tinyllama-q4_k_m.gguf"
exe = root / "build-cpu/fyodor-train.exe"


class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
        "PeakPagefileUsage", "PrivateUsage")]


memory_info = ctypes.WinDLL("kernel32", use_last_error=True).K32GetProcessMemoryInfo
memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
memory_info.restype = wintypes.BOOL


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


record = {"base_sha256": sha(base), "trainer_sha256": sha(exe), "corpus_sha256": sha(corpus),
          "memory_method": "Windows K32GetProcessMemoryInfo, polled every 50 ms. OS peak working set includes mapped/shared pages; sampled private commit is not physical RSS. Whole-process peaks include loading, checkpoint and export.",
          "runs": []}
for phase, steps in (("first", 1), ("resumed", 1), ("full", 2)):
    output, checkpoint = work / (phase + ".gguf"), work / (phase + ".ckpt")
    metrics = evidence / ("real-" + phase + ".csv")
    command = [str(exe), "--mode", "cpt", "--base", str(base), "--rank", "2", "--data", str(corpus),
               "--output", str(output), "--checkpoint", str(checkpoint), "--metrics", str(metrics),
               "--context", "8", "--steps", str(steps), "--memory-mib", "512", "--accumulate", "3"]
    if phase == "resumed":
        command += ["--resume", str(work / "first.ckpt")]
    peak_working = peak_commit = samples = 0
    start = time.perf_counter()
    with (evidence / ("real-" + phase + ".log")).open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            counters = Counters(); counters.cb = ctypes.sizeof(counters)
            if memory_info(int(process._handle), ctypes.byref(counters), counters.cb):
                samples += 1
                peak_working = max(peak_working, counters.PeakWorkingSetSize)
                peak_commit = max(peak_commit, counters.PrivateUsage)
            try:
                code = process.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
    assert code == 0 and samples > 0, (phase, code, samples)
    with metrics.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == steps and all(int(row["microbatches"]) == 3 for row in rows)
    entry = {"phase": phase, "command": command, "process_seconds": time.perf_counter()-start,
             "peak_working_set_bytes": peak_working, "sampled_peak_private_commit_bytes": peak_commit,
             "memory_samples": samples, "model_sha256": sha(output), "checkpoint_sha256": sha(checkpoint), "metrics": rows}
    record["runs"].append(entry)
    if phase == "full":
        resumed = record["runs"][1]
        assert entry["model_sha256"] == resumed["model_sha256"]
        assert entry["checkpoint_sha256"] == resumed["checkpoint_sha256"]
        split = record["runs"][0]["metrics"] + resumed["metrics"]
        assert [row["loss"] for row in split] == [row["loss"] for row in rows]
    else:
        # Remove only this driver's own giant output after recording its hash.
        assert output.resolve().is_relative_to(work.resolve())
        output.unlink()
    (evidence / "real-accumulation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(phase, [(row["loss"], row["tokens"], row["graph_bytes"]) for row in rows],
          "peak working set", peak_working, flush=True)
