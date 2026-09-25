"""Real TinyLlama stop/save/resume correctness; not a throughput benchmark."""
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
work = root / "build-train-profile/real-stop"
work.mkdir(exist_ok=False)
exe = root / "build-cpu/fyodor-train.exe"
base = root / ".tools/tinyllama-q4_k_m.gguf"
corpus = root / "backend/benchmarks/training-20260924/accumulation-corpus.txt"
env = dict(os.environ, NYA_COMPUTE="cpu")
records = []


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(name, steps, stop=False, resume=None):
    model, checkpoint = (work / (name + ext) for ext in (".gguf", ".ckpt"))
    metrics = out / (name + ".csv")
    command = [str(exe), "--mode", "cpt", "--base", str(base), "--rank", "2", "--data", str(corpus),
               "--context", "8", "--memory-mib", "512", "--accumulate", "3", "--steps", str(steps),
               "--output", str(model), "--checkpoint", str(checkpoint), "--metrics", str(metrics)]
    if resume:
        command += ["--resume", str(resume)]
    start = time.perf_counter()
    latency = None
    if stop:
        command += ["--control-stdin", "1"]
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            ready = queue.Queue()
            threading.Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True).start()
            first = ready.get(timeout=120)
            assert first.startswith("step=1 "), first
            requested = time.perf_counter()
            child.stdin.write("S")
            child.stdin.flush()
            stdout, stderr = child.communicate(timeout=180)
            latency = time.perf_counter() - requested
            stdout = first + stdout
            code = child.returncode
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    else:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    elapsed = time.perf_counter() - start
    (out / (name + ".log")).write_text(stdout + stderr, encoding="utf-8")
    assert code == 0, stderr
    with metrics.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    record = {"name": name, "command": command, "exit_code": code, "stdout": stdout,
              "metrics": rows, "wall_seconds": elapsed, "stop_to_exit_seconds": latency,
              "model_sha256": sha(model), "checkpoint_sha256": sha(checkpoint)}
    records.append(record)
    (out / "real-stop.json").write_text(json.dumps({"base_sha256": sha(base), "trainer_sha256": sha(exe),
        "corpus_sha256": sha(corpus), "runs": records}, indent=2), encoding="utf-8")
    print(name, len(rows), "updates; stop latency", latency, flush=True)
    # The test owns these new large exports. Preserve hashes/checkpoints and
    # remove only the exact export beneath this driver's verified directory.
    assert model.resolve().is_relative_to(work.resolve()) and model.suffix == ".gguf"
    model.unlink()
    return record, checkpoint


stopped, checkpoint = run("stopped", 100, stop=True)
count = len(stopped["metrics"])
assert 1 <= count < 100
resumed, _ = run("resumed", 1, resume=checkpoint)
full, _ = run("continuous", count + 1)
for key in ("model_sha256", "checkpoint_sha256"):
    assert resumed[key] == full[key], key
assert [r["loss"] for item in (stopped, resumed) for r in item["metrics"]] == [r["loss"] for r in full["metrics"]]
reference = json.loads((root / "backend/benchmarks/training-20260924/real-accumulation.json").read_text(encoding="utf-8"))
if count in (1, 2):
    old = next(r for r in reference["runs"] if r["phase"] == ("first" if count == 1 else "full"))
    for key in ("model_sha256", "checkpoint_sha256"):
        assert stopped[key] == old[key], key
print("Real model stop and resume: exact losses, checkpoint and export", flush=True)
