"""Random-init accumulation -> restart/resume -> GGUF -> actual HTTP generation."""
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import urllib.request

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
work = root / "build-train-profile/accumulation-lifecycle"
work.mkdir(exist_ok=False)
corpus = work / "corpus.txt"
corpus.write_text("abc" * 20, encoding="utf-8")
env = dict(os.environ, NYA_COMPUTE="cpu")
records = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for phase, steps in (("first", 40), ("resumed", 60), ("full", 100)):
    command = [str(root / "build-cpu/fyodor-train.exe"), "--data", str(corpus),
               "--output", str(work / (phase + ".gguf")), "--checkpoint", str(work / (phase + ".ckpt")),
               "--metrics", str(work / (phase + ".csv")), "--steps", str(steps), "--lr", "0.015", "--accumulate", "3",
               "--dimension", "16", "--ff", "32", "--layers", "1", "--heads", "2", "--kv-heads", "1", "--context", "16"]
    if phase == "resumed":
        command += ["--resume", str(work / "first.ckpt")]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (out / ("lifecycle-" + phase + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    with (work / (phase + ".csv")).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records.append({"phase": phase, "command": command, "metrics": rows,
                    "model_sha256": sha(work / (phase + ".gguf")), "checkpoint_sha256": sha(work / (phase + ".ckpt"))})
assert records[1]["model_sha256"] == records[2]["model_sha256"]
assert records[1]["checkpoint_sha256"] == records[2]["checkpoint_sha256"]
assert [r["loss"] for entry in records[:2] for r in entry["metrics"]] == [r["loss"] for r in records[2]["metrics"]]
first, last = float(records[2]["metrics"][0]["loss"]), float(records[2]["metrics"][-1]["loss"])
assert first > 5 and last < 0.02, (first, last)

command = [str(root / "build-cpu/fyodor-backend.exe"), "--port", "0", "--bind", "127.0.0.1"]
with (out / "lifecycle-server.log").open("wb") as log:
    server = subprocess.Popen(command, env=env, cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log, text=True)
    try:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(server.stdout.readline()), daemon=True).start()
        fields = ready.get(timeout=10).split()
        assert len(fields) == 4 and fields[:2] == ["FYODOR_READY", "127.0.0.1"]
        # The ephemeral authentication token stays in memory, never in evidence.
        endpoint = "http://127.0.0.1:" + fields[2]

        def request(path, body):
            req = urllib.request.Request(endpoint + path, data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + fields[3], "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.load(response)

        loaded = request("/api/v1/model/load", {"path": str(work / "resumed.gguf")})
        completion = request("/v1/completions", {"model": str(loaded["model"]["id"]), "prompt": "a", "temperature": 0, "max_tokens": 8})
        assert completion["choices"][0]["text"] == "bcabcabc", completion
        request("/api/v1/shutdown", {})
        assert server.wait(timeout=10) == 0
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
record = {"runs": records, "server_command": command, "completion": completion}
(out / "lifecycle.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(f"Accumulated loss {first:.9g} -> {last:.9g}; exact restart/resume; HTTP completion: bcabcabc", flush=True)
