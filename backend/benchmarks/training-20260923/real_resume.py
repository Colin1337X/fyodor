"""Real TinyLlama LoRA split-run equality against the saved two-update baseline.
This is a correctness test, not a timing measurement.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
work = root / "build-train-profile/real-resume"
work.mkdir(exist_ok=False)
corpus = evidence / "tinyllama-corpus.txt"
corpus.write_bytes(b"abcabc\r\n")
env = dict(os.environ, NYA_COMPUTE="cpu")
env.pop("NYA_TRAIN_PROFILE", None)
results = []


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for phase in ("first", "resumed"):
    output, checkpoint = work / (phase + ".gguf"), work / (phase + ".ckpt")
    command = [str(root / "build-cpu/fyodor-train.exe"), "--mode", "cpt", "--base",
               str(root / ".tools/tinyllama-q4_k_m.gguf"), "--rank", "2", "--data", str(corpus),
               "--output", str(output), "--checkpoint", str(checkpoint), "--context", "8", "--steps", "1", "--memory-mib", "512"]
    if phase == "resumed":
        command.extend(["--resume", str(work / "first.ckpt")])
    with (evidence / ("real-resume-" + phase + ".log")).open("wb") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    results.append({"phase": phase, "command": command, "model_sha256": sha(output), "checkpoint_sha256": sha(checkpoint)})
    if phase == "resumed":
        assert results[-1]["model_sha256"] == sha(root / "build-train-profile/tinyllama-trained.gguf")
        assert results[-1]["checkpoint_sha256"] == sha(root / "build-train-profile/tinyllama-trained.ckpt")
    output.unlink()
    print(phase, "passed", flush=True)
(evidence / "real-resume.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
