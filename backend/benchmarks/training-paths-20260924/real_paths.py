"""Resume real TinyLlama with every data/artifact path containing Unicode."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
work = root / "build-train-profile" / "real-paths-\ud55c\uae00-\U0001f43e"
work.mkdir(exist_ok=False)
base = work / "\ubaa8\ub378 \U0001f43e.gguf"
corpus = work / "\uc790\ub8cc.txt"
resume = work / "\uc774\uc804.ckpt"
output = work / "\uacb0\uacfc.gguf"
checkpoint = work / "\uc7ac\uac1c.ckpt"
metrics = work / "\uce21\uc815.csv"
# A new hard link avoids duplicating the immutable 669 MB base. No writes are
# made through either name. All owned artifacts live in the fresh work folder.
os.link(root / ".tools/tinyllama-q4_k_m.gguf", base)
shutil.copyfile(root / "backend/benchmarks/training-20260924/accumulation-corpus.txt", corpus)
shutil.copyfile(root / "build-train-profile/real-accumulation/first.ckpt", resume)
exe = root / "build-cpu/fyodor-train.exe"
command = [str(exe), "--mode", "cpt", "--base", str(base), "--rank", "2",
           "--data", str(corpus), "--resume", str(resume), "--output", str(output),
           "--checkpoint", str(checkpoint), "--metrics", str(metrics), "--steps", "1",
           "--context", "8", "--memory-mib", "512", "--accumulate", "3"]
start = time.perf_counter()
result = subprocess.run(command, env=dict(os.environ, NYA_COMPUTE="cpu"), capture_output=True,
                        text=True, encoding="utf-8", timeout=180)
elapsed = time.perf_counter() - start
(out / "real-paths.log").write_text(result.stdout + result.stderr, encoding="utf-8")
assert result.returncode == 0, result.stderr


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


record = {"command": command, "exit_code": result.returncode, "wall_seconds": elapsed,
          "base_sha256": sha(base), "model_sha256": sha(output), "checkpoint_sha256": sha(checkpoint),
          "trainer_sha256": sha(exe), "stdout": result.stdout, "metrics": metrics.read_text(encoding="utf-8")}
old = json.loads((root / "backend/benchmarks/training-20260924/real-accumulation.json").read_text(encoding="utf-8"))
reference = next(r for r in old["runs"] if r["phase"] == "full")
for key in ("model_sha256", "checkpoint_sha256"):
    assert record[key] == reference[key], key
assert record["base_sha256"] == old["base_sha256"]
(out / "real-paths.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
# Only this driver's owned large export is removed, after the exact comparison.
assert output.resolve().is_relative_to(work.resolve()) and output.suffix == ".gguf"
output.unlink()
print("Real TinyLlama Unicode-path resume: checkpoint and GGUF match ASCII-path reference exactly.")
