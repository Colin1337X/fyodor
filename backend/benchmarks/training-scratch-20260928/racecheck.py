"""Instrument a bounded launch window of the full scratch lifetime test."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
tool = root / ".tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe"
exe = root / "build-cuda/nya-test-training-device.exe"
command = [str(tool), "--tool", "racecheck", "--launch-skip", "240", "--launch-count", "256",
           "--error-exitcode", "9", "--log-file", str(out / "racecheck-bounded.log"), str(exe), "--cuda"]
for name in ("racecheck-bounded.log", "racecheck-bounded.out", "racecheck-bounded.json"):
    assert not (out / name).exists(), "Preserve prior attempts instead of overwriting them"
env = dict(os.environ, NYA_CUDA_BLAS="0", NYA_CUDA_CUTLASS="0")
start = datetime.datetime.now(datetime.timezone.utc)
with (out / "racecheck-bounded.out").open("wb") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
record = {"command": command, "exit_code": result.returncode, "started_utc": start.isoformat(),
          "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
          "scope": "Full program executes; race instrumentation is limited to 256 launches after skipping 240. This covers initial scratch allocations and repeated reuse across all 67 allocation lengths, not all 5000 iterations."}
(out / "racecheck-bounded.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Bounded racecheck exit:", result.returncode, flush=True)
raise SystemExit(result.returncode)
