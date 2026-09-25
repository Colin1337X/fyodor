"""Native CUDA F32 accuracy-change cost, isolated from optional GEMM libraries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import argparse

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="f32-native", help="New evidence directory name")
args = parser.parse_args()
out = Path(__file__).resolve().parent / args.output
out.mkdir(exist_ok=False)
env = dict(os.environ, NYA_CUDA_BLAS="0", NYA_CUDA_CUTLASS="0", NYA_CPU_THREADS="6")
for key in ("NYA_CUDA_PROFILE", "NYA_CUDA_REFERENCE", "NYA_COMPUTE", "NYA_TEST_DECODE"):
    env.pop(key, None)
model = root / "build-train-profile/tinyllama-trained.gguf"
executables = {"before": root / ".tools/fyodor-bench-f32-native-baseline.exe",
               "after": root / "build-cuda/fyodor-bench.exe"}


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,pstate,clocks.sm,power.draw,temperature.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()


meta = {"model_sha256": sha(model), "executables": {name: {"path": str(path), "sha256": sha(path)} for name, path in executables.items()},
        "environment": {k: v for k, v in env.items() if k.startswith("NYA_")},
        "caveat": "Interactive desktop. Five measured repetitions and two warmups in each order; no claims of headless thermal isolation.", "runs": []}
for prompt in (8, 33, 128):
    for phase, order in (("a", ["before", "after"]), ("b", ["after", "before"])):
        for version in order:
            name = f"pp{prompt}-{phase}-{version}"
            command = [str(executables[version]), "-m", str(model), "-b", "cuda", "-p", str(prompt),
                       "-n", "16", "-r", "5", "--warmup", "2", "--json"]
            entry = {"name": name, "command": command, "gpu_before": gpu()}
            result = subprocess.run(command, env=env, capture_output=True, check=False)
            (out / (name + ".json")).write_bytes(result.stdout)
            (out / (name + ".log")).write_bytes(result.stderr)
            entry.update(exit_code=result.returncode, gpu_after=gpu())
            meta["runs"].append(entry)
            (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if result.returncode:
                raise RuntimeError(name)
            data = json.loads(result.stdout)
            assert data["execution"] == "resident" and data["kv_type"] == "f32"
            assert all(r["external_matmul_calls"] == 0 for r in data["results"])
            print(name, [(r["test"], r["tokens_per_second"], r["stddev"]) for r in data["results"]], flush=True)
