"""Capture scratch-lifetime evidence; timings are explicitly not accepted."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=root, text=True).strip()


sources = ["CUDA/training.inc", "CUDA/training.cu", "CUDA/cuda.c",
           "backend/core/training_device.c", "backend/core/training_device.h",
           "backend/core/training_device_backend.h", "backend/tests/test_training_device.c",
           "backend/benchmarks/training_device_bench.c"]
model = root / ".tools/tinyllama-q4_k_m.gguf"
reference = root / ".tools/reference-llama-b10809"
upstream = ["LICENSE", "ggml/src/ggml-cuda/common.cuh",
            "ggml/src/ggml-cuda/ggml-cuda.cu", "ggml/src/ggml-cuda/mmf.cu"]
results = {}
for mode in ("forward", "reuse-forward", "reuse-reverse"):
    counters = dict((k, int(v)) for k, v in
                    (field.split("=") for field in (out / f"{mode}.log").read_text().split()))
    rows = [json.loads(line) for line in (out / f"{mode}.out").read_text().splitlines()]
    assert len(rows) == 18 and all(len(row["ms"]) == 5 for row in rows)
    results[mode] = {"counters": counters, "correctness_operations": len(rows),
                     "timing_accepted": False}
assert results["reuse-forward"]["counters"] == results["reuse-reverse"]["counters"]
for field in ("launches", "uploads", "upload_bytes", "downloads", "download_bytes"):
    assert results["forward"]["counters"][field] == results["reuse-forward"]["counters"][field]

record = {
    "revision": command("git", "rev-parse", "HEAD"),
    "os": platform.platform(),
    "gpu": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
    "compiler": command(str(root / ".tools/w64devkit/bin/gcc.exe"), "--version"),
    "source_sha256": {name: sha(root / name) for name in sources},
    "model": {"path": str(model.relative_to(root)), "bytes": model.stat().st_size, "sha256": sha(model)},
    "executable_sha256": sha(root / "build-cuda/nya-bench-training-device.exe"),
    "cmake_cache_sha256": sha(root / "build-cuda/CMakeCache.txt"),
    "environment": {"NYA_CUDA_BLAS": "0", "NYA_CUDA_CUTLASS": "0", "NYA_COMPUTE": "cpu"},
    "method": {"warmups": 2, "measured_batches": 5, "calls_per_batch": 20,
               "checked_elements_per_operation": 64,
               "timing_exclusion": "Desktop GPU activity was not isolated; resource and correctness evidence only."},
    "results": results,
    "source_study": {
        "project": "llama.cpp/ggml", "license": "MIT",
        "revision": command("git", "-C", str(reference), "rev-parse", "HEAD"),
        "sha256": {name: sha(reference / name) for name in upstream},
        "finding": "Pool selection is per device and stream. VMM free rewinds its offset in reverse allocation order. Scoped matrix temporaries use the selected stream and pool.",
        "adaptation": "Original Fyodor C implementation. No upstream source copied; bounded arena with one graph scope and globally unique expiring handles, rather than VMM growth or C++ RAII."
    }
}
(out / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Stored verified counters, source hashes, model identity and provenance.")
