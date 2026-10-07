"""Capture exact source, build, executable and unchanged CUDA module identities."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=root, text=True).strip()


names = ["backend/CMakeLists.txt", "backend/core/training.c", "backend/core/training_parameter.h",
         "backend/core/training_session.c", "backend/core/training_session.h", "backend/core/pretraining.c",
         "backend/include/training.h", "backend/core/training_device.c", "backend/core/training_device.h",
         "backend/tests/test_training_session.c", "CUDA/training.cu", "CUDA/training.inc"]
header = (root / "build-cuda/cuda-provider/cuda_source.h").read_text()
bundles = {}
for name in ("nya_cuda_source", "nya_cuda_training_source"):
    values = re.search(re.escape(name) + r"\[\] = \{([^}]+)\}", header)[1]
    data = bytes(int(value, 0) for value in values.split(","))
    bundles[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
previous = json.loads((root / "backend/benchmarks/training-optimizer-20261006/snapshot.json").read_text())["generated_bundles"]
for name, bundle in bundles.items():
    assert bundle["sha256"] == previous[name]["sha256"], name
configs = {}
for build in ("build-cpu", "build-cuda", "build-cuda-c23", "build-ort", "build-sanitize"):
    configs[build] = {}
    for line in (root / build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("CMAKE_C_COMPILER:", "CMAKE_C_FLAGS:", "CMAKE_C_FLAGS_RELEASE:", "CMAKE_BUILD_TYPE:", "CMAKE_GENERATOR:", "NYA_")) and "=" in line:
            key, value = line.split("=", 1)
            configs[build][key] = value
    configs[build]["engine_flags"] = (root / build / "CMakeFiles/nya-engine.dir/flags.make").read_text()
record = {
    "base_revision": command("git", "rev-parse", "HEAD"),
    "source_sha256": {name: sha(root / name) for name in names},
    "build_configurations": configs,
    "generated_bundles": bundles,
    "cuda_change": "Both inference and training modules byte-identical to the optimizer milestone",
    "compiler": command(str(root / ".tools/w64devkit/bin/gcc.exe"), "--version"),
    "gpu": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
    "gpu_activity_observation": command("nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw", "--format=csv,noheader"),
    "executables_sha256": {name: sha(root / "build-cuda" / name) for name in
                           ("nya-test-training-session.exe", "fyodor-backend.exe")},
    "packaged_trainer_sha256": sha(root / "build-cpu/fyodor-train.exe"),
    "model_sha256": sha(root / ".tools/tinyllama-q4_k_m.gguf"),
    "method": {"steps": 40, "microbatches": 2, "resume_step": 10, "cpu_scaled_tolerance": 1e-6,
               "gpu_resume_comparison": "Byte-identical complete portable checkpoint",
               "timing_accepted": False, "scope": "Ownership and checkpoint bridge; public graph/CLI remain CPU"},
    "provenance": "Original Fyodor C code; no new external source adaptation"
}
(out / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Current source, builds and unchanged inference/training modules recorded.")
