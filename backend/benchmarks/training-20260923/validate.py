"""Local validation driver; run after benchmarks, never concurrently with them."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import argparse

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
env = dict(os.environ)
env["PATH"] = str(root / ".tools/w64devkit/bin") + os.pathsep + env["PATH"]
env["NYA_CUDA_BLAS_LIBRARY"] = str(root / ".tools/llama-b10809-cuda/cublas64_13.dll")
env["NYA_CUDA_CUTLASS_LIBRARY"] = str(root / "build-cutlass/fyodor-cutlass.dll")
for key in ("NYA_TRAIN_PROFILE", "NYA_COMPUTE", "NYA_INFERENCE_REFERENCE", "NYA_CPU_ISA", "NYA_CPU_PROFILE", "NYA_CUDA_PROFILE"):
    env.pop(key, None)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resume", action="store_true", help="Continue failed driver without rerunning passed stages; use only with unchanged sources")
parser.add_argument("--output", type=Path, help="Store a new validation snapshot in this directory")
args = parser.parse_args()
if args.output:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
results = json.loads((out / "validation.json").read_text(encoding="utf-8")) if args.resume else []
results = [entry for entry in results if entry["exit_code"] == 0]


def run(name, command, runenv=env, cwd=root):
    if any(entry["name"] == name and entry["command"] == command for entry in results):
        print(name, "previously passed", flush=True)
        return
    # Preserve each earlier pass/failure instead of overwriting its evidence.
    log_path = out / (name + ".log")
    attempt = 1
    while log_path.exists():
        log_path = out / f"{name}-{attempt}.log"
        attempt += 1
    with log_path.open("wb") as log:
        result = subprocess.run(command, env=runenv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    results.append({"name": name, "log": log_path.name, "command": command, "exit_code": result.returncode})
    (out / "validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed; preserved log")


for build in ("build-cpu", "build-cuda", "build-cuda-c23", "build-ort", "build-sanitize"):
    runenv = env.copy()
    if build == "build-sanitize":
        runenv["PATH"] = str(root / ".tools/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64/bin") + os.pathsep + runenv["PATH"]
    run(build + "-build", ["cmake", "--build", build, "--parallel", "4"], runenv)
    run(build + "-tests", ["ctest", "--test-dir", build, "--output-on-failure"], runenv)
for build, target in (("build-cpu", "verify-without-optcpp"), ("build-cpu", "verify-without-cuda"),
                      ("build-cuda", "verify-without-cutlass")):
    run(target, ["cmake", "--build", build, "--target", target])
node = "C:/Program Files/nodejs/node.exe"
run("frontend-tests", [node, "--test", "frontend/tests/ui.test.mjs", "frontend/tests/sidecar.test.mjs"])
run("frontend-build", [node, "node_modules/vite/bin/vite.js", "build"], cwd=root / "frontend")
hashes = {}
for name in ("backend/core/training.c", "backend/core/train_main.c", "backend/core/train_clock.h", "backend/core/train_control.h",
             "backend/tests/train_stop.py", "backend/tests/train_eval.py", "backend/tests/test_architectures.c", "backend/core/llm_quant.c",
             "backend/core/file.c", "backend/core/file.h", "backend/tests/train_paths.py",
             "backend/core/train_executor.c", "backend/core/train_executor.h", "backend/include/training.h",
             "backend/core/llm_cpu.c", "backend/tests/test_pretraining.c", "CUDA/gemm.cu",
             "backend/tests/test_train_runner.c", "frontend/src-tauri/src/lib.rs", "frontend/src-tauri/Cargo.toml",
             "frontend/src-tauri/Cargo.lock",
             "backend/tests/test_training.c", "backend/tests/TrainCli.cmake", "backend/CMakeLists.txt",
             "frontend/src/main.js", "frontend/src/training-metrics.js", "frontend/tests/ui.test.mjs"):
    hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(out / "source-hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
