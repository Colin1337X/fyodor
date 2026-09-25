"""Supplement the completed matrix after strengthening one test assertion.
Production sources are unchanged; rebuild/run only the affected test target.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
source = root / "backend/tests/test_train_runner.c"
builds = [(name, root / name) for name in ("build-cpu", "build-cuda", "build-cuda-c23", "build-ort", "build-sanitize")]
for kind in ("optcpp", "cuda", "cutlass"):
    name = "verify-without-" + kind
    log = (out / (name + ".log")).read_text()
    matches = re.findall(r"Build files have been written to: ([^\r\n]+)", log)
    assert len(matches) == 1
    build = Path(matches[0]).resolve()
    expected_parent = root / ("build-cuda" if kind == "cutlass" else "build-cpu")
    assert build.is_relative_to(expected_parent.resolve()) and build.parent.name.startswith("without-" + kind + "-")
    copied_test = build.parent / "source/backend/tests/test_train_runner.c"
    assert copied_test.is_file()
    shutil.copyfile(source, copied_test)
    builds.append((name, build))

record = {"test_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "runs": []}
for name, build in builds:
    env = dict(os.environ)
    compiler = ".tools/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64/bin" if name == "build-sanitize" else ".tools/w64devkit/bin"
    env["PATH"] = str(root / compiler) + os.pathsep + env["PATH"]
    commands = [["cmake", "--build", str(build), "--target", "nya-test-train-runner", "--parallel", "4"],
                ["ctest", "--test-dir", str(build), "-R", "^train-accumulation$", "--output-on-failure"]]
    with (out / (name + "-rollback.log")).open("wb") as log:
        for command in commands:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["runs"].append({"name": name, "command": command, "exit_code": result.returncode})
            (out / "rollback-validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            assert result.returncode == 0, name
    print(name, "exact rollback passed", flush=True)
hashes = json.loads((out / "source-hashes.json").read_text())
hashes["backend/tests/test_train_runner.c"] = record["test_sha256"]
(out / "source-hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
